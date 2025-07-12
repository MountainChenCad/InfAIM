# 文件: engine_finetune.py (版本 37 - IoU阈值评估)
# 核心改动:
# 1. (FEATURE) 添加IoU计算函数，支持中心点坐标格式(cx, cy, w, h)
# 2. (REFACTOR) 将评估逻辑从距离阈值改为IoU阈值(>0.3)
# 3. (REFACTOR) 更新匹配逻辑，使用IoU进行真正例/假正例判断
# 4. (FEATURE) 规范可视化命名格式，保持置信度扫描功能
# 5. (OPTIMIZATION) 保持模型结构不变，确保训练连续性

import torch
import torch.nn.functional as F
import util.misc as misc
from util.visualize import save_eval_video
import math
import sys
from typing import Iterable
import numpy as np
import os


def box_cxcywh_to_xyxy(boxes):
    """Convert center format (cx, cy, w, h) to corner format (x1, y1, x2, y2)"""
    cx, cy, w, h = boxes.unbind(-1)
    x1 = cx - w / 2
    y1 = cy - h / 2
    x2 = cx + w / 2
    y2 = cy + h / 2
    return torch.stack([x1, y1, x2, y2], dim=-1)


def compute_iou(boxes1, boxes2):
    """
    Compute IoU between two sets of boxes
    boxes1: (N, 4) in format (x1, y1, x2, y2)
    boxes2: (M, 4) in format (x1, y1, x2, y2)
    Returns: (N, M) IoU matrix
    """
    area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
    area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])

    lt = torch.max(boxes1[:, None, :2], boxes2[:, :2])  # (N, M, 2)
    rb = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])  # (N, M, 2)

    wh = (rb - lt).clamp(min=0)  # (N, M, 2)
    inter = wh[:, :, 0] * wh[:, :, 1]  # (N, M)

    union = area1[:, None] + area2 - inter
    iou = inter / union.clamp(min=1e-6)
    return iou


# ... (Loss Functions, train_one_epoch, Post-processing Functions remain unchanged) ...
def _neg_loss(pred, gt):  # Focal Loss
    pos_inds = gt.eq(1).float();
    neg_inds = gt.lt(1).float();
    neg_weights = torch.pow(1 - gt, 4);
    loss = 0
    pos_loss = torch.log(pred) * torch.pow(1 - pred, 2) * pos_inds;
    neg_loss = torch.log(1 - pred) * torch.pow(pred, 2) * neg_weights * neg_inds
    num_pos = pos_inds.float().sum();
    pos_loss = pos_loss.sum();
    neg_loss = neg_loss.sum()
    if num_pos == 0:
        loss = loss - neg_loss
    else:
        loss = loss - (pos_loss + neg_loss) / num_pos
    return loss


def _reg_l1_loss(output, mask, index, target):  # L1 Loss for WH and Offset
    pred = _transpose_and_gather_feat(output, index);
    mask = mask.unsqueeze(2).expand_as(pred).float()
    loss = F.l1_loss(pred * mask, target * mask, reduction='sum');
    loss = loss / (mask.sum() + 1e-4)
    return loss


def _transpose_and_gather_feat(feat, ind):
    feat = feat.permute(0, 2, 3, 1).contiguous();
    feat = feat.view(feat.size(0), -1, feat.size(3));
    feat = _gather_feat(feat, ind)
    return feat


def _gather_feat(feat, ind, mask=None):
    dim = feat.size(2);
    ind = ind.unsqueeze(2).expand(ind.size(0), ind.size(1), dim);
    feat = feat.gather(1, ind)
    if mask is not None: mask = mask.unsqueeze(2).expand_as(feat); feat = feat[mask]; feat = feat.view(-1, dim)
    return feat


def train_one_epoch(model: torch.nn.Module, data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, log_writer=None, args=None):
    model.train();
    metric_logger = misc.MetricLogger(delimiter="  ");
    metric_logger.add_meter('lr', misc.SmoothedValue(window_size=1, fmt='{value:.6f}'));
    header = f'Epoch: [{epoch}]'
    for data_iter_step, (clips, targets) in enumerate(metric_logger.log_every(data_loader, 20, header)):
        clips = clips.to(device, non_blocking=True)
        # NEW: a more robust way to handle collated targets dictionary
        for key in targets: targets[key] = targets[key].to(device, non_blocking=True)

        # Reshape targets to be (B*T, C, H, W) to match predictions
        b, t, c, h, w = targets['hm'].shape
        targets_reshaped = {k: v.view(b * t, *v.shape[2:]) for k, v in targets.items()}

        outputs = model(clips)

        # Reshape predictions to be (B*T, C, H, W)
        b_out, t_out, c_out_hm, h_out, w_out = outputs['hm'].shape
        hm_pred = outputs['hm'].view(b_out * t_out, c_out_hm, h_out, w_out)
        wh_pred = outputs['wh'].view(b_out * t_out, -1, h_out, w_out)
        offset_pred = outputs['offset'].view(b_out * t_out, -1, h_out, w_out)

        hm_gt = targets_reshaped['hm']
        wh_gt = targets_reshaped['wh']
        offset_gt = targets_reshaped['offset']
        ind_gt = targets_reshaped['ind']
        ind_mask_gt = targets_reshaped['ind_mask']

        hm_loss = _neg_loss(hm_pred, hm_gt);
        wh_loss = _reg_l1_loss(wh_pred, ind_mask_gt, ind_gt, wh_gt);
        offset_loss = _reg_l1_loss(offset_pred, ind_mask_gt, ind_gt, offset_gt)
        loss = hm_loss + 0.1 * wh_loss + offset_loss
        if not math.isfinite(loss.item()): print(f"Loss is {loss.item()}, stopping training"); sys.exit(1)
        optimizer.zero_grad();
        loss.backward();
        optimizer.step()
        metric_logger.update(loss=loss.item(), hm_loss=hm_loss.item(), wh_loss=wh_loss.item(),
                             off_loss=offset_loss.item());
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
    metric_logger.synchronize_between_processes();
    print("Averaged stats:", metric_logger);
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


def _nms(heat, kernel=3):
    pad = (kernel - 1) // 2;
    hmax = F.max_pool2d(heat, (kernel, kernel), stride=1, padding=pad);
    keep = (hmax == heat).float()
    return heat * keep


def _topk(scores, K=40):
    batch, cat, height, width = scores.size();
    topk_scores, topk_inds = torch.topk(scores.view(batch, cat, -1), K)
    topk_inds = topk_inds % (height * width);
    topk_ys = (topk_inds / width).int().float();
    topk_xs = (topk_inds % width).int().float()
    topk_score, topk_ind = torch.topk(topk_scores.view(batch, -1), K);
    topk_clses = (topk_ind / K).int()
    topk_inds = _gather_feat(topk_inds.view(batch, -1, 1), topk_ind).view(batch, K);
    topk_ys = _gather_feat(topk_ys.view(batch, -1, 1), topk_ind).view(batch, K)
    topk_xs = _gather_feat(topk_xs.view(batch, -1, 1), topk_ind).view(batch, K)
    return topk_score, topk_inds, topk_clses, topk_ys, topk_xs


def centernet_decode(heat, wh, reg, K=100):
    batch, cat, height, width = heat.size();
    heat = _nms(heat);
    scores, inds, clses, ys, xs = _topk(heat, K=K)
    if reg is not None:
        reg = _transpose_and_gather_feat(reg, inds);
        reg = reg.view(batch, K, 2);
        xs = xs.view(batch, K, 1) + reg[:, :,
                                    0:1];
        ys = ys.view(
            batch, K, 1) + reg[:, :, 1:2]
    else:
        xs = xs.view(batch, K, 1) + 0.5;
        ys = ys.view(batch, K, 1) + 0.5
    wh = _transpose_and_gather_feat(wh, inds);
    wh = wh.view(batch, K, 2);
    clses = clses.view(batch, K, 1).float();
    scores = scores.view(batch, K, 1)
    # convert to xywh
    xs = xs / width
    ys = ys / height
    wh[..., 0] = wh[..., 0] / width
    wh[..., 1] = wh[..., 1] / height

    bboxes = torch.cat([xs, ys, wh], dim=2)
    detections = torch.cat([bboxes, scores, clses], dim=2)
    return detections


@torch.no_grad()
def evaluate(model, data_loader, device, output_dir, epoch):
    model.eval()
    metric_logger = misc.MetricLogger(delimiter="  ")
    header = 'Test:'

    # Step 1: Collect all raw predictions and ground truths from the dataset
    all_sequence_data = []
    print("Collecting predictions from all validation sequences...")
    for seq_name, sequence_cpu, targets in metric_logger.log_every(data_loader, 10, header):
        sequence_gpu = sequence_cpu.to(device, non_blocking=True)
        sequence_len = sequence_gpu.shape[0]

        all_pred_logits, all_pred_boxes, all_offsets = [], [], []
        # Process sequence in clips
        for i in range(0, sequence_len, model.clip_length):
            clip = sequence_gpu[i:min(i + model.clip_length, sequence_len)]
            if clip.dim() == 4:  # Handle last clip if not full
                clip = clip.unsqueeze(0)

            outputs = model(clip)
            all_pred_logits.append(outputs['hm'].squeeze(0))
            all_pred_boxes.append(outputs['wh'].squeeze(0))
            all_offsets.append(outputs['offset'].squeeze(0))

        seq_pred_heat = torch.cat(all_pred_logits, dim=0)
        seq_pred_wh = torch.cat(all_pred_boxes, dim=0)
        seq_pred_offset = torch.cat(all_offsets, dim=0)

        detections = centernet_decode(seq_pred_heat, seq_pred_wh, seq_pred_offset, K=100)

        all_sequence_data.append({
            'preds': detections,
            'gts': targets,
            'seq_len': sequence_len,
            'seq_name': seq_name
        })

    # Step 2: Initialize metric counters for each threshold
    print("Evaluating metrics across all frames and thresholds...")
    thresholds_to_test = np.arange(0.1, 0.91, 0.05)
    iou_threshold, stability_threshold = 0.3, 0.8

    stats_by_thresh = {f"{th:.2f}": {'tp': 0, 'fp': 0, 'fn': 0, 'stable_sequences': 0} for th in thresholds_to_test}
    total_sequences = len(all_sequence_data)

    # Step 3: Iterate through sequences and frames ONCE
    for seq_data in metric_logger.log_every(all_sequence_data, 1, "Processing sequences:"):
        detections, targets, sequence_len = \
            seq_data['preds'], seq_data['gts'], seq_data['seq_len']

        consistent_frames_by_thresh = {f"{th:.2f}": 0 for th in thresholds_to_test}

        for t in range(sequence_len):
            preds_t = detections[t]
            gt_boxes = targets[t]['boxes'].to(device)
            gt_labels = targets[t]['labels'].to(device)

            # Convert boxes to xyxy format for IoU calculation
            if gt_boxes.shape[0] > 0:
                gt_boxes_xyxy = box_cxcywh_to_xyxy(gt_boxes)
            else:
                gt_boxes_xyxy = torch.empty((0, 4), device=device)

            for conf_threshold in thresholds_to_test:
                key = f"{conf_threshold:.2f}"

                pred_indices = (preds_t[:, 4] > conf_threshold).nonzero(as_tuple=True)[0]

                gt_matched = torch.zeros(gt_boxes.shape[0], device=device, dtype=torch.bool)
                pred_matched_indices = torch.zeros(preds_t.shape[0], device=device, dtype=torch.bool)

                if gt_boxes.shape[0] > 0 and pred_indices.shape[0] > 0:
                    # Get prediction boxes and convert to xyxy format
                    pred_boxes_filtered = preds_t[pred_indices, :4]
                    pred_boxes_xyxy = box_cxcywh_to_xyxy(pred_boxes_filtered)

                    # Compute IoU matrix
                    iou_matrix = compute_iou(pred_boxes_xyxy, gt_boxes_xyxy)

                    # Find best matches
                    max_iou_per_gt, best_pred_idx_per_gt = iou_matrix.max(dim=0)

                    for gt_idx, (best_pred_idx, max_iou) in enumerate(zip(best_pred_idx_per_gt, max_iou_per_gt)):
                        if max_iou > iou_threshold:
                            original_pred_idx = pred_indices[best_pred_idx]
                            pred_class = preds_t[original_pred_idx, 5].int().item()
                            gt_class = gt_labels[gt_idx].item()

                            if pred_class == gt_class and not pred_matched_indices[original_pred_idx]:
                                stats_by_thresh[key]['tp'] += 1
                                gt_matched[gt_idx] = True
                                pred_matched_indices[original_pred_idx] = True

                if gt_matched.any():
                    consistent_frames_by_thresh[key] += 1

                stats_by_thresh[key]['fp'] += pred_indices.shape[0] - gt_matched.sum().item()
                stats_by_thresh[key]['fn'] += gt_boxes.shape[0] - gt_matched.sum().item()

        for conf_threshold in thresholds_to_test:
            key = f"{conf_threshold:.2f}"
            if (consistent_frames_by_thresh[key] / sequence_len if sequence_len > 0 else 0) > stability_threshold:
                stats_by_thresh[key]['stable_sequences'] += 1

    # Step 4: Finalize metrics
    results_by_threshold = []
    for conf_threshold in thresholds_to_test:
        key = f"{conf_threshold:.2f}"
        stats = stats_by_thresh[key]
        total_tp, total_fp, total_fn = stats['tp'], stats['fp'], stats['fn']

        avg_recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
        avg_false_alarm = total_fp / (total_fp + total_tp) if (total_fp + total_tp) > 0 else 0.0
        spatial_stability = stats['stable_sequences'] / total_sequences if total_sequences > 0 else 0.0

        results_by_threshold.append({
            'threshold': conf_threshold, 'avg_recall': avg_recall, 'avg_false_alarm': avg_false_alarm,
            'spatial_stability': spatial_stability, 'tp': total_tp, 'fp': total_fp, 'fn': total_fn
        })

    # Step 5: Correctly synchronize processes
    if misc.is_dist_avail_and_initialized():
        torch.distributed.barrier()

    return results_by_threshold


@torch.no_grad()
def visualize_epoch(model, data_loader, device, output_dir, epoch, vis_folder_name):
    model.eval();
    vis_dir = os.path.join(output_dir, vis_folder_name);
    os.makedirs(vis_dir, exist_ok=True);
    header = f'Visualizing {vis_folder_name}:';
    metric_logger = misc.MetricLogger(delimiter="  ")

    # Add sequence counter to avoid naming conflicts
    seq_counter = 0

    for seq_name_tuple, sequence_cpu, gt_targets in metric_logger.log_every(data_loader, 1, header):
        seq_name = seq_name_tuple[0]  # Dataloader wraps it in a tuple/list
        sequence_gpu = sequence_cpu.to(device);
        sequence_len = sequence_gpu.shape[0]

        all_pred_logits, all_pred_boxes, all_offsets = [], [], []

        for i in range(0, sequence_len, model.clip_length):
            clip = sequence_gpu[i:min(i + model.clip_length, sequence_len)]
            if clip.dim() == 4:
                clip = clip.unsqueeze(0)
            outputs = model(clip)
            all_pred_logits.append(outputs['hm'].squeeze(0));
            all_pred_boxes.append(outputs['wh'].squeeze(0));
            all_offsets.append(outputs['offset'].squeeze(0))

        # --- Corrected Code ---
        heat = torch.cat(all_pred_logits, dim=0)
        wh = torch.cat(all_pred_boxes, dim=0)
        offset = torch.cat(all_offsets, dim=0)

        detections = centernet_decode(heat, wh, offset, K=100)

        preds_for_vis = []
        for t in range(sequence_len):
            preds_t = detections[t]
            scores = preds_t[:, 4];
            keep = scores > 0.3
            preds_for_vis.append({'boxes': preds_t[keep, :4], 'scores': scores[keep], 'labels': preds_t[keep, 5].int()})

        # Enhanced naming format to avoid conflicts
        # Format: {folder_type}_{sequence_counter:03d}_{clean_seq_name}_epoch_{epoch:02d}.mp4
        folder_type = vis_folder_name.split('_')[0]  # 'val' or 'train'
        clean_seq_name = seq_name.replace('/', '_').replace('\\', '_')[:20]  # Clean and limit length
        video_filename = f"{folder_type}_{seq_counter:03d}_{clean_seq_name}_epoch_{epoch:02d}.mp4"
        video_path = os.path.join(vis_dir, video_filename)

        save_eval_video(sequence_cpu, gt_targets, preds_for_vis, video_path)

        seq_counter += 1