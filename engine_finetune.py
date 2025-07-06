# 文件: engine_finetune.py (版本 36 - 阈值分析)
# 核心改动:
# 1. (FEATURE) 重构 `evaluate` 函数以进行全面的性能分析。
# 2. (FEATURE) 函数现在首先对所有验证序列进行一次推理，收集原始预测结果。
# 3. (FEATURE) 然后，它会遍历一系列预定义的置信度阈值 (0.1 to 0.9)。
# 4. (FEATURE) 对每个阈值，它都会重新计算完整的指标（召回率、虚警率、稳定性），
#    从而能够清晰地展示阈值对性能的影响。
# 5. (REFACTOR) 函数现在返回一个包含每个阈值结果的列表，而不再是单个字典。

import torch
import torch.nn.functional as F
import util.misc as misc
from util.visualize import save_eval_video
import math
import sys
from typing import Iterable
import numpy as np
import os


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
    model.eval();
    metric_logger = misc.MetricLogger(delimiter="  ");
    header = 'Test:'

    # NEW: Step 1 - Collect all predictions and ground truths from the dataset
    all_sequence_data = []
    print("Collecting predictions from all validation sequences...")
    for seq_name, sequence_cpu, targets in metric_logger.log_every(data_loader, 10, header):
        sequence_gpu = sequence_cpu.to(device, non_blocking=True);
        sequence_len = sequence_gpu.shape[0]

        all_pred_logits, all_pred_boxes, all_offsets = [], [], []

        # Process sequence in clips
        for i in range(0, sequence_len, model.clip_length):
            clip = sequence_gpu[i:min(i + model.clip_length, sequence_len)]
            if clip.dim() == 4:  # Handle last clip if not full
                clip = clip.unsqueeze(0)

            outputs = model(clip)
            all_pred_logits.append(outputs['hm'].squeeze(0));
            all_pred_boxes.append(outputs['wh'].squeeze(0));
            all_offsets.append(outputs['offset'].squeeze(0))

        seq_pred_heat = torch.cat(all_pred_logits, dim=0);
        seq_pred_wh = torch.cat(all_pred_boxes, dim=0);
        seq_pred_offset = torch.cat(all_offsets, dim=0)

        # Decode heatmaps to get raw bounding box predictions (with scores)
        detections = centernet_decode(seq_pred_heat.unsqueeze(0), seq_pred_wh.unsqueeze(0),
                                      seq_pred_offset.unsqueeze(0), K=100).squeeze(0)

        all_sequence_data.append({'preds': detections, 'gts': targets, 'seq_len': sequence_len})

    # NEW: Step 2 - Evaluate metrics across a range of confidence thresholds
    print("Evaluating collected predictions across multiple thresholds...")
    thresholds_to_test = np.arange(0.1, 0.91, 0.05)
    dist_threshold, stability_threshold = 3.0, 0.8
    results_by_threshold = []

    for conf_threshold in thresholds_to_test:
        stats_by_class = {i: {'tp': 0, 'fp': 0, 'fn': 0} for i in range(model.num_classes)}
        stable_sequences, total_sequences = 0, len(all_sequence_data)

        for seq_data in all_sequence_data:
            detections, targets, sequence_len = seq_data['preds'], seq_data['gts'], seq_data['seq_len']
            consistent_frames = 0

            for t in range(sequence_len):
                preds_t = detections[t];
                scores = preds_t[:, 4];
                keep = scores > conf_threshold;
                preds_t_filtered = preds_t[keep]

                gt_boxes = targets[t]['boxes'].to(device);
                gt_labels = targets[t]['labels'].to(device)

                gt_matched = torch.zeros(gt_boxes.shape[0], device=device, dtype=torch.bool)
                pred_matched = torch.zeros(preds_t_filtered.shape[0], device=device, dtype=torch.bool)

                if gt_boxes.shape[0] > 0 and preds_t_filtered.shape[0] > 0:
                    # Convert normalized coords to feature map scale for distance calculation
                    output_res = seq_pred_heat.shape[-1]
                    pred_centers_feat_scale = preds_t_filtered[:, :2] * output_res
                    gt_centers_feat_scale = gt_boxes[:, :2] * output_res

                    dist_matrix = torch.cdist(pred_centers_feat_scale, gt_centers_feat_scale)

                    if dist_matrix.numel() > 0:
                        closest_pred_dist, closest_pred_idx = dist_matrix.min(dim=0)
                        for gt_idx, pred_idx in enumerate(closest_pred_idx):
                            pred_class = preds_t_filtered[pred_idx, 5].int().item()
                            gt_class = gt_labels[gt_idx].item()
                            if closest_pred_dist[gt_idx] < dist_threshold and pred_class == gt_class and not \
                            pred_matched[pred_idx]:
                                stats_by_class[gt_class]['tp'] += 1;
                                gt_matched[gt_idx] = True;
                                pred_matched[pred_idx] = True

                if gt_matched.any(): consistent_frames += 1

                # Tally FPs and FNs for this frame
                num_fp_frame = preds_t_filtered.shape[0] - pred_matched.sum().item()
                for i in range(preds_t_filtered.shape[0]):
                    if not pred_matched[i]:
                        stats_by_class[preds_t_filtered[i, 5].int().item()]['fp'] += 1

                num_fn_frame = gt_boxes.shape[0] - gt_matched.sum().item()
                for i in range(gt_boxes.shape[0]):
                    if not gt_matched[i]:
                        stats_by_class[gt_labels[i].item()]['fn'] += 1

            if (
            consistent_frames / sequence_len if sequence_len > 0 else 0) > stability_threshold: stable_sequences += 1

        # Aggregate and calculate final metrics for the current threshold
        total_tp = sum(stats['tp'] for stats in stats_by_class.values());
        total_fp = sum(stats['fp'] for stats in stats_by_class.values());
        total_fn = sum(stats['fn'] for stats in stats_by_class.values())

        avg_recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
        # NEW: False Alarm Rate is FP / (FP + TP)
        avg_false_alarm = total_fp / (total_fp + total_tp) if (total_fp + total_tp) > 0 else 0.0
        spatial_stability = stable_sequences / total_sequences if total_sequences > 0 else 0.0

        results_by_threshold.append({
            'threshold': conf_threshold,
            'avg_recall': avg_recall,
            'avg_false_alarm': avg_false_alarm,
            'spatial_stability': spatial_stability,
            'tp': total_tp, 'fp': total_fp, 'fn': total_fn
        })

    misc.synchronize_between_processes()
    return results_by_threshold


@torch.no_grad()
def visualize_epoch(model, data_loader, device, output_dir, epoch, vis_folder_name):
    model.eval();
    vis_dir = os.path.join(output_dir, vis_folder_name);
    os.makedirs(vis_dir, exist_ok=True);
    header = f'Visualizing {vis_folder_name}:';
    metric_logger = misc.MetricLogger(delimiter="  ")
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

        heat = torch.cat(all_pred_logits, dim=0).unsqueeze(0);
        wh = torch.cat(all_pred_boxes, dim=0).unsqueeze(0);
        offset = torch.cat(all_offsets, dim=0).unsqueeze(0)

        detections = centernet_decode(heat, wh, offset, K=100).squeeze(0)

        preds_for_vis = []
        for t in range(sequence_len):
            preds_t = detections[t]
            scores = preds_t[:, 4];
            keep = scores > 0.3
            preds_for_vis.append({'boxes': preds_t[keep, :4], 'scores': scores[keep], 'labels': preds_t[keep, 5].int()})

        video_path = os.path.join(vis_dir, f"{seq_name}_epoch_{epoch}.mp4");
        save_eval_video(sequence_cpu, gt_targets, preds_for_vis, video_path)