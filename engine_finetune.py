# 文件: engine_finetune.py (版本 35 - 修复评估中的拆包错误)
# 核心改动:
# 1. (BUG FIX) 修复了在 `evaluate` 函数中由于一个简单的拼写错误而导致的 `ValueError`。
# 2. (BUG FIX) 初始化列表时，现在会为 `all_pred_logits`, `all_pred_boxes`, 和 `all_offsets`
#    正确地创建三个独立的空列表，从而解决了变量解包失败的问题。

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
        for key in targets: targets[key] = targets[key].to(device, non_blocking=True)
        outputs = model(clips)
        hm_pred = outputs['hm'].view(-1, *outputs['hm'].shape[2:]);
        wh_pred = outputs['wh'].view(-1, *outputs['wh'].shape[2:]);
        offset_pred = outputs['offset'].view(-1, *outputs['offset'].shape[2:])
        hm_gt = targets['hm'].view(-1, *targets['hm'].shape[2:]);
        wh_gt = targets['wh'].view(-1, *targets['wh'].shape[2:]);
        offset_gt = targets['offset'].view(-1, *targets['offset'].shape[2:])
        ind_gt = targets['ind'].view(-1, *targets['ind'].shape[2:]);
        ind_mask_gt = targets['ind_mask'].view(-1, *targets['ind_mask'].shape[2:])
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
        reg = _transpose_and_gather_feat(reg, inds); reg = reg.view(batch, K, 2); xs = xs.view(batch, K, 1) + reg[:, :,
                                                                                                              0:1]; ys = ys.view(
            batch, K, 1) + reg[:, :, 1:2]
    else:
        xs = xs.view(batch, K, 1) + 0.5; ys = ys.view(batch, K, 1) + 0.5
    wh = _transpose_and_gather_feat(wh, inds);
    wh = wh.view(batch, K, 2);
    clses = clses.view(batch, K, 1).float();
    scores = scores.view(batch, K, 1)
    bboxes = torch.cat([xs, ys, wh], dim=2)
    detections = torch.cat([bboxes, scores, clses], dim=2)
    return detections


@torch.no_grad()
def evaluate(model, data_loader, device, output_dir, epoch):
    model.eval();
    metric_logger = misc.MetricLogger(delimiter="  ");
    header = 'Test:'
    stats_by_class = {i: {'tp': 0, 'fp': 0, 'fn': 0} for i in range(model.num_classes)}
    stable_sequences, total_sequences = 0, 0
    conf_threshold, dist_threshold, stability_threshold = 0.3, 3.0, 0.8
    clip_length = model.clip_length

    for seq_name, sequence_cpu, targets in metric_logger.log_every(data_loader, 10, header):
        total_sequences += 1;
        sequence_gpu = sequence_cpu.to(device);
        sequence_len = sequence_gpu.shape[0]

        # ========================= 核心修改点 =========================
        all_pred_logits, all_pred_boxes, all_offsets = [], [], []
        # ==============================================================

        for i in range(0, sequence_len, clip_length):
            clip = sequence_gpu[i:min(i + clip_length, sequence_len)].unsqueeze(0);
            outputs = model(clip)
            all_pred_logits.append(outputs['hm'].squeeze(0));
            all_pred_boxes.append(outputs['wh'].squeeze(0));
            all_offsets.append(outputs['offset'].squeeze(0))

        seq_pred_heat = torch.cat(all_pred_logits, dim=0);
        seq_pred_wh = torch.cat(all_pred_boxes, dim=0);
        seq_pred_offset = torch.cat(all_offsets, dim=0)
        detections = centernet_decode(seq_pred_heat, seq_pred_wh, seq_pred_offset, K=100)
        consistent_frames = 0

        for t in range(sequence_len):
            preds_t = detections[t];
            scores = preds_t[:, 4];
            keep = scores > conf_threshold;
            preds_t = preds_t[keep]
            gt_boxes = targets[t]['boxes'].to(device);
            gt_labels = targets[t]['labels'].to(device)
            output_res = seq_pred_heat.shape[-1];
            gt_centers_feat_scale = gt_boxes[:, :2] * output_res
            gt_matched = torch.zeros(gt_boxes.shape[0], device=device, dtype=torch.bool);
            pred_matched = torch.zeros(preds_t.shape[0], device=device, dtype=torch.bool)

            if gt_boxes.shape[0] > 0 and preds_t.shape[0] > 0:
                pred_centers_feat_scale = preds_t[:, :2];
                dist_matrix = torch.cdist(pred_centers_feat_scale, gt_centers_feat_scale)
                closest_pred_dist, closest_pred_idx = dist_matrix.min(dim=0)
                for gt_idx, pred_idx in enumerate(closest_pred_idx):
                    if closest_pred_dist[gt_idx] < dist_threshold and preds_t[pred_idx, 5].int().item() == gt_labels[
                        gt_idx].item() and not pred_matched[pred_idx]:
                        stats_by_class[gt_labels[gt_idx].item()]['tp'] += 1;
                        gt_matched[gt_idx] = True;
                        pred_matched[pred_idx] = True
                if gt_matched.any(): consistent_frames += 1

            for i in range(preds_t.shape[0]):
                if not pred_matched[i]: stats_by_class[preds_t[i, 5].int().item()]['fp'] += 1
            for i in range(gt_boxes.shape[0]):
                if not gt_matched[i]: stats_by_class[gt_labels[i].item()]['fn'] += 1

        if (consistent_frames / sequence_len if sequence_len > 0 else 0) > stability_threshold: stable_sequences += 1

    metric_logger.synchronize_between_processes()
    total_tp = sum(stats['tp'] for stats in stats_by_class.values());
    total_fp = sum(stats['fp'] for stats in stats_by_class.values());
    total_fn = sum(stats['fn'] for stats in stats_by_class.values())
    avg_recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
    avg_false_alarm = total_fp / (total_fp + total_tp) if (total_fp + total_tp) > 0 else 0.0
    spatial_stability = stable_sequences / total_sequences if total_sequences > 0 else 0.0

    return {'avg_recall': avg_recall, 'avg_false_alarm': avg_false_alarm, 'spatial_stability': spatial_stability}


@torch.no_grad()
def visualize_epoch(model, data_loader, device, output_dir, epoch, vis_folder_name):
    model.eval();
    vis_dir = os.path.join(output_dir, vis_folder_name);
    os.makedirs(vis_dir, exist_ok=True);
    header = f'Visualizing {vis_folder_name}:';
    metric_logger = misc.MetricLogger(delimiter="  ")
    for seq_name, sequence_cpu, gt_targets in metric_logger.log_every(data_loader, 1, header):
        sequence_gpu = sequence_cpu.to(device);
        sequence_len = sequence_gpu.shape[0]
        preds_for_vis = []
        for t in range(sequence_len):
            clip = sequence_gpu[t].unsqueeze(0).unsqueeze(0);
            outputs = model(clip)
            heat = outputs['hm'].squeeze().unsqueeze(0);
            wh = outputs['wh'].squeeze().unsqueeze(0);
            offset = outputs['offset'].squeeze().unsqueeze(0)
            detections = centernet_decode(heat, wh, offset, K=100).squeeze(0)
            scores = detections[:, 4];
            keep = scores > 0.3
            final_boxes = detections[keep, :4]
            output_res = heat.shape[-1]
            final_boxes[:, 0] /= output_res;
            final_boxes[:, 1] /= output_res;
            final_boxes[:, 2] /= output_res;
            final_boxes[:, 3] /= output_res
            preds_for_vis.append({'boxes': final_boxes, 'scores': scores[keep], 'labels': detections[keep, 5].int()})
        video_path = os.path.join(vis_dir, f"{seq_name}_epoch_{epoch}.mp4");
        save_eval_video(sequence_cpu, gt_targets, preds_for_vis, video_path)