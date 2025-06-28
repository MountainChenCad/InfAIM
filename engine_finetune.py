# 文件: engine_finetune.py (版本 11 - 修复评估指标 & 添加可视化)
# 核心改动:
# 1. (BUG FIX) 在 evaluate 中加入置信度阈值。只有当预测得分高于阈值时，才将其视为有效预测。
#    这可以防止低置信度的噪声预测被计为 False Positives，从而修复指标。
# 2. (FEATURE) 导入新的可视化工具，并在每次评估时，将第一个验证序列的检测结果保存为视频。

import math
import sys
from typing import Iterable
import torch
import torch.nn.functional as F
import util.misc as misc
from util.visualize import save_eval_video  # <--- 导入新工具
from torchvision.ops.boxes import box_iou, generalized_box_iou
import numpy as np


def box_cxcywh_to_xyxy(x):
    x_c, y_c, w, h = x.unbind(-1)
    b = [(x_c - 0.5 * w), (y_c - 0.5 * h),
         (x_c + 0.5 * w), (y_c + 0.5 * h)]
    return torch.stack(b, dim=-1)


# train_one_epoch 函数保持不变...
def train_one_epoch(model: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, log_writer=None, args=None):
    model.train()
    metric_logger = misc.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', misc.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 20
    model_without_ddp = model.module if hasattr(model, 'module') else model
    no_object_class_idx = model_without_ddp.num_classes
    for data_iter_step, (clips, targets) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        clips = clips.to(device, non_blocking=True)
        outputs = model(clips)
        pred_logits = outputs['pred_logits']
        pred_boxes_cxcywh = outputs['pred_boxes']
        batch_size, num_frames = pred_logits.shape[:2]
        total_loss_ce = torch.tensor(0.0, device=device)
        total_loss_bbox = torch.tensor(0.0, device=device)
        num_matched_boxes = 0
        for b in range(batch_size):
            for t in range(num_frames):
                frame_pred_logits = pred_logits[b, t]
                frame_pred_box_cxcywh = pred_boxes_cxcywh[b, t]
                frame_target = targets[b][t]
                gt_boxes_cxcywh = frame_target['boxes'].to(device)
                gt_labels = frame_target['labels'].to(device)
                num_gt = gt_boxes_cxcywh.shape[0]
                if num_gt == 0:
                    target_class = torch.tensor([no_object_class_idx], device=device, dtype=torch.long)
                    loss_ce = F.cross_entropy(frame_pred_logits.unsqueeze(0), target_class)
                    total_loss_ce += loss_ce
                else:
                    cost_bbox = torch.cdist(frame_pred_box_cxcywh.unsqueeze(0), gt_boxes_cxcywh, p=1)
                    pred_prob = frame_pred_logits.softmax(0)
                    cost_class = -pred_prob[gt_labels]
                    total_cost = (cost_bbox.squeeze(0) + cost_class)
                    min_cost, best_gt_idx = torch.min(total_cost, dim=0)
                    target_class = gt_labels[best_gt_idx].unsqueeze(0)
                    loss_ce = F.cross_entropy(frame_pred_logits.unsqueeze(0), target_class)
                    matched_gt_box_cxcywh = gt_boxes_cxcywh[best_gt_idx]
                    loss_bbox = F.l1_loss(frame_pred_box_cxcywh, matched_gt_box_cxcywh)
                    total_loss_ce += loss_ce
                    total_loss_bbox += loss_bbox
                    num_matched_boxes += 1
        loss_ce_avg = total_loss_ce / (batch_size * num_frames)
        loss_bbox_avg = total_loss_bbox / num_matched_boxes if num_matched_boxes > 0 else torch.tensor(0.0,
                                                                                                       device=device)
        weight_dict = {'loss_ce': 2, 'loss_bbox': 1}
        losses = loss_ce_avg * weight_dict['loss_ce'] + loss_bbox_avg * weight_dict['loss_bbox']
        loss_value = losses.item()
        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)
        optimizer.zero_grad()
        losses.backward()
        optimizer.step()
        torch.cuda.synchronize()
        metric_logger.update(loss=loss_value, loss_ce=loss_ce_avg.item(), loss_bbox=loss_bbox_avg.item())
        lr = optimizer.param_groups[0]["lr"]
        metric_logger.update(lr=lr)
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def evaluate(model, data_loader, device, output_dir, epoch):
    model.eval()

    metric_logger = misc.MetricLogger(delimiter="  ")
    header = 'Test:'

    stats_by_class = {i: {'tp': 0, 'fp': 0, 'fn': 0} for i in range(model.num_classes)}
    stable_sequences = 0
    total_sequences = 0
    iou_threshold = 0.3
    stability_threshold = 0.8
    conf_threshold = 0.5  # <--- BUG FIX: 置信度阈值

    clip_length = model.clip_length

    for sequence_cpu, targets in metric_logger.log_every(data_loader, 10, header):
        total_sequences += 1
        sequence_gpu = sequence_cpu.to(device)
        sequence_len = sequence_gpu.shape[0]

        all_pred_logits = []
        all_pred_boxes = []

        for i in range(0, sequence_len, clip_length):
            end_i = min(i + clip_length, sequence_len)
            clip = sequence_gpu[i:end_i]
            clip_batch = clip.unsqueeze(0)
            outputs = model(clip_batch)
            all_pred_logits.append(outputs['pred_logits'].squeeze(0))
            all_pred_boxes.append(outputs['pred_boxes'].squeeze(0))

        pred_logits = torch.cat(all_pred_logits, dim=0)
        pred_boxes_cxcywh = torch.cat(all_pred_boxes, dim=0)
        pred_boxes_xyxy = box_cxcywh_to_xyxy(pred_boxes_cxcywh)

        consistent_frames = 0

        pred_probs = pred_logits.softmax(-1)
        pred_scores, pred_labels = pred_probs.max(-1)

        for t in range(sequence_len):
            frame_pred_score = pred_scores[t]
            frame_pred_label = pred_labels[t]

            gt_boxes_xyxy = box_cxcywh_to_xyxy(targets[t]['boxes'].to(device))
            gt_labels = targets[t]['labels'].to(device)

            if gt_boxes_xyxy.shape[0] > 0:
                # 仅在预测置信度足够高时才计算IoU
                if frame_pred_score > conf_threshold and frame_pred_label < model.num_classes:
                    iou_matrix = box_iou(pred_boxes_xyxy[t].unsqueeze(0), gt_boxes_xyxy)
                    if iou_matrix.max() > iou_threshold:
                        consistent_frames += 1

            gt_matched = [False] * len(gt_labels)

            # BUG FIX: 只有当预测置信度高于阈值时，才将其视为一个有效的预测
            if frame_pred_score > conf_threshold and frame_pred_label < model.num_classes:
                is_tp = False
                if gt_boxes_xyxy.shape[0] > 0:
                    iou_with_gts = box_iou(pred_boxes_xyxy[t].unsqueeze(0), gt_boxes_xyxy).squeeze(0)
                    best_iou, best_gt_idx = iou_with_gts.max(0)
                    if best_iou > iou_threshold and frame_pred_label == gt_labels[best_gt_idx] and not gt_matched[
                        best_gt_idx]:
                        stats_by_class[frame_pred_label.item()]['tp'] += 1
                        gt_matched[best_gt_idx] = True
                        is_tp = True

                if not is_tp:
                    stats_by_class[frame_pred_label.item()]['fp'] += 1

            for i, matched in enumerate(gt_matched):
                if not matched:
                    stats_by_class[gt_labels[i].item()]['fn'] += 1

        temporal_consistency = consistent_frames / sequence_len if sequence_len > 0 else 0
        if temporal_consistency > stability_threshold:
            stable_sequences += 1

        # FEATURE: 为第一个验证序列保存可视化视频
        if total_sequences == 1 and output_dir:
            video_path = f"{output_dir}/eval_epoch_{epoch}.mp4"
            save_eval_video(sequence_cpu, targets, pred_boxes_cxcywh, pred_logits, video_path, model.num_classes,
                            conf_threshold)

    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)

    total_tp = sum(stats['tp'] for stats in stats_by_class.values())
    total_fp = sum(stats['fp'] for stats in stats_by_class.values())
    total_fn = sum(stats['fn'] for stats in stats_by_class.values())

    avg_recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
    avg_false_alarm = total_fp / (total_fp + total_tp) if (total_fp + total_tp) > 0 else 0.0
    spatial_stability = stable_sequences / total_sequences if total_sequences > 0 else 0.0

    return {
        'avg_recall': avg_recall,
        'avg_false_alarm': avg_false_alarm,
        'spatial_stability': spatial_stability
    }