# 文件: engine_finetune.py (版本 5 - 添加评估函数)

import math
import sys
from typing import Iterable
import torch
import torch.nn.functional as F
import util.misc as misc
from torchvision.ops.boxes import box_iou, generalized_box_iou
import numpy as np


def box_cxcywh_to_xyxy(x):
    x_c, y_c, w, h = x.unbind(-1)
    b = [(x_c - 0.5 * w), (y_c - 0.5 * h),
         (x_c + 0.5 * w), (y_c + 0.5 * h)]
    return torch.stack(b, dim=-1)


# (train_one_epoch 函数保持版本 4 不变)
def train_one_epoch(model: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, log_writer=None, args=None):
    # ... (此处省略，与上一版完全相同)
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
        total_loss_giou = torch.tensor(0.0, device=device)
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
                    frame_pred_box_xyxy = box_cxcywh_to_xyxy(frame_pred_box_cxcywh)
                    gt_boxes_xyxy = box_cxcywh_to_xyxy(gt_boxes_cxcywh)

                    cost_bbox = torch.cdist(frame_pred_box_cxcywh.unsqueeze(0), gt_boxes_cxcywh, p=1)
                    cost_giou = -generalized_box_iou(frame_pred_box_xyxy.unsqueeze(0), gt_boxes_xyxy)

                    pred_prob = frame_pred_logits.softmax(0)
                    cost_class = -pred_prob[gt_labels]

                    total_cost = (cost_bbox.squeeze(0) +
                                  # cost_giou.squeeze(0)
                                  + cost_class)

                    min_cost, best_gt_idx = torch.min(total_cost, dim=0)

                    target_class = gt_labels[best_gt_idx].unsqueeze(0)
                    loss_ce = F.cross_entropy(frame_pred_logits.unsqueeze(0), target_class)

                    matched_gt_box_cxcywh = gt_boxes_cxcywh[best_gt_idx]
                    loss_bbox = F.l1_loss(frame_pred_box_cxcywh, matched_gt_box_cxcywh)

                    matched_gt_box_xyxy = gt_boxes_xyxy[best_gt_idx]
                    loss_giou = (1 - generalized_box_iou(frame_pred_box_xyxy.unsqueeze(0),
                                                         matched_gt_box_xyxy.unsqueeze(0))).squeeze()

                    total_loss_ce += loss_ce
                    total_loss_bbox += loss_bbox
                    total_loss_giou += loss_giou
                    num_matched_boxes += 1

        loss_ce_avg = total_loss_ce / (batch_size * num_frames)
        loss_bbox_avg = total_loss_bbox / num_matched_boxes if num_matched_boxes > 0 else torch.tensor(0.0,
                                                                                                       device=device)
        loss_giou_avg = total_loss_giou / num_matched_boxes if num_matched_boxes > 0 else torch.tensor(0.0,
                                                                                                       device=device)

        weight_dict = {'loss_ce': 1, 'loss_bbox': 1, 'loss_giou': 1}
        losses = loss_ce_avg * weight_dict['loss_ce'] + loss_bbox_avg * weight_dict['loss_bbox'] + loss_giou_avg * \
                 weight_dict['loss_giou']

        loss_value = losses.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)

        optimizer.zero_grad()
        losses.backward()
        optimizer.step()

        torch.cuda.synchronize()
        metric_logger.update(loss=loss_value, loss_ce=loss_ce_avg.item(), loss_bbox=loss_bbox_avg.item(),
                             loss_giou=loss_giou_avg.item())
        lr = optimizer.param_groups[0]["lr"]
        metric_logger.update(lr=lr)

    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def evaluate(model, data_loader, device):
    model.eval()

    # 准备用于计算指标的统计数据
    stats_by_class = {i: {'tp': 0, 'fp': 0, 'fn': 0} for i in range(model.num_classes)}
    stable_sequences = 0
    total_sequences = 0
    iou_threshold = 0.3
    stability_threshold = 0.8

    for sequence, targets in data_loader:
        total_sequences += 1
        sequence = sequence.to(device)
        sequence_len = sequence.shape[1]

        # 由于模型输入是固定长度的clip，我们需要逐个clip地处理长序列
        # 这里为了简化，我们假设评估时可以一次性处理整个序列
        # 注意：如果序列过长导致OOM，需要实现滑窗推理
        outputs = model(sequence)  # [1, SeqLen, C+1], [1, SeqLen, 4]

        pred_logits = outputs['pred_logits'].squeeze(0)
        pred_boxes_cxcywh = outputs['pred_boxes'].squeeze(0)
        pred_boxes_xyxy = box_cxcywh_to_xyxy(pred_boxes_cxcywh)

        consistent_frames = 0

        for t in range(sequence_len):
            frame_pred_logits = pred_logits[t]
            frame_pred_box_xyxy = pred_boxes_xyxy[t]

            pred_prob, pred_label = frame_pred_logits.softmax(-1).max(-1)

            frame_target = targets[t]
            gt_boxes_xyxy = box_cxcywh_to_xyxy(frame_target['boxes'].to(device))
            gt_labels = frame_target['labels'].to(device)

            # --- 时空序列稳定性计算 ---
            if gt_boxes_xyxy.shape[0] > 0:
                # 只考虑有目标的帧
                iou_matrix = box_iou(frame_pred_box_xyxy.unsqueeze(0), gt_boxes_xyxy)
                if iou_matrix.max() > iou_threshold:
                    consistent_frames += 1

            # --- 目标识别精度统计 ---
            gt_matched = [False] * len(gt_labels)

            # 如果预测为非背景类
            if pred_label < model.num_classes:
                is_tp = False
                if gt_boxes_xyxy.shape[0] > 0:
                    iou_with_gts = box_iou(frame_pred_box_xyxy.unsqueeze(0), gt_boxes_xyxy).squeeze(0)
                    best_iou, best_gt_idx = iou_with_gts.max(0)

                    if best_iou > iou_threshold and pred_label == gt_labels[best_gt_idx] and not gt_matched[
                        best_gt_idx]:
                        stats_by_class[pred_label.item()]['tp'] += 1
                        gt_matched[best_gt_idx] = True
                        is_tp = True

                if not is_tp:
                    stats_by_class[pred_label.item()]['fp'] += 1

            # 计算漏检 (FN)
            for i, matched in enumerate(gt_matched):
                if not matched:
                    stats_by_class[gt_labels[i].item()]['fn'] += 1

        # 计算当前序列的时序一致性
        temporal_consistency = consistent_frames / sequence_len if sequence_len > 0 else 0
        if temporal_consistency > stability_threshold:
            stable_sequences += 1

    # --- 汇总计算最终指标 ---
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