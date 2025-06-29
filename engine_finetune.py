# 文件: engine_finetune.py (版本 26 - 重新引入 L2 回归损失)
# 核心改动:
# 1. (BUG FIX) 重新引入了一个直接的回归损失（L2 Loss），以解决边界框尺寸不正确和多目标检测效果差的问题。
#    这个损失函数会直接惩罚预测框和真实框之间的坐标差异。
# 2. (FEATURE) 更新了 `weight_dict`，为新的 `loss_l2` 分配了一个较低的权重（2），
#    同时保持 CIoU Loss 作为主要的几何损失（权重为 5），以平衡训练过程。
# 3. (REFACTOR) 更新了日志记录，以包含所有三个损失项，从而提供对训练动态的全面洞察。

import math
import sys
from typing import Iterable
import torch
import torch.nn.functional as F
import util.misc as misc
from util.visualize import save_eval_video
from torchvision.ops.boxes import box_iou
import numpy as np
import os


def box_cxcywh_to_xyxy(x):
    x_c, y_c, w, h = x.unbind(-1)
    b = [(x_c - 0.5 * w), (y_c - 0.5 * h), (x_c + 0.5 * w), (y_c + 0.5 * h)]
    return torch.stack(b, dim=-1)


def ciou_loss(pred_cxcywh, target_cxcywh, eps=1e-7):
    pred_xyxy = box_cxcywh_to_xyxy(pred_cxcywh)
    target_xyxy = box_cxcywh_to_xyxy(target_cxcywh)
    inter_mins = torch.max(pred_xyxy[:, :2], target_xyxy[:, :2])
    inter_maxs = torch.min(pred_xyxy[:, 2:], target_xyxy[:, 2:])
    inter_wh = (inter_maxs - inter_mins).clamp(min=0)
    inter_area = inter_wh[:, 0] * inter_wh[:, 1]
    pred_area = pred_cxcywh[:, 2] * pred_cxcywh[:, 3]
    target_area = target_cxcywh[:, 2] * target_cxcywh[:, 3]
    union_area = pred_area + target_area - inter_area
    iou = inter_area / (union_area + eps)
    enclose_mins = torch.min(pred_xyxy[:, :2], target_xyxy[:, :2])
    enclose_maxs = torch.max(pred_xyxy[:, 2:], target_xyxy[:, 2:])
    enclose_wh = (enclose_maxs - enclose_mins).clamp(min=0)
    c2 = enclose_wh[:, 0] ** 2 + enclose_wh[:, 1] ** 2 + eps
    rho2 = ((pred_cxcywh[:, 0] - target_cxcywh[:, 0]) ** 2 + (pred_cxcywh[:, 1] - target_cxcywh[:, 1]) ** 2)
    v = (4 / (math.pi ** 2)) * torch.pow(torch.atan(target_cxcywh[:, 2] / (target_cxcywh[:, 3] + 1e-3)) - torch.atan(
        pred_cxcywh[:, 2] / (pred_cxcywh[:, 3] + 1e-3)), 2)
    with torch.no_grad():
        alpha = v / (1 - iou + v + eps)
    return 1 - iou + (rho2 / c2) + (alpha * v)


def train_one_epoch(model: torch.nn.Module, data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, log_writer=None, args=None):
    model.train()
    metric_logger = misc.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', misc.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = f'Epoch: [{epoch}]'
    print_freq = 20
    model_without_ddp = model.module if hasattr(model, 'module') else model
    no_object_class_idx = model_without_ddp.num_classes

    for data_iter_step, (clips, targets) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        clips = clips.to(device, non_blocking=True)
        outputs = model(clips)
        pred_logits, pred_boxes = outputs['pred_logits'], outputs['pred_boxes']
        B, T, N, _ = pred_logits.shape

        total_loss_ce, total_loss_ciou, total_loss_l2 = torch.tensor(0.0, device=device), torch.tensor(0.0,
                                                                                                       device=device), torch.tensor(
            0.0, device=device)
        num_pos = 0

        for b in range(B):
            for t in range(T):
                frame_logits, frame_boxes = pred_logits[b, t], pred_boxes[b, t]
                gt_boxes, gt_labels = targets[b][t]['boxes'].to(device), targets[b][t]['labels'].to(device)

                if gt_boxes.shape[0] == 0:
                    target_classes_o = torch.full((N,), no_object_class_idx, dtype=torch.int64, device=device)
                    total_loss_ce += F.cross_entropy(frame_logits, target_classes_o)
                    continue

                cost_class = -F.softmax(frame_logits, dim=-1)[:, gt_labels]
                cost_bbox = torch.cdist(frame_boxes, gt_boxes, p=1)
                cost_iou = -ciou_loss(frame_boxes.unsqueeze(1).repeat(1, gt_boxes.shape[0], 1).flatten(0, 1),
                                      gt_boxes.unsqueeze(0).repeat(frame_boxes.shape[0], 1, 1).flatten(0, 1)).view(
                    frame_boxes.shape[0], -1)
                cost = 5 * cost_class + 5 * cost_bbox + 2 * cost_iou

                matched_pred_indices = cost.argmin(dim=0)
                num_pos += len(gt_boxes)

                matched_preds_logits = frame_logits[matched_pred_indices]
                matched_preds_boxes = frame_boxes[matched_pred_indices]
                total_loss_ce += F.cross_entropy(matched_preds_logits, gt_labels)

                # ========================= 核心修改点 1: 添加 L2 Loss 计算 =========================
                total_loss_ciou += ciou_loss(matched_preds_boxes, gt_boxes).sum()
                total_loss_l2 += F.mse_loss(matched_preds_boxes, gt_boxes, reduction='sum')
                # =================================================================================

                unmatched_mask = torch.ones(N, dtype=torch.bool, device=device)
                unmatched_mask[matched_pred_indices] = False
                unmatched_logits = frame_logits[unmatched_mask]
                if unmatched_logits.shape[0] > 0:
                    target_classes_o = torch.full((unmatched_logits.shape[0],), no_object_class_idx, dtype=torch.int64,
                                                  device=device)
                    total_loss_ce += F.cross_entropy(unmatched_logits, target_classes_o)

        loss_ce_avg = total_loss_ce / (B * T)
        loss_ciou_avg = total_loss_ciou / num_pos if num_pos > 0 else torch.tensor(0.0, device=device)
        loss_l2_avg = total_loss_l2 / num_pos if num_pos > 0 else torch.tensor(0.0, device=device)

        # ========================= 核心修改点 2: 更新权重字典 =========================
        weight_dict = {'loss_ce': 2, 'loss_ciou': 5, 'loss_l2': 2}
        losses = (loss_ce_avg * weight_dict['loss_ce'] +
                  loss_ciou_avg * weight_dict['loss_ciou'] +
                  loss_l2_avg * weight_dict['loss_l2'])
        # =========================================================================

        if not math.isfinite(losses.item()):
            print(f"Loss is {losses.item()}, stopping training")
            sys.exit(1)

        optimizer.zero_grad()
        losses.backward()
        optimizer.step()
        torch.cuda.synchronize()

        # ========================= 核心修改点 3: 更新日志记录 =========================
        metric_logger.update(loss=losses.item(),
                             loss_ce=loss_ce_avg.item(),
                             loss_ciou=loss_ciou_avg.item(),
                             loss_l2=loss_l2_avg.item())
        # =========================================================================

        metric_logger.update(lr=optimizer.param_groups[0]["lr"])

    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def evaluate(model, data_loader, device, output_dir, epoch):
    model.eval()
    metric_logger = misc.MetricLogger(delimiter="  ")
    header = 'Test:'
    stats_by_class = {i: {'tp': 0, 'fp': 0, 'fn': 0} for i in range(model.num_classes)}
    stable_sequences, total_sequences = 0, 0
    iou_threshold, stability_threshold, conf_threshold = 0.3, 0.8, 0.5
    clip_length = model.clip_length

    for seq_name, sequence_cpu, targets in metric_logger.log_every(data_loader, 10, header):
        total_sequences += 1
        sequence_gpu = sequence_cpu.to(device)
        sequence_len = sequence_gpu.shape[0]

        all_pred_logits, all_pred_boxes = [], []
        for i in range(0, sequence_len, clip_length):
            clip = sequence_gpu[i:min(i + clip_length, sequence_len)].unsqueeze(0)
            outputs = model(clip)
            all_pred_logits.append(outputs['pred_logits'].squeeze(0))
            all_pred_boxes.append(outputs['pred_boxes'].squeeze(0))

        seq_pred_logits = torch.cat(all_pred_logits, dim=0)
        seq_pred_boxes = torch.cat(all_pred_boxes, dim=0)

        consistent_frames = 0
        for t in range(sequence_len):
            frame_logits = seq_pred_logits[t]
            frame_boxes_cxcywh = seq_pred_boxes[t]
            gt_boxes_xyxy = box_cxcywh_to_xyxy(targets[t]['boxes']).to(device)
            gt_labels = targets[t]['labels'].to(device)

            probs = F.softmax(frame_logits, -1)
            scores, labels = probs.max(-1)
            keep = scores > conf_threshold

            frame_pred_boxes_xyxy = box_cxcywh_to_xyxy(frame_boxes_cxcywh[keep])
            frame_pred_labels = labels[keep]

            gt_matched = torch.zeros(gt_labels.shape[0], device=device, dtype=torch.bool)

            if frame_pred_boxes_xyxy.shape[0] > 0 and gt_boxes_xyxy.shape[0] > 0:
                iou_matrix = box_iou(frame_pred_boxes_xyxy, gt_boxes_xyxy)
                if iou_matrix.max() > iou_threshold:
                    consistent_frames += 1
                for i in range(frame_pred_boxes_xyxy.shape[0]):
                    pred_label = frame_pred_labels[i].item()
                    if pred_label < model.num_classes:
                        best_iou, best_gt_idx = iou_matrix[i].max(0)
                        if best_iou > iou_threshold and pred_label == gt_labels[best_gt_idx].item() and not gt_matched[
                            best_gt_idx]:
                            stats_by_class[pred_label]['tp'] += 1
                            gt_matched[best_gt_idx] = True
                        else:
                            stats_by_class[pred_label]['fp'] += 1
            elif frame_pred_boxes_xyxy.shape[0] > 0:
                for i in range(frame_pred_boxes_xyxy.shape[0]):
                    pred_label = frame_pred_labels[i].item()
                    if pred_label < model.num_classes:
                        stats_by_class[pred_label]['fp'] += 1

            unmatched_gt_labels = gt_labels[gt_matched.logical_not()]
            for label in unmatched_gt_labels:
                stats_by_class[label.item()]['fn'] += 1

        temporal_consistency = consistent_frames / sequence_len if sequence_len > 0 else 0
        if temporal_consistency > stability_threshold:
            stable_sequences += 1

    metric_logger.synchronize_between_processes()
    total_tp = sum(stats['tp'] for stats in stats_by_class.values())
    total_fp = sum(stats['fp'] for stats in stats_by_class.values())
    total_fn = sum(stats['fn'] for stats in stats_by_class.values())

    avg_recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
    avg_false_alarm = total_fp / (total_fp + total_tp) if (total_fp + total_tp) > 0 else 0.0
    spatial_stability = stable_sequences / total_sequences if total_sequences > 0 else 0.0

    return {'avg_recall': avg_recall, 'avg_false_alarm': avg_false_alarm, 'spatial_stability': spatial_stability}


@torch.no_grad()
def visualize_epoch(model, data_loader, device, output_dir, epoch, vis_folder_name):
    model.eval()
    vis_dir = os.path.join(output_dir, vis_folder_name)
    os.makedirs(vis_dir, exist_ok=True)
    clip_length = model.clip_length
    header = f'Visualizing {vis_folder_name}:'
    metric_logger = misc.MetricLogger(delimiter="  ")
    for seq_name, sequence_cpu, targets in metric_logger.log_every(data_loader, 1, header):
        sequence_gpu = sequence_cpu.to(device)
        sequence_len = sequence_gpu.shape[0]

        all_pred_logits, all_pred_boxes = [], []
        for i in range(0, sequence_len, clip_length):
            clip = sequence_gpu[i:min(i + clip_length, sequence_len)].unsqueeze(0)
            outputs = model(clip)
            all_pred_logits.append(outputs['pred_logits'].squeeze(0))
            all_pred_boxes.append(outputs['pred_boxes'].squeeze(0))

        full_pred_logits = torch.cat(all_pred_logits, dim=0)
        full_pred_boxes = torch.cat(all_pred_boxes, dim=0)

        probs = F.softmax(full_pred_logits, dim=-1)
        all_scores, _ = probs[..., :-1].max(dim=-1)
        best_patch_indices = all_scores.argmax(dim=-1)

        vis_logits = torch.stack([full_pred_logits[t, best_patch_indices[t]] for t in range(sequence_len)])
        vis_boxes = torch.stack([full_pred_boxes[t, best_patch_indices[t]] for t in range(sequence_len)])

        video_path = os.path.join(vis_dir, f"{seq_name}_epoch_{epoch}.mp4")
        save_eval_video(sequence_cpu, targets, vis_boxes.cpu(), vis_logits.cpu(),
                        video_path, model.num_classes, conf_threshold=0.5)