# util/visualize.py (版本 3 - 实现帧内 NMS)
# 核心改动:
# 1. (FEATURE) `save_eval_video` 函数现在可以接受密集的预测张量（T, N, Dims）。
# 2. (BUG FIX) 修复了 `IndexError`。函数现在会在其主循环内部逐帧执行置信度过滤和 NMS。
#    这确保了它可以正确处理有多个、一个或零个检测的帧，而不会崩溃。

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torchvision.ops.boxes import nms


def box_cxcywh_to_xyxy_numpy(box):
    x_c, y_c, w, h = box
    x1 = x_c - w / 2
    y1 = y_c - h / 2
    x2 = x_c + w / 2
    y2 = y_c + h / 2
    return np.array([x1, y1, x2, y2])


def box_cxcywh_to_xyxy_torch(x):
    x_c, y_c, w, h = x.unbind(-1)
    b = [(x_c - 0.5 * w), (y_c - 0.5 * h),
         (x_c + 0.5 * w), (y_c + 0.5 * h)]
    return torch.stack(b, dim=-1)


# A distinct color for each class ID
CLASS_COLORS = [
    (255, 56, 56), (255, 157, 151), (255, 112, 31), (255, 178, 29),
    (207, 210, 49), (72, 249, 10), (146, 204, 23)
]
GT_COLOR = (0, 255, 0)
TEXT_COLOR = (255, 255, 255)


def save_eval_video(image_sequence, gt_targets, pred_boxes_cxcywh, pred_logits, output_path, num_classes,
                    conf_threshold=0.5, nms_threshold=0.5):
    _, _, H, W = image_sequence.shape
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(output_path, fourcc, 10.0, (W, H))

    for i in range(len(image_sequence)):
        frame = image_sequence[i].permute(1, 2, 0).numpy()
        frame = (frame * np.array([0.200, 0.200, 0.200]) + np.array([0.425, 0.425, 0.425])) * 255
        frame = frame.astype(np.uint8)
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        # Draw Ground Truth boxes
        gt_boxes_np = gt_targets[i]['boxes'].numpy()
        gt_labels_np = gt_targets[i]['labels'].numpy()
        for box, label in zip(gt_boxes_np, gt_labels_np):
            box_xyxy = box_cxcywh_to_xyxy_numpy(box)
            x1, y1, x2, y2 = (box_xyxy * np.array([W, H, W, H])).astype(int)
            cv2.rectangle(frame, (x1, y1), (x2, y2), GT_COLOR, 2)
            cv2.putText(frame, f'GT: {label}', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, GT_COLOR, 2)

        # ========================= 核心修改点: 在函数内部执行 NMS =========================
        # 1. 获取当前帧的预测
        frame_logits = pred_logits[i]
        frame_boxes_cxcywh = pred_boxes_cxcywh[i]

        # 2. 置信度过滤
        probs = F.softmax(frame_logits, dim=-1)
        scores, labels = probs.max(-1)
        keep = (labels < num_classes) & (scores > conf_threshold)

        if keep.sum() > 0:
            # 3. 对保留的预测应用 NMS
            kept_boxes_xyxy = box_cxcywh_to_xyxy_torch(frame_boxes_cxcywh[keep])
            kept_scores = scores[keep]

            nms_indices = nms(kept_boxes_xyxy, kept_scores, nms_threshold)

            # 4. 绘制所有通过 NMS 的框
            for idx in nms_indices:
                box_xyxy = kept_boxes_xyxy[idx].cpu().numpy()
                score = kept_scores[idx].item()
                label = labels[keep][idx].item()

                x1, y1, x2, y2 = (box_xyxy * np.array([W, H, W, H])).astype(int)
                color = CLASS_COLORS[label % len(CLASS_COLORS)]
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                text = f'P: {label} ({score:.2f})'
                cv2.putText(frame, text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, TEXT_COLOR, 2)
        # =================================================================================

        video_writer.write(frame)

    video_writer.release()
    print(f"Saved evaluation video to {output_path}")