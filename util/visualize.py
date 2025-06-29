# util/visualize.py (版本 2 - 修复 numpy 转换错误)
# 核心改动:
# 1. 在将 pred_logits 和 pred_boxes 张量转换为 numpy 或从中计算派生值之前，
#    调用 .detach()。这可以防止在张量需要计算梯度时（例如在训练循环中）发生运行时错误。

import cv2
import numpy as np
import torch

# A distinct color for each class ID
CLASS_COLORS = [
    (255, 56, 56), (255, 157, 151), (255, 112, 31), (255, 178, 29),
    (207, 210, 49), (72, 249, 10), (146, 204, 23)
]
# Green for Ground Truth
GT_COLOR = (0, 255, 0)
# White for text
TEXT_COLOR = (255, 255, 255)


def box_cxcywh_to_xyxy_numpy(box):
    x_c, y_c, w, h = box
    x1 = x_c - w / 2
    y1 = y_c - h / 2
    x2 = x_c + w / 2
    y2 = y_c + h / 2
    return np.array([x1, y1, x2, y2])


def save_eval_video(image_sequence, gt_targets, pred_boxes, pred_logits, output_path, num_classes, conf_threshold=0.5):
    """
    Saves a video with ground truth and predicted bounding boxes.
    - image_sequence: (T, C, H, W) tensor of original images.
    - gt_targets: List of T dictionaries, each with 'boxes' and 'labels'.
    - pred_boxes: (T, 4) tensor of predicted boxes (cxcywh format).
    - pred_logits: (T, num_classes + 1) tensor of predicted logits.
    - output_path: Path to save the .mp4 video.
    """
    _, _, H, W = image_sequence.shape

    # Setup video writer
    # Use 'mp4v' for .mp4 files
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(output_path, fourcc, 10.0, (W, H))

    # ========================= 核心修改点 1 =========================
    # 在计算前分离张量，以避免梯度问题
    pred_probs = pred_logits.detach().softmax(-1)
    # ==============================================================
    pred_scores, pred_labels = pred_probs.max(-1)

    for i in range(len(image_sequence)):
        # Convert image tensor to OpenCV format (H, W, C)
        frame = image_sequence[i].permute(1, 2, 0).numpy()
        # Denormalize and convert to uint8
        frame = (frame * np.array([0.200, 0.200, 0.200]) + np.array([0.425, 0.425, 0.425])) * 255
        frame = frame.astype(np.uint8)
        # Convert RGB to BGR for OpenCV
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        # Draw Ground Truth boxes (Green)
        gt_boxes_cxcywh = gt_targets[i]['boxes'].numpy()
        gt_labels = gt_targets[i]['labels'].numpy()
        for box, label in zip(gt_boxes_cxcywh, gt_labels):
            box_xyxy = box_cxcywh_to_xyxy_numpy(box)
            # Denormalize coordinates
            x1, y1, x2, y2 = (box_xyxy * np.array([W, H, W, H])).astype(int)
            cv2.rectangle(frame, (x1, y1), (x2, y2), GT_COLOR, 2)
            cv2.putText(frame, f'GT: {label}', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, GT_COLOR, 2)

        # Draw Predicted boxes (Red, if confidence is high enough)
        score = pred_scores[i].item()
        label = pred_labels[i].item()
        if score > conf_threshold and label < num_classes:
            # ========================= 核心修改点 2 =========================
            # 在转换前分离张量，以避免梯度问题
            box_cxcywh = pred_boxes[i].cpu().detach().numpy()
            # ==============================================================
            box_xyxy = box_cxcywh_to_xyxy_numpy(box_cxcywh)
            # Denormalize coordinates
            x1, y1, x2, y2 = (box_xyxy * np.array([W, H, W, H])).astype(int)

            color = CLASS_COLORS[label % len(CLASS_COLORS)]
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            text = f'P: {label} ({score:.2f})'
            cv2.putText(frame, text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, TEXT_COLOR, 2)

        video_writer.write(frame)

    video_writer.release()
    print(f"Saved evaluation video to {output_path}")