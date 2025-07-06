# util/visualize.py (版本 4 - CenterNet 可视化工具)
# 核心改动:
# 1. (REFACTOR) `save_eval_video` 函数的签名被彻底改变。它现在接受一个 `predictions` 列表，
#    其中每个元素都是一个包含单帧所有检测结果的字典。
# 2. (REFACTOR) 函数不再执行任何模型相关的逻辑（如 softmax, NMS）。它只负责绘制
#    已经经过后处理的、干净的边界框列表。
# 3. (FEATURE) 内部循环现在会遍历每一帧的预测框列表，并在图像上绘制所有检测到的目标，
#    从而能够正确地可视化多目标检测结果。

import cv2
import numpy as np
import torch


def box_cxcywh_to_xyxy_numpy(box):
    x_c, y_c, w, h = box
    x1 = x_c - w / 2
    y1 = y_c - h / 2
    x2 = x_c + w / 2
    y2 = y_c + h / 2
    return np.array([x1, y1, x2, y2])


CLASS_COLORS = [
    (255, 56, 56), (255, 157, 151), (255, 112, 31), (255, 178, 29),
    (207, 210, 49), (72, 249, 10), (146, 204, 23)
]
GT_COLOR = (0, 255, 0)
TEXT_COLOR = (255, 255, 255)


def save_eval_video(image_sequence, gt_targets, predictions, output_path):
    """
    Saves a video with ground truth and predicted bounding boxes from CenterNet-style output.
    - image_sequence: (T, C, H, W) tensor of original images.
    - gt_targets: List of T dictionaries, each with 'boxes' and 'labels' (original GT).
    - predictions: List of T dictionaries, each with post-processed 'boxes', 'scores', 'labels'.
    - output_path: Path to save the .mp4 video.
    """
    _, _, H, W = image_sequence.shape
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(output_path, fourcc, 10.0, (W, H))

    for i in range(len(image_sequence)):
        frame = image_sequence[i].permute(1, 2, 0).numpy()
        frame = (frame * np.array([0.200, 0.200, 0.200]) + np.array([0.425, 0.425, 0.425])) * 255
        frame = frame.astype(np.uint8)
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        # Draw Ground Truth boxes
        if i < len(gt_targets):
            gt_boxes_np = gt_targets[i]['boxes'].numpy()
            gt_labels_np = gt_targets[i]['labels'].numpy()
            for box, label in zip(gt_boxes_np, gt_labels_np):
                box_xyxy = box_cxcywh_to_xyxy_numpy(box)
                x1, y1, x2, y2 = (box_xyxy * np.array([W, H, W, H])).astype(int)
                cv2.rectangle(frame, (x1, y1), (x2, y2), GT_COLOR, 2)
                cv2.putText(frame, f'GT: {label}', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, GT_COLOR, 2)

        # Draw Predicted boxes for the current frame
        if i < len(predictions):
            pred_boxes = predictions[i]['boxes'].cpu().numpy()
            pred_scores = predictions[i]['scores'].cpu().numpy()
            pred_labels = predictions[i]['labels'].cpu().numpy()

            for box, score, label in zip(pred_boxes, pred_scores, pred_labels):
                box_xyxy = box_cxcywh_to_xyxy_numpy(box)
                x1, y1, x2, y2 = (box_xyxy * np.array([W, H, W, H])).astype(int)

                color = CLASS_COLORS[label % len(CLASS_COLORS)]
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                text = f'P: {label} ({score:.2f})'
                cv2.putText(frame, text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, TEXT_COLOR, 2)

        video_writer.write(frame)

    video_writer.release()
    print(f"Saved evaluation video to {output_path}")
