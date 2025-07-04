# 文件: dataset_video.py (版本 7 - 支持多场景数据集)
# 核心改动:
# 1. (REFACTOR) `__init__` 现在接受一个 `scene_paths` 列表，而不是单个 `data_root`。
# 2. (REFACTOR) 数据集现在会遍历所有提供的场景路径，并聚合在每个场景中找到的所有时间序列 (data1, data2, ...)。
# 3. (FEATURE) 路径构建逻辑已更新，以处理新的 `scene_X/images/dataY` 层次结构。
# 4. (FEATURE) `__getitem__` 现在返回一个唯一的序列名称，例如 `scene_1_data1`，以防止在可视化时发生文件名冲突。

import os
import torch
from torch.utils.data import Dataset
from PIL import Image
import numpy as np


class VideoDetectionDataset(Dataset):
    def __init__(self, scene_paths, clip_length=8, transform=None, return_clips=True):
        self.scene_paths = scene_paths
        self.clip_length = clip_length
        self.transform = transform
        self.return_clips = return_clips
        self.data_samples = self._make_dataset()

    def _make_dataset(self):
        data_samples = []
        for scene_path in self.scene_paths:
            image_dir = os.path.join(scene_path, 'images')
            if not os.path.isdir(image_dir):
                print(f"Warning: Image directory not found at {image_dir}")
                continue

            sequences = sorted([d for d in os.listdir(image_dir) if os.path.isdir(os.path.join(image_dir, d))])

            for seq_name in sequences:
                seq_path = os.path.join(image_dir, seq_name)
                frame_files = sorted(
                    [f for f in os.listdir(seq_path) if f.lower().endswith(('.bmp', '.png', '.jpg', '.jpeg'))],
                    key=lambda x: int(os.path.splitext(x)[0].split('_')[-1]))
                num_frames = len(frame_files)

                if self.return_clips:
                    if num_frames >= self.clip_length:
                        for start_frame in range(num_frames - self.clip_length + 1):
                            data_samples.append(('clip', scene_path, seq_name, start_frame, frame_files))
                else:
                    data_samples.append(('sequence', scene_path, seq_name, frame_files))
        return data_samples

    def __len__(self):
        return len(self.data_samples)

    def __getitem__(self, idx):
        sample_info = self.data_samples[idx]
        sample_type = sample_info[0]

        if sample_type == 'clip':
            _, scene_path, seq_name, start_frame, frame_files = sample_info
            return self._get_clip(scene_path, seq_name, start_frame, frame_files)
        elif sample_type == 'sequence':
            _, scene_path, seq_name, frame_files = sample_info
            return self._get_sequence(scene_path, seq_name, frame_files)
        else:
            raise ValueError(f"Unknown sample type: {sample_type}")

    def _get_clip(self, scene_path, seq_name, start_frame, frame_files):
        clip_images, clip_targets = [], []
        for i in range(self.clip_length):
            frame_idx = start_frame + i
            frame_filename = frame_files[frame_idx]
            img_path = os.path.join(scene_path, 'images', seq_name, frame_filename)
            image = Image.open(img_path).convert('RGB')
            if self.transform:
                image = self.transform(image)
            clip_images.append(image)

            label_filename = os.path.splitext(frame_filename)[0] + '.txt'
            label_path = os.path.join(scene_path, 'labels', seq_name, label_filename)

            boxes, labels = [], []
            if os.path.exists(label_path):
                with open(label_path, 'r') as f:
                    for line in f.readlines():
                        parts = line.strip().split()
                        if not parts: continue
                        class_id = int(parts[0])
                        boxes.append([float(p) for p in parts[1:]])
                        labels.append(class_id)
            target = {'boxes': torch.as_tensor(boxes, dtype=torch.float32).reshape(-1, 4),
                      'labels': torch.as_tensor(labels, dtype=torch.int64)}
            clip_targets.append(target)
        return torch.stack(clip_images, dim=0), clip_targets

    def _get_sequence(self, scene_path, seq_name, frame_files):
        sequence_images, sequence_targets = [], []
        for frame_filename in frame_files:
            img_path = os.path.join(scene_path, 'images', seq_name, frame_filename)
            image = Image.open(img_path).convert('RGB')
            if self.transform:
                image = self.transform(image)
            sequence_images.append(image)

            label_filename = os.path.splitext(frame_filename)[0] + '.txt'
            label_path = os.path.join(scene_path, 'labels', seq_name, label_filename)

            boxes, labels = [], []
            if os.path.exists(label_path):
                with open(label_path, 'r') as f:
                    for line in f.readlines():
                        parts = line.strip().split()
                        if not parts: continue
                        class_id = int(parts[0])
                        boxes.append([float(p) for p in parts[1:]])
                        labels.append(class_id)
            target = {'boxes': torch.as_tensor(boxes, dtype=torch.float32).reshape(-1, 4),
                      'labels': torch.as_tensor(labels, dtype=torch.int64)}
            sequence_targets.append(target)

        unique_seq_name = f"{os.path.basename(scene_path)}_{seq_name}"
        return unique_seq_name, torch.stack(sequence_images, dim=0), sequence_targets