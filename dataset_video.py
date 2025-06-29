# 文件: dataset_video.py (版本 6 - 支持全序列可视化)
# 核心改动:
# 1. 在 __init__ 中添加 `return_clips` 参数。这允许我们为同一个数据集(例如训练集)
#    创建两种加载器：一种返回用于训练的片段，另一种返回用于可视化的完整序列。
# 2. 修改 _get_sequence，使其同时返回序列名，以便在保存可视化文件时使用。

import os
import torch
from torch.utils.data import Dataset
from PIL import Image
import numpy as np


class VideoDetectionDataset(Dataset):
    """
    用于红外视频检测的数据集类。
    支持 'train' 和 'val' 两种模式。
    """

    def __init__(self, data_root, clip_length=8, transform=None, mode='train', return_clips=True):
        self.data_root = data_root
        self.clip_length = clip_length
        self.transform = transform
        self.mode = mode
        self.image_dir = os.path.join(data_root, 'images')
        self.label_dir = os.path.join(data_root, 'labels')

        all_sequences = sorted(
            [d for d in os.listdir(self.image_dir) if os.path.isdir(os.path.join(self.image_dir, d))])

        val_sequences_names = ['data01', 'data07', 'data10', 'data20']
        if mode == 'train':
            self.sequences = [seq for seq in all_sequences if seq not in val_sequences_names]
        else:
            self.sequences = val_sequences_names

        self.data_samples = []
        # 根据 return_clips 标志决定是生成片段还是完整序列
        if mode == 'train' and return_clips:
            # 训练模式: 创建视频片段
            for seq_name in self.sequences:
                seq_path = os.path.join(self.image_dir, seq_name)
                frame_files = sorted(os.listdir(seq_path), key=lambda x: int(os.path.splitext(x)[0].split('_')[-1]))
                num_frames = len(frame_files)
                if num_frames >= clip_length:
                    for start_frame in range(num_frames - clip_length + 1):
                        self.data_samples.append(('clip', seq_name, start_frame, frame_files))
        else:
            # 验证或可视化模式: 每个样本是一个完整序列
            for seq_name in self.sequences:
                seq_path = os.path.join(self.image_dir, seq_name)
                frame_files = sorted(os.listdir(seq_path), key=lambda x: int(os.path.splitext(x)[0].split('_')[-1]))
                self.data_samples.append(('sequence', seq_name, frame_files))

    def __len__(self):
        return len(self.data_samples)

    def __getitem__(self, idx):
        sample_info = self.data_samples[idx]
        sample_type = sample_info[0]

        if sample_type == 'clip':
            _, seq_name, start_frame, frame_files = sample_info
            return self._get_clip(seq_name, start_frame, frame_files)
        elif sample_type == 'sequence':
            _, seq_name, frame_files = sample_info
            return self._get_sequence(seq_name, frame_files)
        else:
            raise ValueError(f"Unknown sample type: {sample_type}")

    def _get_clip(self, seq_name, start_frame, frame_files):
        clip_images = []
        clip_targets = []
        for i in range(self.clip_length):
            frame_idx = start_frame + i
            frame_filename = frame_files[frame_idx]
            img_path = os.path.join(self.image_dir, seq_name, frame_filename)
            image = Image.open(img_path).convert('RGB')
            if self.transform:
                image = self.transform(image)
            clip_images.append(image)
            label_filename = os.path.splitext(frame_filename)[0] + '.txt'
            label_path = os.path.join(self.label_dir, seq_name, label_filename)
            boxes, labels = [], []
            if os.path.exists(label_path):
                with open(label_path, 'r') as f:
                    for line in f.readlines():
                        parts = line.strip().split()
                        if not parts: continue
                        class_id = int(parts[0])
                        boxes.append([float(p) for p in parts[1:]])
                        labels.append(class_id)
            target = {'boxes': torch.as_tensor(boxes, dtype=torch.float32).reshape(-1, 4), 'labels': torch.as_tensor(labels, dtype=torch.int64)}
            clip_targets.append(target)
        return torch.stack(clip_images, dim=0), clip_targets

    def _get_sequence(self, seq_name, frame_files):
        sequence_images, sequence_targets = [], []
        for frame_filename in frame_files:
            img_path = os.path.join(self.image_dir, seq_name, frame_filename)
            image = Image.open(img_path).convert('RGB')
            if self.transform:
                image = self.transform(image)
            sequence_images.append(image)
            label_filename = os.path.splitext(frame_filename)[0] + '.txt'
            label_path = os.path.join(self.label_dir, seq_name, label_filename)
            boxes, labels = [], []
            if os.path.exists(label_path):
                with open(label_path, 'r') as f:
                    for line in f.readlines():
                        parts = line.strip().split()
                        if not parts: continue
                        class_id = int(parts[0])
                        boxes.append([float(p) for p in parts[1:]])
                        labels.append(class_id)
            target = {'boxes': torch.as_tensor(boxes, dtype=torch.float32).reshape(-1, 4), 'labels': torch.as_tensor(labels, dtype=torch.int64)}
            sequence_targets.append(target)
        # 返回序列名、图像序列和目标序列
        return seq_name, torch.stack(sequence_images, dim=0), sequence_targets