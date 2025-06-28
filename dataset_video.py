# 文件: dataset_video.py (版本 3 - 支持 train/val 划分和评估模式)

import os
import torch
from torch.utils.data import Dataset
from PIL import Image
import numpy as np


class VideoDetectionDataset(Dataset):
    """
    用于红外视频检测的数据集类。
    支持 'train' 和 'val' 两种模式。
    - train 模式: 返回随机的视频片段 (clips)。
    - val 模式: 返回整个视频序列。
    """

    def __init__(self, data_root, clip_length=8, transform=None, mode='train', val_split=0.2):
        self.data_root = data_root
        self.clip_length = clip_length
        self.transform = transform
        self.mode = mode
        self.image_dir = os.path.join(data_root, 'images')
        self.label_dir = os.path.join(data_root, 'labels')

        all_sequences = sorted(
            [d for d in os.listdir(self.image_dir) if os.path.isdir(os.path.join(self.image_dir, d))])

        # 数据集划分
        split_idx = int(len(all_sequences) * (1 - val_split))
        if mode == 'train':
            self.sequences = all_sequences[:split_idx]
        else:  # 'val' mode
            self.sequences = all_sequences[split_idx:]

        self.data_samples = []
        if self.mode == 'train':
            # 在训练模式下，创建视频片段
            for seq_name in self.sequences:
                seq_path = os.path.join(self.image_dir, seq_name)
                frame_files = sorted(os.listdir(seq_path), key=lambda x: int(os.path.splitext(x)[0].split('_')[-1]))
                num_frames = len(frame_files)

                if num_frames >= clip_length:
                    for start_frame in range(num_frames - clip_length + 1):
                        self.data_samples.append(('clip', seq_name, start_frame, frame_files))
        else:  # 'val' mode
            # 在验证模式下，每个样本是一个完整的序列
            for seq_name in self.sequences:
                seq_path = os.path.join(self.image_dir, seq_name)
                frame_files = sorted(os.listdir(seq_path), key=lambda x: int(os.path.splitext(x)[0].split('_')[-1]))
                self.data_samples.append(('sequence', seq_name, frame_files))

    def __len__(self):
        return len(self.data_samples)

    def __getitem__(self, idx):
        sample_type, seq_name, arg, frame_files = self.data_samples[idx]

        if sample_type == 'clip':
            start_frame = arg
            return self._get_clip(seq_name, start_frame, frame_files)
        else:  # 'sequence'
            return self._get_sequence(seq_name, frame_files)

    def _get_clip(self, seq_name, start_frame, frame_files):
        clip_images = []
        clip_targets = []
        for i in range(self.clip_length):
            frame_idx = start_frame + i
            frame_filename = frame_files[frame_idx]

            # ... (这部分与之前版本相同)
            img_path = os.path.join(self.image_dir, seq_name, frame_filename)
            image = Image.open(img_path).convert('RGB')
            if self.transform:
                image = self.transform(image)
            clip_images.append(image)

            label_filename = os.path.splitext(frame_filename)[0] + '.txt'
            label_path = os.path.join(self.label_dir, seq_name, label_filename)

            boxes = []
            labels = []
            if os.path.exists(label_path):
                with open(label_path, 'r') as f:
                    for line in f.readlines():
                        parts = line.strip().split()
                        if not parts: continue
                        class_id = int(parts[0])
                        cxcywh_box = [float(p) for p in parts[1:]]
                        boxes.append(cxcywh_box)
                        labels.append(class_id)

            target = {
                'boxes': torch.as_tensor(boxes, dtype=torch.float32).reshape(-1, 4),
                'labels': torch.as_tensor(labels, dtype=torch.int64)
            }
            clip_targets.append(target)

        return torch.stack(clip_images, dim=0), clip_targets

    def _get_sequence(self, seq_name, frame_files):
        # 为验证返回整个序列
        sequence_images = []
        sequence_targets = []
        for frame_filename in frame_files:
            # ... (与 _get_clip 几乎相同)
            img_path = os.path.join(self.image_dir, seq_name, frame_filename)
            image = Image.open(img_path).convert('RGB')
            if self.transform:
                image = self.transform(image)
            sequence_images.append(image)

            label_filename = os.path.splitext(frame_filename)[0] + '.txt'
            label_path = os.path.join(self.label_dir, seq_name, label_filename)

            boxes = []
            labels = []
            if os.path.exists(label_path):
                with open(label_path, 'r') as f:
                    for line in f.readlines():
                        parts = line.strip().split()
                        if not parts: continue
                        class_id = int(parts[0])
                        cxcywh_box = [float(p) for p in parts[1:]]
                        boxes.append(cxcywh_box)
                        labels.append(class_id)

            target = {
                'boxes': torch.as_tensor(boxes, dtype=torch.float32).reshape(-1, 4),
                'labels': torch.as_tensor(labels, dtype=torch.int64)
            }
            sequence_targets.append(target)

        return torch.stack(sequence_images, dim=0), sequence_targets