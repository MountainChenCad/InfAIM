# 文件: dataset_video.py (版本 16 - 修复 0-based 标签处理)
# 核心改动:
# 1. (BUG FIX) 修复了由于错误地假设标签是 1-based 而导致的致命错误。
# 2. (REFACTOR) 彻底移除了所有 `class_id - 1` 的操作。代码现在正确地将文件中的
#    类别ID（0-5）作为 0-based 索引直接使用。
# 3. (REFACTOR) 更新了验证逻辑，以确保 `class_id` 在正确的 `[0, num_classes - 1]` 范围内。

import os
import torch
from torch.utils.data import Dataset
from PIL import Image
import numpy as np
import math

# ... (高斯辅助函数保持不变) ...
def gaussian_radius(det_size, min_overlap=0.7):
    height, width = det_size
    a1=1; b1=-(height+width); c1=(1-min_overlap)*width*height/(1+min_overlap); sq1=np.sqrt(b1**2-4*a1*c1); r1=(b1+sq1)/(2*a1)
    a2=4; b2=2*(height+width); c2=(1-min_overlap)*width*height; sq2=np.sqrt(b2**2-4*a2*c2); r2=(b2+sq2)/(2*a2)
    a3=4*min_overlap; b3=-2*min_overlap*(height+width); c3=(min_overlap-1)*width*height; sq3=np.sqrt(b3**2-4*a3*c3); r3=(b3+sq3)/(2*a3)
    return min(r1, r2, r3)
def gaussian2D(shape, sigma=1):
    m,n=[(ss-1.)/2. for ss in shape]; y,x=np.ogrid[-m:m+1,-n:n+1]; h=np.exp(-(x*x+y*y)/(2*sigma*sigma)); h[h<np.finfo(h.dtype).eps*h.max()]=0
    return h
def draw_umich_gaussian(heatmap, center, radius, k=1):
    diameter=2*radius+1; gaussian=gaussian2D((diameter,diameter),sigma=diameter/6); x,y=int(center[0]),int(center[1]); H,W=heatmap.shape[0:2]
    left,right=min(x,radius),min(W-x,radius+1); top,bottom=min(y,radius),min(H-y,radius+1)
    masked_heatmap=heatmap[y-top:y+bottom,x-left:x+right]; masked_gaussian=gaussian[radius-top:radius+bottom,radius-left:radius+right]
    if min(masked_gaussian.shape)>0 and min(masked_heatmap.shape)>0: np.maximum(masked_heatmap,masked_gaussian*k,out=masked_heatmap)
    return heatmap

class VideoDetectionDataset(Dataset):
    def __init__(self, scene_paths, clip_length=8, transform=None, return_clips=True, num_classes=6, output_res=28):
        self.scene_paths = scene_paths
        self.clip_length = clip_length
        self.transform = transform
        self.return_clips = return_clips
        self.num_classes = num_classes
        self.output_res = output_res
        self.data_samples = self._make_dataset()

    def _make_dataset(self):
        data_samples = []
        for scene_path in self.scene_paths:
            image_dir = os.path.join(scene_path, 'images')
            if not os.path.isdir(image_dir): continue
            sequences = sorted([d for d in os.listdir(image_dir) if os.path.isdir(os.path.join(image_dir, d))])
            for seq_name in sequences:
                seq_path = os.path.join(image_dir, seq_name)
                frame_files = sorted([f for f in os.listdir(seq_path) if f.lower().endswith(('.bmp', '.png', '.jpg', '.jpeg'))], key=lambda x: int(os.path.splitext(x)[0].split('_')[-1]))
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

    def _generate_target_heatmaps(self, boxes, labels):
        max_objs = 512
        hm = np.zeros((self.num_classes, self.output_res, self.output_res), dtype=np.float32)
        wh = np.zeros((max_objs, 2), dtype=np.float32)
        offset = np.zeros((max_objs, 2), dtype=np.float32)
        ind = np.zeros((max_objs), dtype=np.int64)
        ind_mask = np.zeros((max_objs), dtype=np.uint8)
        for k, (box, label) in enumerate(zip(boxes, labels)):
            if k >= max_objs: break
            cx, cy, w, h = box
            ct = np.array([cx * self.output_res, cy * self.output_res], dtype=np.float32)
            ct_int = np.clip(ct.astype(np.int32), 0, self.output_res - 1)
            radius = gaussian_radius((math.ceil(h * self.output_res), math.ceil(w * self.output_res))); radius = max(0, int(radius))
            draw_umich_gaussian(hm[label], ct_int, radius)
            wh[k] = w * self.output_res, h * self.output_res
            ind[k] = ct_int[1] * self.output_res + ct_int[0]
            offset[k] = ct - ct_int
            ind_mask[k] = 1
        return {'hm': torch.from_numpy(hm), 'wh': torch.from_numpy(wh), 'offset': torch.from_numpy(offset), 'ind': torch.from_numpy(ind), 'ind_mask': torch.from_numpy(ind_mask)}

    def _get_clip(self, scene_path, seq_name, start_frame, frame_files):
        clip_images, clip_targets = [], []
        for i in range(self.clip_length):
            frame_idx = start_frame + i; frame_filename = frame_files[frame_idx]
            img_path = os.path.join(scene_path, 'images', seq_name, frame_filename); image = Image.open(img_path).convert('RGB')
            if self.transform: image = self.transform(image)
            clip_images.append(image)
            label_filename = os.path.splitext(frame_filename)[0] + '.txt'; label_path = os.path.join(scene_path, 'labels', seq_name, label_filename)
            boxes, labels = [], []
            if os.path.exists(label_path):
                with open(label_path, 'r') as f:
                    for line in f.readlines():
                        parts = line.strip().split()
                        if not parts: continue
                        class_id = int(parts[0])
                        # ========================= 核心修改点 =========================
                        if 0 <= class_id < self.num_classes:
                            boxes.append([float(p) for p in parts[1:]])
                            labels.append(class_id)
                        # ==========================================================
            target_heatmaps = self._generate_target_heatmaps(boxes, labels); clip_targets.append(target_heatmaps)
        return torch.stack(clip_images, dim=0), clip_targets

    def _get_sequence(self, scene_path, seq_name, frame_files):
        sequence_images, sequence_targets = [], []
        for frame_filename in frame_files:
            img_path = os.path.join(scene_path, 'images', seq_name, frame_filename); image = Image.open(img_path).convert('RGB')
            if self.transform: image = self.transform(image)
            sequence_images.append(image)
            label_filename = os.path.splitext(frame_filename)[0] + '.txt'; label_path = os.path.join(scene_path, 'labels', seq_name, label_filename)
            boxes, labels = [], [];
            if os.path.exists(label_path):
                with open(label_path, 'r') as f:
                    for line in f.readlines():
                        parts = line.strip().split()
                        if not parts: continue
                        class_id = int(parts[0])
                        # ========================= 核心修改点 =========================
                        if 0 <= class_id < self.num_classes:
                            boxes.append([float(p) for p in parts[1:]])
                            labels.append(class_id)
                        # ==========================================================
            target = {'boxes': torch.as_tensor(boxes, dtype=torch.float32).reshape(-1, 4), 'labels': torch.as_tensor(labels, dtype=torch.int64)}
            sequence_targets.append(target)
        unique_seq_name = f"{os.path.basename(scene_path)}_{seq_name}"
        return unique_seq_name, torch.stack(sequence_images, dim=0), sequence_targets

    def __getitem__(self, idx):
        sample_info = self.data_samples[idx]
        sample_type, scene_path, seq_name = sample_info[:3]
        if sample_type == 'clip':
            _, _, _, start_frame, frame_files = sample_info
            return self._get_clip(scene_path, seq_name, start_frame, frame_files)
        else: # sequence
            _, _, _, frame_files = sample_info
            return self._get_sequence(scene_path, seq_name, frame_files)