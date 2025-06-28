# 文件: models_aim.py

from functools import partial
import torch
import torch.nn as nn
from models_infmae_skip4 import infmae_vit_base_patch16
from vision_transformer import Block


class TemporalAdapter(nn.Module):
    # ... (无变化)
    def __init__(self, dim, num_heads=8, mlp_ratio=4.0):
        super().__init__()
        self.attn = nn.MultiheadAttention(dim, num_heads)
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)

        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_hidden_dim),
            nn.GELU(),
            nn.Linear(mlp_hidden_dim, dim)
        )

    def forward(self, x):
        B, T, N, C = x.shape
        x_reshaped = x.view(B * N, T, C).permute(1, 0, 2)
        attn_out, _ = self.attn(x_reshaped, x_reshaped, x_reshaped)
        x = x + attn_out.permute(1, 0, 2).view(B, N, T, C).permute(0, 2, 1, 3)
        x = self.norm1(x)
        x = x + self.mlp(x)
        x = self.norm2(x)
        return x


class AIM_InfMAE_Detector(nn.Module):
    # ... (init 部分无变化)
    def __init__(self, pretrained_infmae_path, num_classes, clip_length=8):
        super().__init__()
        self.num_classes = num_classes
        self.clip_length = clip_length

        self.backbone = infmae_vit_base_patch16(norm_pix_loss=False)
        print(f"Loading pretrained weights from {pretrained_infmae_path}")
        checkpoint = torch.load(pretrained_infmae_path, map_location='cpu')
        self.backbone.load_state_dict(checkpoint['model'], strict=False)
        print("Pretrained weights loaded successfully.")

        for name, param in self.backbone.named_parameters():
            param.requires_grad = False

        embed_dim = self.backbone.pos_embed.shape[-1]
        self.temporal_adapter = TemporalAdapter(dim=embed_dim)
        self.detection_head = nn.Linear(embed_dim, num_classes + 1 + 4)

    def forward(self, video_clip):
        B, T, C, H, W = video_clip.shape
        frames_flat = video_clip.view(B * T, C, H, W)
        latent, _, _ = self.backbone.forward_encoder(frames_flat, mask_ratio=0.0)

        N = latent.shape[1]
        features_temporal = latent.view(B, T, N, -1)
        adapted_features = self.temporal_adapter(features_temporal)
        frame_features = adapted_features.mean(dim=2)

        predictions = self.detection_head(frame_features)
        pred_logits = predictions[..., :self.num_classes + 1]
        pred_boxes = predictions[..., self.num_classes + 1:]

        # ================================== 核心修复点 ==================================
        # 模型输出的是 (cx, cy, w, h)，并用 sigmoid 确保它们在 [0, 1] 范围内
        # 这是一个更稳定的表示法
        return {'pred_logits': pred_logits.softmax(-1), 'pred_boxes': pred_boxes.sigmoid()}
        # ==============================================================================