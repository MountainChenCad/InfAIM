# models_aim.py (版本 2 - 支持多目标检测)
# 核心改动:
# 1. (FEATURE) 移除了 `latent.mean(dim=1)`。这是关键的架构更改。
# 2. (FEATURE) 现在将检测头应用于每个 patch token，而不仅仅是平均后的特征。
#    这使得模型能够为每个 patch token 输出一个独立的预测，从而实现每帧多目标检测。

from functools import partial
import torch
import torch.nn as nn
from models_infmae_skip4 import infmae_vit_base_patch16


class Adapter(nn.Module):
    def __init__(self, dim, mlp_ratio=0.25):
        super().__init__()
        hidden_dim = int(dim * mlp_ratio)
        self.fc1 = nn.Linear(dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, dim)
        self.activation = nn.GELU()
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.zeros_(self.fc1.bias)
        nn.init.zeros_(self.fc2.weight)
        nn.init.zeros_(self.fc2.bias)

    def forward(self, x):
        return self.fc2(self.activation(self.fc1(x)))


class AIMBlock(nn.Module):
    def __init__(self, original_block, mlp_ratio=0.25, scale=0.5):
        super().__init__()
        self.scale = scale
        dim = original_block.norm1.normalized_shape[0]
        self.attn = original_block.attn
        self.norm1 = original_block.norm1
        self.mlp = original_block.mlp
        self.norm2 = original_block.norm2
        self.drop_path = original_block.drop_path
        self.t_adapter = Adapter(dim, mlp_ratio)
        self.s_adapter = Adapter(dim, mlp_ratio)
        self.mlp_adapter = Adapter(dim, mlp_ratio)

    def forward(self, x, num_frames):
        bt, n, d = x.shape
        b = bt // num_frames
        t = num_frames
        x_norm_t = self.norm1(x)
        x_for_t_attn = x_norm_t.view(b, t, n, d).permute(0, 2, 1, 3).reshape(b * n, t, d)
        t_attn_out = self.attn(x_for_t_attn)
        t_attn_out_reshaped = t_attn_out.view(b, n, t, d).permute(0, 2, 1, 3).reshape(bt, n, d)
        x_after_t = x + self.t_adapter(t_attn_out_reshaped)
        s_attn_out = self.attn(self.norm1(x_after_t))
        x_after_s = x_after_t + self.s_adapter(s_attn_out)
        x_mlp_in = x_after_s
        x_norm2 = self.norm2(x_mlp_in)
        mlp_out = self.mlp(x_norm2)
        mlp_adapter_out = self.mlp_adapter(x_norm2)
        final_x = x_mlp_in + self.drop_path(mlp_out) + self.scale * self.drop_path(mlp_adapter_out)
        return final_x


class AIM_InfMAE_Detector(nn.Module):
    def __init__(self, pretrained_infmae_path, num_classes, clip_length=8):
        super().__init__()
        self.num_classes = num_classes
        self.clip_length = clip_length
        self.backbone = infmae_vit_base_patch16(norm_pix_loss=False)
        print(f"Loading pretrained weights from {pretrained_infmae_path}")
        checkpoint = torch.load(pretrained_infmae_path, map_location='cpu')
        self.backbone.load_state_dict(checkpoint['model'], strict=False)
        print("Pretrained weights loaded successfully.")
        for param in self.backbone.parameters():
            param.requires_grad = False
        self.backbone.blocks3 = nn.ModuleList([
            AIMBlock(original_block=blk) for blk in self.backbone.blocks3
        ])
        embed_dim = self.backbone.pos_embed.shape[-1]
        self.detection_head = nn.Linear(embed_dim, num_classes + 1 + 4)

    def forward(self, video_clip):
        B, T, C, H, W = video_clip.shape
        frames_flat = video_clip.view(B * T, C, H, W)
        backbone = self.backbone
        mask_shape_1 = (frames_flat.shape[0], 56, 56)
        mask_for_patch1 = torch.ones(mask_shape_1, device=frames_flat.device).unsqueeze(1)
        mask_shape_2 = (frames_flat.shape[0], 28, 28)
        mask_for_patch2 = torch.ones(mask_shape_2, device=frames_flat.device).unsqueeze(1)
        x = backbone.patch_embed1(frames_flat)
        for blk in backbone.blocks1: x = blk(x, mask_for_patch1)
        stage1_embed = backbone.stage1_output_decode(x).flatten(2).permute(0, 2, 1)
        x = backbone.patch_embed2(x)
        for blk in backbone.blocks2: x = blk(x, mask_for_patch2)
        stage2_embed = backbone.stage2_output_decode(x).flatten(2).permute(0, 2, 1)
        x = backbone.patch_embed3(x).flatten(2).permute(0, 2, 1)
        x = backbone.patch_embed4(x)
        x = x + backbone.pos_embed
        for blk in backbone.blocks3: x = blk(x, num_frames=T)
        x = x + stage1_embed + stage2_embed
        latent = backbone.norm(x)

        # ========================= 核心修改点 =========================
        # 移除 latent.mean(dim=1)，将检测头应用于每个 patch token
        # latent shape: (B*T, N, D), where N=196
        predictions = self.detection_head(latent)  # (B*T, N, num_classes + 1 + 4)

        # Reshape to (B, T, N, ...)
        num_patches = predictions.shape[1]
        predictions = predictions.view(B, T, num_patches, -1)
        # ==============================================================

        pred_logits = predictions[..., :self.num_classes + 1]
        pred_boxes = predictions[..., self.num_classes + 1:]
        return {'pred_logits': pred_logits, 'pred_boxes': pred_boxes.sigmoid()}