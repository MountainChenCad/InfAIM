# 文件: models_aim.py (版本 10 - 真正的AIM架构，无复用)
# 核心改动:
# 1. (REFACTOR) 移除所有模块复用，创建完全独立的temporal和spatial分支
# 2. (FEATURE) 每个分支有自己的attention、MLP、norm和adapter
# 3. (ALIGNMENT) 确保数据通路与标准AIM架构完全一致
# 4. (ENHANCEMENT) 实现真正的双分支独立学习

import torch
import torch.nn as nn
from models_infmae_skip4 import infmae_vit_base_patch16
import math
import copy


# ========================= 独立模块定义 =========================
class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


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


class TrueAIMBlock(nn.Module):
    """完全符合AIM架构的实现：无任何模块复用"""

    def __init__(self, original_block, mlp_ratio=0.25, scale=0.5):
        super().__init__()
        self.scale = scale
        dim = original_block.norm1.normalized_shape[0]
        mlp_hidden_dim = int(dim * 4)  # 标准的4倍隐藏层

        # ========================= Temporal分支（完全独立）=========================
        self.temporal_attn = Attention(
            dim=dim,
            num_heads=original_block.attn.num_heads,
            qkv_bias=True,
            qk_scale=None,
            attn_drop=0.,
            proj_drop=0.
        )
        self.temporal_norm1 = nn.LayerNorm(dim)
        self.temporal_mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim)
        self.temporal_norm2 = nn.LayerNorm(dim)
        self.temporal_drop_path = nn.Identity()  # 或使用 DropPath(drop_path_rate)
        self.temporal_adapter = Adapter(dim, mlp_ratio)

        # ========================= Spatial分支（完全独立）=========================
        self.spatial_attn = Attention(
            dim=dim,
            num_heads=original_block.attn.num_heads,
            qkv_bias=True,
            qk_scale=None,
            attn_drop=0.,
            proj_drop=0.
        )
        self.spatial_norm1 = nn.LayerNorm(dim)
        self.spatial_mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim)
        self.spatial_norm2 = nn.LayerNorm(dim)
        self.spatial_drop_path = nn.Identity()  # 或使用 DropPath(drop_path_rate)
        self.spatial_adapter = Adapter(dim, mlp_ratio)

        # ========================= 融合适配器 =========================
        self.fusion_adapter = Adapter(dim, mlp_ratio)

        # 初始化所有新创建的模块
        self._init_weights()

        # 参数分析
        self._analyze_parameters()

    def _init_weights(self):
        """初始化所有新创建的模块权重"""
        for module in [self.temporal_attn, self.spatial_attn]:
            nn.init.xavier_uniform_(module.qkv.weight)
            nn.init.zeros_(module.qkv.bias)
            nn.init.xavier_uniform_(module.proj.weight)
            nn.init.zeros_(module.proj.bias)

        for module in [self.temporal_mlp, self.spatial_mlp]:
            nn.init.xavier_uniform_(module.fc1.weight)
            nn.init.zeros_(module.fc1.bias)
            nn.init.xavier_uniform_(module.fc2.weight)
            nn.init.zeros_(module.fc2.bias)

    def _analyze_parameters(self):
        """分析参数分布"""
        temporal_params = (
                sum(p.numel() for p in self.temporal_attn.parameters()) +
                sum(p.numel() for p in self.temporal_mlp.parameters()) +
                sum(p.numel() for p in self.temporal_norm1.parameters()) +
                sum(p.numel() for p in self.temporal_norm2.parameters()) +
                sum(p.numel() for p in self.temporal_adapter.parameters())
        )

        spatial_params = (
                sum(p.numel() for p in self.spatial_attn.parameters()) +
                sum(p.numel() for p in self.spatial_mlp.parameters()) +
                sum(p.numel() for p in self.spatial_norm1.parameters()) +
                sum(p.numel() for p in self.spatial_norm2.parameters()) +
                sum(p.numel() for p in self.spatial_adapter.parameters())
        )

        fusion_params = sum(p.numel() for p in self.fusion_adapter.parameters())
        total = temporal_params + spatial_params + fusion_params

        print(f"=== True AIM Block Parameters ===")
        print(f"Temporal branch: {temporal_params:,}")
        print(f"Spatial branch:  {spatial_params:,}")
        print(f"Fusion adapter:  {fusion_params:,}")
        print(f"Total:           {total:,}")
        print(f"Temporal/Spatial ratio: {temporal_params / spatial_params:.3f}")
        print(f"================================")

    def forward(self, x, num_frames):
        """
        AIM标准数据流：
        1. Temporal processing
        2. Spatial processing
        3. Fusion
        """
        bt, n, d = x.shape
        b = bt // num_frames
        t = num_frames

        # ========================= Temporal Branch =========================
        # 1. Temporal attention
        x_norm_t = self.temporal_norm1(x)
        x_for_t_attn = x_norm_t.view(b, t, n, d).permute(0, 2, 1, 3).reshape(b * n, t, d)
        t_attn_out = self.temporal_attn(x_for_t_attn)
        t_attn_out_reshaped = t_attn_out.view(b, n, t, d).permute(0, 2, 1, 3).reshape(bt, n, d)
        x_after_temporal_attn = x + self.temporal_drop_path(t_attn_out_reshaped) + self.temporal_adapter(
            t_attn_out_reshaped)

        # 2. Temporal MLP
        x_temporal_mlp = self.temporal_norm2(x_after_temporal_attn)
        temporal_mlp_out = self.temporal_mlp(x_temporal_mlp)
        x_after_temporal = x_after_temporal_attn + self.temporal_drop_path(temporal_mlp_out)

        # ========================= Spatial Branch =========================
        # 3. Spatial attention
        spatial_attn_out = self.spatial_attn(self.spatial_norm1(x_after_temporal))
        x_after_spatial_attn = x_after_temporal + self.spatial_drop_path(spatial_attn_out) + self.spatial_adapter(
            spatial_attn_out)

        # 4. Spatial MLP
        x_spatial_mlp = self.spatial_norm2(x_after_spatial_attn)
        spatial_mlp_out = self.spatial_mlp(x_spatial_mlp)
        x_after_spatial = x_after_spatial_attn + self.spatial_drop_path(spatial_mlp_out)

        # ========================= Fusion =========================
        # 5. Final fusion with adapter
        final_x = x_after_spatial + self.scale * self.fusion_adapter(x_after_spatial)

        return final_x


# =========================================================================

class CenterNetHead(nn.Module):
    def __init__(self, in_channels, head_channels, num_classes):
        super(CenterNetHead, self).__init__()
        self.heatmap = nn.Sequential(
            nn.Conv2d(in_channels, head_channels, kernel_size=3, padding=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(head_channels, num_classes, kernel_size=1, stride=1, padding=0, bias=True)
        )
        self.wh = nn.Sequential(
            nn.Conv2d(in_channels, head_channels, kernel_size=3, padding=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(head_channels, 2, kernel_size=1, stride=1, padding=0, bias=True)
        )
        self.offset = nn.Sequential(
            nn.Conv2d(in_channels, head_channels, kernel_size=3, padding=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(head_channels, 2, kernel_size=1, stride=1, padding=0, bias=True)
        )

    def forward(self, x):
        hm = self.heatmap(x).sigmoid()
        wh = self.wh(x)
        offset = self.offset(x)
        return {'hm': hm, 'wh': wh, 'offset': offset}


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

        print("Enabling full fine-tuning: ALL backbone parameters are now trainable.")

        # 统计原始参数
        original_trainable_params = sum(p.numel() for p in self.backbone.parameters() if p.requires_grad)
        single_block_params = sum(p.numel() for p in self.backbone.blocks3[0].parameters() if p.requires_grad)
        print(f"Original backbone parameters: {original_trainable_params:,}")
        print(f"Single original block parameters: {single_block_params:,}")

        # ========================= 核心修改点 =========================
        # 注入真正的AIM blocks（无任何复用）
        print("Injecting True AIM blocks with NO module reuse...")
        self.backbone.blocks3 = nn.ModuleList([
            TrueAIMBlock(original_block=blk) for blk in self.backbone.blocks3
        ])
        print("True AIM injection complete.")
        # ==========================================================

        # 重新统计参数
        new_trainable_params = sum(p.numel() for p in self.backbone.parameters() if p.requires_grad)
        new_single_block_params = sum(p.numel() for p in self.backbone.blocks3[0].parameters() if p.requires_grad)

        print(f"New backbone parameters: {new_trainable_params:,}")
        print(f"Single True AIM block parameters: {new_single_block_params:,}")
        print(f"Parameter increase per block: {(new_single_block_params / single_block_params):.2f}x")
        print(f"Total parameter increase: {(new_trainable_params / original_trainable_params):.2f}x")

        # 定义检测头
        in_channels_for_head = 384
        self.detection_head = CenterNetHead(in_channels=in_channels_for_head, head_channels=256,
                                            num_classes=self.num_classes)

        # 最终统计
        total_trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"Total trainable parameters in full model: {total_trainable:,}")

    def forward(self, video_clip):
        B, T, C, H, W = video_clip.shape
        frames_flat = video_clip.view(B * T, C, H, W)
        backbone = self.backbone

        # InfMAE forward pass
        x = backbone.patch_embed1(frames_flat)
        for blk in backbone.blocks1: x = blk(x)

        x = backbone.patch_embed2(x)
        for blk in backbone.blocks2: x = blk(x)
        features_for_head = x

        # Stage 3 with True AIM blocks
        x = backbone.patch_embed3(features_for_head).flatten(2).permute(0, 2, 1)
        x = backbone.patch_embed4(x)
        x = x + backbone.pos_embed
        for blk in self.backbone.blocks3:
            x = blk(x, num_frames=T)

        # Detection Head
        head_outputs = self.detection_head(features_for_head)

        outputs = {}
        for key, value in head_outputs.items():
            _, C_out, H_out, W_out = value.shape
            outputs[key] = value.view(B, T, C_out, H_out, W_out)

        return outputs