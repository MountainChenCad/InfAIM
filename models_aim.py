# 文件: models_aim.py (版本 7 - 终极架构: AIM主干 + CenterNet头)
# 核心改动:
# 1. (FEATURE) 实现了理论上最优的混合架构，结合了两种设计的优点：
#    - AIM + Adapter: 用于主干网络，专门为视频任务优化，解耦时空注意力。
#    - CenterNet Head: 用于检测头，专门为微小和密集目标检测优化。
# 2. (REFACTOR) 重新引入了 `Adapter` 和 `AIMBlock` 的定义。
# 3. (REFACTOR) 在模型初始化时，使用 `AIMBlock` 替换了骨干网络的 `blocks3`，
#    同时保留了可训练的 `CenterNetHead`。
# 4. (REFACTOR) `forward` 传递过程被更新，以正确地将第二阶段的特征送入检测头，
#    同时完成整个骨干网络的时空计算。

import torch
import torch.nn as nn
from models_infmae_skip4 import infmae_vit_base_patch16
import math


# ========================= AIM + Adapter 模块定义 =========================
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
        # Reference frozen layers
        self.attn = original_block.attn
        self.norm1 = original_block.norm1
        self.mlp = original_block.mlp
        self.norm2 = original_block.norm2
        self.drop_path = original_block.drop_path
        # Create new trainable adapters
        self.t_adapter = Adapter(dim, mlp_ratio)
        self.s_adapter = Adapter(dim, mlp_ratio)
        self.mlp_adapter = Adapter(dim, mlp_ratio)

    def forward(self, x, num_frames):
        bt, n, d = x.shape
        b = bt // num_frames
        t = num_frames
        # Temporal Adaptation
        x_norm_t = self.norm1(x)
        x_for_t_attn = x_norm_t.view(b, t, n, d).permute(0, 2, 1, 3).reshape(b * n, t, d)
        t_attn_out = self.attn(x_for_t_attn)
        t_attn_out_reshaped = t_attn_out.view(b, n, t, d).permute(0, 2, 1, 3).reshape(bt, n, d)
        x_after_t = x + self.t_adapter(t_attn_out_reshaped)
        # Spatial Adaptation
        s_attn_out = self.attn(self.norm1(x_after_t))
        x_after_s = x_after_t + self.s_adapter(s_attn_out)
        # Joint Adaptation
        x_mlp_in = x_after_s
        x_norm2 = self.norm2(x_mlp_in)
        mlp_out = self.mlp(x_norm2)
        mlp_adapter_out = self.mlp_adapter(x_norm2)
        final_x = x_mlp_in + self.drop_path(mlp_out) + self.scale * self.drop_path(mlp_adapter_out)
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

        # 1. Freeze all original backbone parameters
        for param in self.backbone.parameters():
            param.requires_grad = False

        # 2. Inject AIMBlocks with trainable adapters into Stage 3
        print("Injecting AIM blocks with trainable adapters...")
        self.backbone.blocks3 = nn.ModuleList([
            AIMBlock(original_block=blk) for blk in self.backbone.blocks3
        ])
        print("AIM injection complete.")

        # 3. Define the trainable CenterNet detection head
        in_channels_for_head = 384  # Stage 2 feature map dimension
        self.detection_head = CenterNetHead(in_channels=in_channels_for_head, head_channels=256,
                                            num_classes=self.num_classes)

    def forward(self, video_clip):
        B, T, C, H, W = video_clip.shape
        frames_flat = video_clip.view(B * T, C, H, W)
        backbone = self.backbone

        # --- InfMAE forward pass ---
        # Stage 1
        x = backbone.patch_embed1(frames_flat)
        for blk in backbone.blocks1: x = blk(x)  # No mask needed

        # Stage 2
        x = backbone.patch_embed2(x)
        for blk in backbone.blocks2: x = blk(x)  # No mask needed
        features_for_head = x  # Features from Stage 2 go to the head

        # Stage 3 (ViT part with AIM blocks)
        x = backbone.patch_embed3(features_for_head).flatten(2).permute(0, 2, 1)
        x = backbone.patch_embed4(x)
        x = x + backbone.pos_embed
        for blk in self.backbone.blocks3:
            x = blk(x, num_frames=T)  # Pass T to AIMBlock

        # --- Detection Head ---
        head_outputs = self.detection_head(features_for_head)

        outputs = {}
        for key, value in head_outputs.items():
            _, C_out, H_out, W_out = value.shape
            outputs[key] = value.view(B, T, C_out, H_out, W_out)

        return outputs