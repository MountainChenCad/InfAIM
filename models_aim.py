# 文件: models_aim.py (版本 3 - LoRA 微调)
# 核心改动:
# 1. (REFACTOR) 完全移除了之前的 `Adapter` 和 `AIMBlock` 类。
# 2. (FEATURE) 实现了标准的 LoRA (Low-Rank Adaptation) 微调方法。
#    - `LoraLayer`: 定义了 LoRA 的核心低秩矩阵 A 和 B。
#    - `LoraInjectedLinear`: 一个包装器，将 LoraLayer "注入" 到一个冻结的 nn.Linear 层中。
# 3. (REFACTOR) 在 `AIM_InfMAE_Detector` 的初始化过程中，我们现在遍历 Transformer blocks，
#    并将 `qkv` 和 `proj` 线性层替换为 `LoraInjectedLinear` 包装器。这是一种更简洁、更标准的 PEFT 方法。
# 4. (REFACTOR) `forward` 传递过程被简化，以匹配原始的 InfMAE 骨干网络，从而移除了复杂的 AIMBlock 逻辑。

from functools import partial
import torch
import torch.nn as nn
import math
from models_infmae_skip4 import infmae_vit_base_patch16


class LoraLayer(nn.Module):
    """
    LoRA (Low-Rank Adaptation) a a single linear layer.
    """

    def __init__(self, in_features, out_features, r, alpha):
        super().__init__()
        self.r = r
        self.alpha = alpha

        self.lora_A = nn.Linear(in_features, r, bias=False)
        self.lora_B = nn.Linear(r, out_features, bias=False)

        # Initialize A with Kaiming uniform and B with zeros
        nn.init.kaiming_uniform_(self.lora_A.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B.weight)

    def forward(self, x):
        return self.lora_B(self.lora_A(x)) * self.alpha


class LoraInjectedLinear(nn.Module):
    """
    A wrapper for a linear layer that injects LoRA.
    """

    def __init__(self, original_layer, r, alpha):
        super().__init__()
        self.original_layer = original_layer  # This layer is frozen

        self.lora_layer = LoraLayer(
            original_layer.in_features,
            original_layer.out_features,
            r,
            alpha
        )

    def forward(self, x):
        # The original layer is in eval mode and frozen
        original_output = self.original_layer(x)
        # The LoRA layer is in train mode and trainable
        lora_output = self.lora_layer(x)
        return original_output + lora_output


class AIM_InfMAE_Detector(nn.Module):
    """
    Infrared video detector using a frozen InfMAE backbone adapted with LoRA.
    """

    def __init__(self, pretrained_infmae_path, num_classes, clip_length=8, lora_rank=16, lora_alpha=16):
        super().__init__()
        self.num_classes = num_classes
        self.clip_length = clip_length

        # 1. Load backbone
        self.backbone = infmae_vit_base_patch16(norm_pix_loss=False)
        print(f"Loading pretrained weights from {pretrained_infmae_path}")
        checkpoint = torch.load(pretrained_infmae_path, map_location='cpu')
        self.backbone.load_state_dict(checkpoint['model'], strict=False)
        print("Pretrained weights loaded successfully.")

        # 2. Freeze all backbone parameters
        for param in self.backbone.parameters():
            param.requires_grad = False

        # 3. Inject LoRA layers into the attention blocks
        for block in self.backbone.blocks3:
            block.attn.qkv = LoraInjectedLinear(block.attn.qkv, r=lora_rank, alpha=lora_alpha)
            block.attn.proj = LoraInjectedLinear(block.attn.proj, r=lora_rank, alpha=lora_alpha)

        # 4. Define the trainable detection head
        embed_dim = self.backbone.pos_embed.shape[-1]
        self.detection_head = nn.Linear(embed_dim, num_classes + 1 + 4)  # +1 for background

    def forward(self, video_clip):
        B, T, C, H, W = video_clip.shape
        frames_flat = video_clip.view(B * T, C, H, W)
        backbone = self.backbone

        # --- Standard InfMAE forward pass (no masking) ---
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

        # Apply the original ViT blocks (now with injected LoRA layers)
        for blk in backbone.blocks3:
            x = blk(x)

        x = x + stage1_embed + stage2_embed
        latent = backbone.norm(x)  # Final features: (B*T, N, D)

        # --- Detection Head ---
        predictions = self.detection_head(latent)  # (B*T, N, num_classes + 1 + 4)

        num_patches = predictions.shape[1]
        predictions = predictions.view(B, T, num_patches, -1)

        pred_logits = predictions[..., :self.num_classes + 1]
        pred_boxes = predictions[..., self.num_classes + 1:]
        return {'pred_logits': pred_logits, 'pred_boxes': pred_boxes.sigmoid()}