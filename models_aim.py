# models_aim.py

from functools import partial
import torch
import torch.nn as nn
from models_infmae_skip4 import infmae_vit_base_patch16


class Adapter(nn.Module):
    """
    A simple bottleneck adapter module as described in the AIM paper.
    """

    def __init__(self, dim, mlp_ratio=0.25):
        super().__init__()
        hidden_dim = int(dim * mlp_ratio)
        self.fc1 = nn.Linear(dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, dim)
        self.activation = nn.GELU()

        # Initialize weights for identity function at the start
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.zeros_(self.fc1.bias)
        nn.init.zeros_(self.fc2.weight)
        nn.init.zeros_(self.fc2.bias)

    def forward(self, x):
        return self.fc2(self.activation(self.fc1(x)))


class AIMBlock(nn.Module):
    """
    An AIM-adapted Transformer block.
    It wraps a frozen original block and applies AIM's adaptation logic.
    """

    def __init__(self, original_block, mlp_ratio=0.25, scale=0.5):
        super().__init__()
        self.scale = scale
        dim = original_block.norm1.normalized_shape[0]

        # Reference frozen layers from the original block
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

        # --- Temporal Adaptation (Formula 3 from AIM paper) ---
        x_norm_t = self.norm1(x)
        x_for_t_attn = x_norm_t.view(b, t, n, d).permute(0, 2, 1, 3).reshape(b * n, t, d)
        t_attn_out = self.attn(x_for_t_attn)
        t_attn_out_reshaped = t_attn_out.view(b, n, t, d).permute(0, 2, 1, 3).reshape(bt, n, d)

        x_after_t = x + self.t_adapter(t_attn_out_reshaped)

        # --- Spatial Adaptation (Formula 4 from AIM paper) ---
        s_attn_out = self.attn(self.norm1(x_after_t))
        x_after_s = x_after_t + self.s_adapter(s_attn_out)

        # --- Joint Adaptation (Formula 5 from AIM paper) ---
        x_mlp_in = x_after_s
        x_norm2 = self.norm2(x_mlp_in)
        mlp_out = self.mlp(x_norm2)
        mlp_adapter_out = self.mlp_adapter(x_norm2)

        # Combine MLP and its parallel adapter
        final_x = x_mlp_in + self.drop_path(mlp_out) + self.scale * self.drop_path(mlp_adapter_out)

        return final_x


class AIM_InfMAE_Detector(nn.Module):
    """
    Infrared video detector using a frozen InfMAE backbone adapted with AIM.
    """

    def __init__(self, pretrained_infmae_path, num_classes, clip_length=8):
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

        # 3. Inject AIM blocks, replacing the original transformer blocks
        # The new AIMBlocks will have trainable adapters by default.
        self.backbone.blocks3 = nn.ModuleList([
            AIMBlock(original_block=blk) for blk in self.backbone.blocks3
        ])

        # 4. Define the trainable detection head
        embed_dim = self.backbone.pos_embed.shape[-1]
        self.detection_head = nn.Linear(embed_dim, num_classes + 1 + 4)  # +1 for background class

    def forward(self, video_clip):
        B, T, C, H, W = video_clip.shape
        frames_flat = video_clip.view(B * T, C, H, W)

        # --- Replicated forward_encoder from InfMAE (no masking) ---
        backbone = self.backbone

        # Create all-ones masks for CBlocks
        mask_shape_1 = (frames_flat.shape[0], 56, 56)
        mask_for_patch1 = torch.ones(mask_shape_1, device=frames_flat.device).unsqueeze(1)
        mask_shape_2 = (frames_flat.shape[0], 28, 28)
        mask_for_patch2 = torch.ones(mask_shape_2, device=frames_flat.device).unsqueeze(1)

        # Stage 1
        x = backbone.patch_embed1(frames_flat)
        for blk in backbone.blocks1:
            x = blk(x, mask_for_patch1)
        stage1_embed = backbone.stage1_output_decode(x).flatten(2).permute(0, 2, 1)

        # Stage 2
        x = backbone.patch_embed2(x)
        for blk in backbone.blocks2:
            x = blk(x, mask_for_patch2)
        stage2_embed = backbone.stage2_output_decode(x).flatten(2).permute(0, 2, 1)

        # Stage 3 (ViT part with AIM blocks)
        x = backbone.patch_embed3(x).flatten(2).permute(0, 2, 1)
        x = backbone.patch_embed4(x)
        x = x + backbone.pos_embed

        # Apply AIM Transformer blocks
        for blk in backbone.blocks3:
            x = blk(x, num_frames=T)  # Pass T to handle reshaping

        # Add skip connections from earlier stages
        x = x + stage1_embed + stage2_embed
        latent = backbone.norm(x)  # Final features: (B*T, N, D)

        # --- Continue to Detection Head ---
        # Average patch features for each frame
        frame_features = latent.mean(dim=1)  # (B*T, D)
        frame_features = frame_features.view(B, T, -1)  # (B, T, D)

        predictions = self.detection_head(frame_features)
        pred_logits = predictions[..., :self.num_classes + 1]
        pred_boxes = predictions[..., self.num_classes + 1:]

        # Per the original code, apply sigmoid to boxes for stable [0, 1] range
        return {'pred_logits': pred_logits, 'pred_boxes': pred_boxes.sigmoid()}