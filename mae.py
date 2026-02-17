"""
Masked Autoencoder (MAE) for Vision Transformer Pretraining

Based on "Masked Autoencoders Are Scalable Vision Learners" (He et al., 2021)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
import numpy as np

from vit import PatchEmbedding, TransformerBlock


class MAE(nn.Module):
    """
    Masked Autoencoder for self-supervised pretraining.

    Architecture:
    1. Patchify image into N patches
    2. Randomly mask 75% of patches
    3. Encode only visible patches (25%) with ViT encoder
    4. Decode all patches (visible + masked) with lightweight decoder
    5. Reconstruct masked patches and compute MSE loss
    """

    def __init__(
        self,
        img_size=256,
        patch_size=32,
        in_channels=1,
        # Encoder config
        encoder_embed_dim=768,
        encoder_num_layers=12,
        encoder_num_heads=12,
        encoder_mlp_dim=3072,
        # Decoder config
        decoder_embed_dim=512,
        decoder_num_layers=4,
        decoder_num_heads=8,
        decoder_mlp_dim=2048,
        # MAE config
        mask_ratio=0.75,
        dropout=0.0,
        norm_pix_loss=True
    ):
        super().__init__()

        self.img_size = img_size
        self.patch_size = patch_size
        self.in_channels = in_channels
        self.mask_ratio = mask_ratio
        self.norm_pix_loss = norm_pix_loss

        self.num_patches = (img_size // patch_size) ** 2
        self.patch_dim = patch_size * patch_size * in_channels

        # ---------- Encoder ----------
        # Patch embedding
        self.patch_embed = PatchEmbedding(
            img_size, patch_size, in_channels, encoder_embed_dim
        )

        # Class token
        self.cls_token = nn.Parameter(torch.zeros(1, 1, encoder_embed_dim))

        # Position embeddings for encoder (without class token position)
        self.encoder_pos_embed = nn.Parameter(
            torch.zeros(1, self.num_patches + 1, encoder_embed_dim)
        )

        # Encoder transformer blocks
        self.encoder_blocks = nn.ModuleList([
            TransformerBlock(encoder_embed_dim, encoder_num_heads, encoder_mlp_dim, dropout)
            for _ in range(encoder_num_layers)
        ])

        self.encoder_norm = nn.LayerNorm(encoder_embed_dim)

        # ---------- Decoder ----------
        # Project encoder output to decoder dimension
        self.decoder_embed = nn.Linear(encoder_embed_dim, decoder_embed_dim, bias=True)

        # Learnable mask token
        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))

        # Position embeddings for decoder
        self.decoder_pos_embed = nn.Parameter(
            torch.zeros(1, self.num_patches + 1, decoder_embed_dim)
        )

        # Decoder transformer blocks
        self.decoder_blocks = nn.ModuleList([
            TransformerBlock(decoder_embed_dim, decoder_num_heads, decoder_mlp_dim, dropout)
            for _ in range(decoder_num_layers)
        ])

        self.decoder_norm = nn.LayerNorm(decoder_embed_dim)

        # Reconstruction head (project back to patch pixels)
        self.decoder_pred = nn.Linear(decoder_embed_dim, self.patch_dim, bias=True)

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        # Initialize position embeddings
        nn.init.trunc_normal_(self.encoder_pos_embed, std=0.02)
        nn.init.trunc_normal_(self.decoder_pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.mask_token, std=0.02)

        # Initialize other layers
        self.apply(self._init_layer_weights)

    def _init_layer_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.LayerNorm):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)

    def patchify(self, imgs):
        """
        Convert images to patches.

        Args:
            imgs: (B, C, H, W)
        Returns:
            patches: (B, num_patches, patch_dim)
        """
        patches = rearrange(
            imgs,
            'b c (h p1) (w p2) -> b (h w) (p1 p2 c)',
            p1=self.patch_size,
            p2=self.patch_size
        )
        return patches

    def unpatchify(self, patches):
        """
        Reconstruct images from patches.

        Args:
            patches: (B, num_patches, patch_dim)
        Returns:
            imgs: (B, C, H, W)
        """
        h = w = int(self.num_patches ** 0.5)
        imgs = rearrange(
            patches,
            'b (h w) (p1 p2 c) -> b c (h p1) (w p2)',
            h=h, w=w,
            p1=self.patch_size,
            p2=self.patch_size,
            c=self.in_channels
        )
        return imgs

    def random_masking(self, x, mask_ratio):
        """
        Perform per-sample random masking.

        Args:
            x: (B, N, D) - patch embeddings
            mask_ratio: fraction of patches to mask

        Returns:
            x_masked: (B, N_visible, D) - visible patches only
            mask: (B, N) - binary mask, 0 = keep, 1 = remove
            ids_restore: (B, N) - indices to restore original order
        """
        B, N, D = x.shape
        num_keep = int(N * (1 - mask_ratio))

        # Generate random noise for each sample
        noise = torch.rand(B, N, device=x.device)

        # Sort noise to get shuffled indices
        ids_shuffle = torch.argsort(noise, dim=1)  # ascend: small is keep, large is remove
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        # Keep first num_keep indices
        ids_keep = ids_shuffle[:, :num_keep]

        # Gather visible patches
        x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).expand(-1, -1, D))

        # Generate binary mask: 0 = keep, 1 = remove
        mask = torch.ones(B, N, device=x.device)
        mask[:, :num_keep] = 0
        mask = torch.gather(mask, dim=1, index=ids_restore)

        return x_masked, mask, ids_restore

    def forward_encoder(self, x, mask_ratio):
        """
        Encode visible patches only.

        Args:
            x: (B, C, H, W) - input images
            mask_ratio: fraction of patches to mask

        Returns:
            latent: (B, N_visible + 1, D) - encoded features (with CLS token)
            mask: (B, N) - binary mask
            ids_restore: (B, N) - restoration indices
        """
        # Embed patches
        x = self.patch_embed(x)  # (B, N, D)

        # Add position embeddings (without CLS position)
        x = x + self.encoder_pos_embed[:, 1:, :]

        # Random masking
        x, mask, ids_restore = self.random_masking(x, mask_ratio)

        # Add class token
        cls_token = self.cls_token + self.encoder_pos_embed[:, :1, :]
        cls_tokens = repeat(cls_token, '1 1 d -> b 1 d', b=x.shape[0])
        x = torch.cat([cls_tokens, x], dim=1)  # (B, N_visible + 1, D)

        # Apply transformer blocks
        for block in self.encoder_blocks:
            x = block(x)

        x = self.encoder_norm(x)

        return x, mask, ids_restore

    def forward_decoder(self, x, ids_restore):
        """
        Decode all patches (visible + masked).

        Args:
            x: (B, N_visible + 1, D_enc) - encoder output
            ids_restore: (B, N) - indices to restore original order

        Returns:
            pred: (B, N, patch_dim) - reconstructed patches
        """
        # Project to decoder dimension
        x = self.decoder_embed(x)  # (B, N_visible + 1, D_dec)

        # Create mask tokens for all masked positions
        mask_tokens = repeat(
            self.mask_token,
            '1 1 d -> b n d',
            b=x.shape[0],
            n=ids_restore.shape[1] + 1 - x.shape[1]  # num_masked
        )

        # Append mask tokens to visible tokens (excluding CLS)
        x_no_cls = x[:, 1:, :]  # (B, N_visible, D_dec)
        x_full = torch.cat([x_no_cls, mask_tokens], dim=1)  # (B, N, D_dec)

        # Unshuffle to restore original order
        x_full = torch.gather(
            x_full,
            dim=1,
            index=ids_restore.unsqueeze(-1).expand(-1, -1, x_full.shape[2])
        )

        # Add back CLS token
        x = torch.cat([x[:, :1, :], x_full], dim=1)  # (B, N + 1, D_dec)

        # Add position embeddings
        x = x + self.decoder_pos_embed

        # Apply transformer blocks
        for block in self.decoder_blocks:
            x = block(x)

        x = self.decoder_norm(x)

        # Predict pixel values (remove CLS token)
        pred = self.decoder_pred(x[:, 1:, :])  # (B, N, patch_dim)

        return pred

    def forward_loss(self, imgs, pred, mask):
        """
        Compute MSE loss on masked patches only.

        Args:
            imgs: (B, C, H, W) - original images
            pred: (B, N, patch_dim) - predicted patches
            mask: (B, N) - binary mask (1 = masked, 0 = visible)

        Returns:
            loss: scalar
        """
        target = self.patchify(imgs)  # (B, N, patch_dim)

        if self.norm_pix_loss:
            # Normalize target by patch mean and variance
            mean = target.mean(dim=-1, keepdim=True)
            var = target.var(dim=-1, keepdim=True)
            target = (target - mean) / (var + 1e-6).sqrt()

        # MSE loss
        loss = (pred - target) ** 2
        loss = loss.mean(dim=-1)  # (B, N) - mean loss per patch

        # Only compute loss on masked patches
        loss = (loss * mask).sum() / mask.sum()

        return loss

    def forward(self, imgs, mask_ratio=None):
        """
        Forward pass for MAE pretraining.

        Args:
            imgs: (B, C, H, W) - input images
            mask_ratio: optional override for mask ratio

        Returns:
            loss: reconstruction loss
            pred: predicted patches
            mask: binary mask
        """
        if mask_ratio is None:
            mask_ratio = self.mask_ratio

        # Encode visible patches
        latent, mask, ids_restore = self.forward_encoder(imgs, mask_ratio)

        # Decode all patches
        pred = self.forward_decoder(latent, ids_restore)

        # Compute loss on masked patches
        loss = self.forward_loss(imgs, pred, mask)

        return loss, pred, mask

    def get_encoder_state_dict(self):
        """
        Extract encoder weights for fine-tuning.

        Returns a state dict compatible with ViTEncoder.
        """
        encoder_state = {}

        # Patch embedding
        encoder_state['patch_embed.projection.weight'] = self.patch_embed.projection.weight.data
        encoder_state['patch_embed.projection.bias'] = self.patch_embed.projection.bias.data

        # Class token
        encoder_state['cls_token'] = self.cls_token.data

        # Position embeddings
        encoder_state['pos_embed'] = self.encoder_pos_embed.data

        # Transformer blocks
        for i, block in enumerate(self.encoder_blocks):
            for name, param in block.named_parameters():
                encoder_state[f'blocks.{i}.{name}'] = param.data

        # Final norm
        encoder_state['norm.weight'] = self.encoder_norm.weight.data
        encoder_state['norm.bias'] = self.encoder_norm.bias.data

        return encoder_state

    @torch.no_grad()
    def reconstruct(self, imgs, mask_ratio=None):
        """
        Reconstruct images for visualization.

        Args:
            imgs: (B, C, H, W) - input images
            mask_ratio: optional override

        Returns:
            original: (B, C, H, W) - original images
            reconstructed: (B, C, H, W) - reconstructed images
            masked: (B, C, H, W) - masked input (visible patches only)
            mask: (B, N) - binary mask
        """
        if mask_ratio is None:
            mask_ratio = self.mask_ratio

        # Forward pass
        loss, pred, mask = self.forward(imgs, mask_ratio)

        # Get original patches
        orig_patches = self.patchify(imgs)

        # If normalized pixel loss, denormalize predictions
        if self.norm_pix_loss:
            mean = orig_patches.mean(dim=-1, keepdim=True)
            var = orig_patches.var(dim=-1, keepdim=True)
            pred = pred * (var + 1e-6).sqrt() + mean

        # Merge: use pred only for masked patches
        mask_expanded = mask.unsqueeze(-1).expand_as(orig_patches)

        merged_patches = pred * mask_expanded + orig_patches * (1 - mask_expanded)

        reconstructed = self.unpatchify(merged_patches)

        # Create masked input visualization
        patches = self.patchify(imgs)
        # Zero out masked patches
        mask_expanded = mask.unsqueeze(-1).expand_as(patches)
        masked_patches = patches * (1 - mask_expanded)
        masked = self.unpatchify(masked_patches)

        return imgs, reconstructed, masked, mask


class MAESmall(MAE):
    """
    Optimized MAE variant for GASF images.

    Architecture:
    - Encoder: 384 dim, 6 layers, 6 heads (for good downstream transfer)
    - Decoder: 192 dim, 4 layers, 6 heads (lightweight, just for reconstruction)
    - MLP dims are 4x embed dims (standard transformer ratio)
    """

    def __init__(
        self,
        img_size=256,
        patch_size=32,              # 8x8=64 patches
        in_channels=1,              # Grayscale GASF
        mask_ratio=0.75,            # Standard 75% masking
        dropout=0.1,                # Regularization
        norm_pix_loss=True          # Normalized pixel loss
    ):
        super().__init__(
            img_size=img_size,
            patch_size=patch_size,
            in_channels=in_channels,
            # Encoder config (this matters for downstream)
            encoder_embed_dim=384,
            encoder_num_layers=6,
            encoder_num_heads=6,
            encoder_mlp_dim=1536,       # 4x embed_dim
            # Decoder config (lightweight, just for reconstruction)
            decoder_embed_dim=192,
            decoder_num_layers=4,
            decoder_num_heads=6,
            decoder_mlp_dim=768,        # 4x embed_dim
            mask_ratio=mask_ratio,
            dropout=dropout,
            norm_pix_loss=norm_pix_loss
        )


class MAEBase(MAE):
    """Base MAE variant matching ViT-Base configuration."""

    def __init__(
        self,
        img_size=256,
        patch_size=32,
        in_channels=1,
        mask_ratio=0.75,
        dropout=0.0,
        norm_pix_loss=True
    ):
        super().__init__(
            img_size=img_size,
            patch_size=patch_size,
            in_channels=in_channels,
            encoder_embed_dim=768,
            encoder_num_layers=12,
            encoder_num_heads=12,
            encoder_mlp_dim=3072,
            decoder_embed_dim=512,
            decoder_num_layers=4,
            decoder_num_heads=8,
            decoder_mlp_dim=2048,
            mask_ratio=mask_ratio,
            dropout=dropout,
            norm_pix_loss=norm_pix_loss
        )


if __name__ == "__main__":
    # Test MAE
    print("Testing MAE...")

    # Create model
    model = MAEBase(
        img_size=256,
        patch_size=32,
        in_channels=1,
        mask_ratio=0.75
    )

    # Test input
    x = torch.randn(4, 1, 256, 256)

    # Forward pass
    loss, pred, mask = model(x)

    print(f"Input shape: {x.shape}")
    print(f"Prediction shape: {pred.shape}")
    print(f"Mask shape: {mask.shape}")
    print(f"Mask ratio (actual): {mask.sum() / mask.numel():.2%}")
    print(f"Loss: {loss.item():.4f}")

    # Test reconstruction
    orig, recon, masked, mask = model.reconstruct(x)
    print(f"\nReconstruction shapes:")
    print(f"  Original: {orig.shape}")
    print(f"  Reconstructed: {recon.shape}")
    print(f"  Masked input: {masked.shape}")

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    encoder_params = sum(p.numel() for p in model.encoder_blocks.parameters())
    encoder_params += model.patch_embed.projection.weight.numel()
    encoder_params += model.patch_embed.projection.bias.numel()
    encoder_params += model.cls_token.numel()
    encoder_params += model.encoder_pos_embed.numel()
    encoder_params += model.encoder_norm.weight.numel()
    encoder_params += model.encoder_norm.bias.numel()

    print(f"\nParameter counts:")
    print(f"  Total: {total_params:,}")
    print(f"  Encoder: {encoder_params:,}")
    print(f"  Decoder: {total_params - encoder_params:,}")
