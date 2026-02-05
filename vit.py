"""
Vision Transformer (ViT) Implementation in PyTorch

Based on "An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale"
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat


class PatchEmbedding(nn.Module):
    """Convert image into patches and embed them."""

    def __init__(self, img_size=256, patch_size=32, in_channels=1, embed_dim=768):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2
        self.patch_dim = patch_size * patch_size * in_channels

        # Linear projection of flattened patches
        self.projection = nn.Linear(self.patch_dim, embed_dim)

    def forward(self, x):
        """
        Args:
            x: (B, C, H, W) tensor
        Returns:
            (B, num_patches, embed_dim) tensor
        """
        B, C, H, W = x.shape

        # Reshape into patches: (B, C, H, W) -> (B, num_patches, patch_dim)
        x = rearrange(
            x,
            'b c (h p1) (w p2) -> b (h w) (p1 p2 c)',
            p1=self.patch_size,
            p2=self.patch_size
        )

        # Project to embedding dimension
        x = self.projection(x)
        return x


class MultiHeadAttention(nn.Module):
    """Multi-Head Self-Attention mechanism."""

    def __init__(self, embed_dim=768, num_heads=12, dropout=0.0):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.qkv = nn.Linear(embed_dim, embed_dim * 3, bias=True)
        self.proj = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        """
        Args:
            x: (B, N, D) tensor
        Returns:
            (B, N, D) tensor
        """
        B, N, D = x.shape

        # Compute Q, K, V
        qkv = self.qkv(x)
        qkv = rearrange(qkv, 'b n (three h d) -> three b h n d',
                        three=3, h=self.num_heads)
        q, k, v = qkv[0], qkv[1], qkv[2]

        # Scaled dot-product attention
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)

        # Apply attention to values
        x = attn @ v
        x = rearrange(x, 'b h n d -> b n (h d)')

        # Output projection
        x = self.proj(x)
        return x


class MLP(nn.Module):
    """MLP block with GELU activation."""

    def __init__(self, embed_dim=768, mlp_dim=3072, dropout=0.0):
        super().__init__()
        self.fc1 = nn.Linear(embed_dim, mlp_dim)
        self.fc2 = nn.Linear(mlp_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.fc1(x)
        x = F.gelu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x


class TransformerBlock(nn.Module):
    """Transformer encoder block."""

    def __init__(self, embed_dim=768, num_heads=12, mlp_dim=3072, dropout=0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = MultiHeadAttention(embed_dim, num_heads, dropout)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.mlp = MLP(embed_dim, mlp_dim, dropout)

    def forward(self, x):
        # Pre-norm architecture (as in original ViT)
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class ViTEncoder(nn.Module):
    """
    Vision Transformer Encoder.

    This is the encoder part that will be used for MAE pretraining.
    """

    def __init__(
        self,
        img_size=256,
        patch_size=32,
        in_channels=1,
        embed_dim=768,
        num_layers=12,
        num_heads=12,
        mlp_dim=3072,
        dropout=0.0
    ):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.num_patches = (img_size // patch_size) ** 2

        # Patch embedding
        self.patch_embed = PatchEmbedding(img_size, patch_size, in_channels, embed_dim)

        # Class token
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))

        # Position embeddings (learnable)
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches + 1, embed_dim))

        # Transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, mlp_dim, dropout)
            for _ in range(num_layers)
        ])

        # Final layer norm
        self.norm = nn.LayerNorm(embed_dim)

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        # Initialize position embeddings
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)

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

    def forward(self, x, return_all_tokens=False):
        """
        Args:
            x: (B, C, H, W) tensor
            return_all_tokens: If True, return all tokens including CLS
        Returns:
            If return_all_tokens: (B, N+1, D) tensor
            Else: (B, D) tensor (CLS token only)
        """
        B = x.shape[0]

        # Patch embedding
        x = self.patch_embed(x)  # (B, N, D)

        # Add class token
        cls_tokens = repeat(self.cls_token, '1 1 d -> b 1 d', b=B)
        x = torch.cat([cls_tokens, x], dim=1)  # (B, N+1, D)

        # Add position embeddings
        x = x + self.pos_embed

        # Transformer blocks
        for block in self.blocks:
            x = block(x)

        # Final normalization
        x = self.norm(x)

        if return_all_tokens:
            return x
        else:
            return x[:, 0]  # Return CLS token only


class ViTClassifier(nn.Module):
    """
    Complete Vision Transformer for classification.

    Used after pretraining for fine-tuning on downstream tasks.
    """

    def __init__(
        self,
        img_size=256,
        patch_size=32,
        in_channels=1,
        num_classes=2,
        embed_dim=768,
        num_layers=12,
        num_heads=12,
        mlp_dim=3072,
        dropout=0.0
    ):
        super().__init__()

        self.encoder = ViTEncoder(
            img_size=img_size,
            patch_size=patch_size,
            in_channels=in_channels,
            embed_dim=embed_dim,
            num_layers=num_layers,
            num_heads=num_heads,
            mlp_dim=mlp_dim,
            dropout=dropout
        )

        # Classification head
        self.classifier = nn.Linear(embed_dim, num_classes)

    def forward(self, x):
        """
        Args:
            x: (B, C, H, W) tensor
        Returns:
            (B, num_classes) logits
        """
        x = self.encoder(x)  # (B, D)
        x = self.classifier(x)  # (B, num_classes)
        return x

    def load_pretrained_encoder(self, pretrained_encoder_state_dict):
        """Load pretrained encoder weights from MAE."""
        self.encoder.load_state_dict(pretrained_encoder_state_dict)


if __name__ == "__main__":
    # Test the model
    config = {
        "img_size": 256,
        "patch_size": 32,
        "in_channels": 1,
        "num_classes": 2,
        "embed_dim": 768,
        "num_layers": 12,
        "num_heads": 12,
        "mlp_dim": 3072,
        "dropout": 0.1
    }

    model = ViTClassifier(**config)

    # Test forward pass
    x = torch.randn(2, 1, 256, 256)
    out = model(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {out.shape}")

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
