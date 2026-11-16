"""Multi-scale 1D Convolutional Neural Network baseline for per-residue prediction."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    """Residual Conv1d block with LayerNorm and GELU."""

    def __init__(self, channels: int, kernel_size: int, dilation: int = 1, dropout: float = 0.2):
        super().__init__()
        padding = (kernel_size - 1) * dilation // 2
        self.conv = nn.Conv1d(
            channels,
            channels,
            kernel_size=kernel_size,
            padding=padding,
            dilation=dilation,
        )
        self.norm = nn.LayerNorm(channels)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (B, L, C)
        res = x
        out = x.transpose(1, 2)  # (B, C, L)
        out = self.conv(out).transpose(1, 2)  # (B, L, C)
        out = F.gelu(out)
        out = self.norm(out + res)
        return self.dropout(out)


class CNNSecondaryStructureModel(nn.Module):
    """Multi-scale dilated CNN for secondary structure prediction."""

    def __init__(
        self,
        d_in: int = 50,
        d_model: int = 256,
        n_layers: int = 4,
        dropout: float = 0.2,
        q8_classes: int = 8,
        q3_classes: int = 3,
    ):
        super().__init__()
        self.in_proj = nn.Sequential(
            nn.Linear(d_in, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        
        # Dilated residual blocks with varying kernel sizes (3, 5, 7)
        self.blocks = nn.ModuleList([
            ConvBlock(
                d_model,
                kernel_size=3 if i % 2 == 0 else 5,
                dilation=2 ** (i % 3),
                dropout=dropout,
            )
            for i in range(n_layers)
        ])
        
        self.q8_head = nn.Linear(d_model, q8_classes)
        self.q3_head = nn.Linear(d_model, q3_classes)

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor | None = None,
        lengths: torch.Tensor | None = None,
        **kwargs,
    ) -> dict[str, torch.Tensor]:
        # x: (B, L, d_in)
        h = self.in_proj(x)
        for block in self.blocks:
            h = block(h)
            if mask is not None:
                h = h * mask.unsqueeze(-1)
                
        q8_logits = self.q8_head(h)
        q3_logits = self.q3_head(h)
        
        return {
            "q8_logits": q8_logits,
            "q3_logits": q3_logits,
            "embeddings": h,
        }
