"""Hybrid CNN-BiLSTM with Linear-Chain Conditional Random Field (CRF) decoding."""

from __future__ import annotations

import torch
import torch.nn as nn
from ssp.models.bilstm import CNNBiLSTMSecondaryStructureModel
from ssp.models.crf import LinearChainCRF


class CNNBiLSTMCRFSecondaryStructureModel(nn.Module):
    """CNN-BiLSTM encoder coupled with a Linear-Chain CRF for structured Q8 prediction."""

    def __init__(
        self,
        d_in: int = 50,
        d_model: int = 256,
        kernel_sizes: tuple[int, ...] = (3, 5, 7),
        lstm_layers: int = 2,
        lstm_hidden: int = 256,
        dropout: float = 0.3,
        q8_classes: int = 8,
        q3_classes: int = 3,
    ):
        super().__init__()
        self.encoder = CNNBiLSTMSecondaryStructureModel(
            d_in=d_in,
            d_model=d_model,
            kernel_sizes=kernel_sizes,
            lstm_layers=lstm_layers,
            lstm_hidden=lstm_hidden,
            dropout=dropout,
            q8_classes=q8_classes,
            q3_classes=q3_classes,
        )
        self.crf = LinearChainCRF(num_tags=q8_classes)

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor | None = None,
        lengths: torch.Tensor | None = None,
        tags: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        enc_out = self.encoder(x, mask=mask, lengths=lengths)
        q8_emissions = enc_out["q8_logits"]
        
        loss = None
        if tags is not None and mask is not None:
            # Mask out ignore_index labels for CRF loss computation
            valid_mask = mask & (tags >= 0)
            clean_tags = torch.clamp(tags, min=0)
            loss = self.crf(q8_emissions, clean_tags, valid_mask)
            
        return {
            "q8_logits": q8_emissions,
            "q3_logits": enc_out["q3_logits"],
            "embeddings": enc_out["embeddings"],
            "crf_loss": loss,
        }

    def decode_q8(self, emissions: torch.Tensor, mask: torch.Tensor) -> list[list[int]]:
        return self.crf.decode(emissions, mask)

