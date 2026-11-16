"""Model factory for secondary structure architectures."""

from __future__ import annotations

from typing import Any
import torch.nn as nn
from ssp.models.cnn import CNNSecondaryStructureModel
from ssp.models.bilstm import CNNBiLSTMSecondaryStructureModel
from ssp.models.bilstm_crf import CNNBiLSTMCRFSecondaryStructureModel


def build_model(cfg: dict[str, Any]) -> nn.Module:
    """Build model based on configuration dictionary."""
    model_cfg = cfg.get("model", {})
    name = model_cfg.get("name", "cnn_bilstm")
    d_in = model_cfg.get("d_in", 50)
    d_model = model_cfg.get("d_model", 256)
    dropout = model_cfg.get("dropout", 0.3)
    q8_classes = model_cfg.get("q8_classes", 8)
    q3_classes = model_cfg.get("q3_classes", 3)

    if name == "cnn":
        return CNNSecondaryStructureModel(
            d_in=d_in,
            d_model=d_model,
            n_layers=model_cfg.get("n_cnn_layers", 4),
            dropout=dropout,
            q8_classes=q8_classes,
            q3_classes=q3_classes,
        )
    elif name == "cnn_bilstm":
        return CNNBiLSTMSecondaryStructureModel(
            d_in=d_in,
            d_model=d_model,
            kernel_sizes=tuple(model_cfg.get("kernel_sizes", [3, 5, 7])),
            lstm_layers=model_cfg.get("lstm_layers", 2),
            lstm_hidden=model_cfg.get("lstm_hidden", 256),
            dropout=dropout,
            q8_classes=q8_classes,
            q3_classes=q3_classes,
        )
    elif name == "bilstm_crf":
        return CNNBiLSTMCRFSecondaryStructureModel(
            d_in=d_in,
            d_model=d_model,
            kernel_sizes=tuple(model_cfg.get("kernel_sizes", [3, 5, 7])),
            lstm_layers=model_cfg.get("lstm_layers", 2),
            lstm_hidden=model_cfg.get("lstm_hidden", 256),
            dropout=dropout,
            q8_classes=q8_classes,
            q3_classes=q3_classes,
        )
    else:
        raise ValueError(f"Unknown model architecture: {name}")

