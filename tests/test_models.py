"""Unit tests for secondary structure neural network models."""

import pytest
import torch

from ssp.models.cnn import CNNSecondaryStructureModel
from ssp.models.bilstm import CNNBiLSTMSecondaryStructureModel
from ssp.models.crf import LinearChainCRF
from ssp.models.bilstm_crf import CNNBiLSTMCRFSecondaryStructureModel


def test_dilated_cnn_baseline_forward() -> None:
    B, L, D = 4, 30, 50
    x = torch.randn(B, L, D)
    mask = torch.ones(B, L, dtype=torch.bool)
    mask[0, 25:] = False  # padding

    model = CNNSecondaryStructureModel(d_in=D, d_model=64, n_layers=4, dropout=0.0)
    out = model(x, mask=mask)

    assert "q8_logits" in out
    assert "q3_logits" in out
    assert "embeddings" in out
    assert out["q8_logits"].shape == (B, L, 8)
    assert out["q3_logits"].shape == (B, L, 3)
    assert out["embeddings"].shape == (B, L, 64)


def test_cnn_bilstm_forward() -> None:
    B, L, D = 2, 20, 50
    x = torch.randn(B, L, D)
    mask = torch.ones(B, L, dtype=torch.bool)
    lengths = torch.tensor([20, 15])
    mask[1, 15:] = False

    model = CNNBiLSTMSecondaryStructureModel(d_in=D, d_model=64, lstm_hidden=32, lstm_layers=2, dropout=0.0)
    out = model(x, mask=mask, lengths=lengths)

    assert out["q8_logits"].shape == (B, L, 8)
    assert out["q3_logits"].shape == (B, L, 3)
    assert out["embeddings"].shape == (B, L, 64)  # 2 * lstm_hidden


def test_linear_chain_crf() -> None:
    B, L, C = 2, 5, 8
    emissions = torch.randn(B, L, C)
    tags = torch.randint(0, C, (B, L))
    mask = torch.ones(B, L, dtype=torch.bool)
    mask[1, 3:] = False

    crf = LinearChainCRF(num_tags=C)
    loss = crf(emissions, tags, mask=mask)
    assert loss.dim() == 0
    assert not torch.isnan(loss)

    paths = crf.decode(emissions, mask=mask)
    assert len(paths) == B
    assert len(paths[0]) == 5
    assert len(paths[1]) == 3


def test_bilstm_crf_model() -> None:
    B, L, D = 2, 10, 50
    x = torch.randn(B, L, D)
    q8 = torch.randint(0, 8, (B, L))
    mask = torch.ones(B, L, dtype=torch.bool)
    lengths = torch.tensor([10, 8])
    mask[1, 8:] = False

    model = CNNBiLSTMCRFSecondaryStructureModel(d_in=D, d_model=64, lstm_hidden=32, lstm_layers=1, dropout=0.0)
    out = model(x, mask=mask, lengths=lengths, tags=q8)

    assert "crf_loss" in out
    assert out["crf_loss"] is not None
    assert out["q8_logits"].shape == (B, L, 8)
    assert out["q3_logits"].shape == (B, L, 3)

    decoded = model.decode_q8(out["q8_logits"], mask)
    assert len(decoded) == B
    assert len(decoded[0]) == 10
    assert len(decoded[1]) == 8
