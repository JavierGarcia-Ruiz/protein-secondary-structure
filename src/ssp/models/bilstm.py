"""Multi-scale CNN + Bidirectional LSTM architecture for protein secondary structure."""

from __future__ import annotations

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class CNNBiLSTMSecondaryStructureModel(nn.Module):
    """Hybrid CNN-BiLSTM architecture.

    1. Multi-scale 1D CNN captures local sequence motifs (H-bonding distance 3-7 residues).
    2. Bidirectional LSTM layers model global sequence dependencies.
    3. Multi-task linear heads predict 8-state (Q8) and 3-state (Q3) secondary structure.
    """

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
        self.d_in = d_in
        self.d_model = d_model
        
        # Multi-scale parallel convolutions
        self.convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(d_in, d_model // len(kernel_sizes), kernel_size=k, padding=k // 2),
                nn.LayerNorm(d_model // len(kernel_sizes)),
                nn.GELU(),
                nn.Dropout(dropout),
            )
            for k in kernel_sizes
        ])
        
        conv_out_dim = (d_model // len(kernel_sizes)) * len(kernel_sizes)
        self.proj_conv = nn.Linear(conv_out_dim, d_model)
        
        # Bidirectional LSTM
        self.lstm = nn.LSTM(
            input_size=d_model,
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            bidirectional=True,
            batch_first=True,
            dropout=dropout if lstm_layers > 1 else 0.0,
        )
        
        rep_dim = lstm_hidden * 2
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(rep_dim)
        
        # Multi-task heads
        self.q8_head = nn.Linear(rep_dim, q8_classes)
        self.q3_head = nn.Linear(rep_dim, q3_classes)

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor | None = None,
        lengths: torch.Tensor | None = None,
        **kwargs,
    ) -> dict[str, torch.Tensor]:
        # x: (B, L, d_in)
        B, L, _ = x.shape
        
        # 1. Multi-scale CNN
        x_conv = x.transpose(1, 2)  # (B, d_in, L)
        conv_outs = []
        for conv in self.convs:
            # apply conv: (B, C, L) -> transpose to (B, L, C) for LayerNorm inside sequential
            c = conv[0](x_conv).transpose(1, 2)
            c = conv[1](c)
            c = conv[2](c)
            c = conv[3](c)
            conv_outs.append(c)
            
        h_conv = torch.cat(conv_outs, dim=-1)  # (B, L, conv_out_dim)
        h_conv = self.proj_conv(h_conv)
        
        # 2. Packed BiLSTM
        if lengths is not None:
            lengths_cpu = lengths.cpu()
            packed = pack_padded_sequence(h_conv, lengths_cpu, batch_first=True, enforce_sorted=False)
            packed_out, _ = self.lstm(packed)
            h_lstm, _ = pad_packed_sequence(packed_out, batch_first=True, total_length=L)
        else:
            h_lstm, _ = self.lstm(h_conv)
            
        h_rep = self.norm(self.dropout(h_lstm))
        
        if mask is not None:
            h_rep = h_rep * mask.unsqueeze(-1)
            
        q8_logits = self.q8_head(h_rep)
        q3_logits = self.q3_head(h_rep)
        
        return {
            "q8_logits": q8_logits,
            "q3_logits": q3_logits,
            "embeddings": h_rep,
        }
