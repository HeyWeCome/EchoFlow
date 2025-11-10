from typing import Dict, Any

import torch
import torch.nn as nn
import torch.nn.functional as F


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float) * (-torch.log(torch.tensor(10000.0)) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        T = x.size(1)
        # Keep the same dtype as input to avoid dtype mismatch under AMP
        return x + self.pe[:T].unsqueeze(0).to(x.dtype)


class TransformerBackbone(nn.Module):
    """Reusable Transformer encoding backbone without optimizer or Lightning logic.

    Provides `forward_features(batch)` returning (B, T, D) for upper-level models.
    """

    def __init__(self, vocab_size: int, d_model: int = 256, n_heads: int = 4, n_layers: int = 2, dropout: float = 0.1, pad_idx: int = 0):
        super().__init__()
        self.pad_idx = pad_idx
        self.embed = nn.Embedding(vocab_size, d_model, padding_idx=self.pad_idx)
        self.pos = PositionalEncoding(d_model)
        self.time_mlp = nn.Sequential(
            nn.Linear(1, d_model), nn.ReLU(), nn.Linear(d_model, d_model)
        )
        enc_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=n_heads, dim_feedforward=4 * d_model, dropout=dropout, batch_first=True)
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=n_layers)
        self.norm = nn.LayerNorm(d_model)

    def forward_features(self, batch: Dict[str, Any]) -> torch.Tensor:
        input_ids: torch.Tensor = batch["input_ids"]
        key_padding_mask: torch.Tensor = batch["key_padding_mask"]
        x = self.embed(input_ids)
        if "timestamps" in batch:
            timestamps: torch.Tensor = batch["timestamps"].float()
            deltas = timestamps.clone()
            deltas[:, 1:] = timestamps[:, 1:] - timestamps[:, :-1]
            deltas[:, 0] = 0.0
            deltas = torch.log1p(torch.clamp(deltas, min=0.0))
            t_feat = self.time_mlp(deltas.unsqueeze(-1)).to(x.dtype)
            x = x + t_feat
        x = self.pos(x)
        T = input_ids.size(1)
        causal_mask = torch.triu(torch.ones(T, T, dtype=torch.bool, device=input_ids.device), diagonal=1)
        # Conditionally enable autocast: precision controlled by external Trainer (e.g., bf16)
        device_type = "cuda" if x.is_cuda else "cpu"
        with torch.autocast(device_type=device_type, enabled=torch.is_autocast_enabled()):
            x = self.encoder(x, mask=causal_mask, src_key_padding_mask=key_padding_mask)
        x = self.norm(x)
        return x
        