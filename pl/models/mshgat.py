from typing import Dict, Any, List, Tuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl

from utils.metrics import compute_hit_map_at_k
# Reuse DyHGCN's time snapshot and diffusion-edge builders
from pl.models.dyhgcn import (
    DiffusionGCN,
    TimeAttention,
    _build_diffusion_edges_from_events,
    _build_sparse_adj_snapshots,
)
from pl.layers.time_bins import _build_time_bins, _map_ts_to_bin, _choose_time_step_split


class MSHGAT_B(pl.LightningModule):
    """MS-HGAT (without social graph): diffusion branch + dual decoder blocks.

    Modifications:
    - Completely remove the social graph (friendship GCN); keep only diffusion hypergraph time snapshots and their GCN;
    - Use DyHGCN's time binning and sparse GCN to produce full-user embeddings for each snapshot;
    - Fuse snapshot features via time attention in a causal manner to obtain (B, T, D);
    - Option B: add learnable positional embeddings (pos_dim), concatenate features along the last dimension, then pass through two Transformer decoder blocks (causal mask) to obtain (B, T, D+pos_dim).

    L. Sun, Y. Rao, X. Zhang, Y. Lan, and S. Yu, "MS-HGAT: Memory-Enhanced Sequential Hypergraph Attention Network for Information Diffusion Prediction", AAAI, vol. 36, no. 4, pp. 4156-4164, Jun. 2022.
    """

    def __init__(
        self,
        vocab_size: int,
        events: List[Tuple[int, int, int]],
        d_model: int = 256,
        n_heads: int = 4,
        dropout: float = 0.1,
        time_step_split: int = 5,
        pos_dim: int = 8,
        lr: float = 1e-3,
        weight_decay: float = 0.01,
        topk: Optional[List[int]] = None,
        max_pos_len: int = 1000,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.pad_idx = 0
        self.vocab_size = int(vocab_size)
        self.d_model = int(d_model)
        self.time_step_split = int(time_step_split)
        self.pos_dim = int(pos_dim)
        # Option B does not use "masking previously seen users" during evaluation
        # Lightning optimizer reads lr/weight_decay from module attributes
        self.lr = float(lr)
        self.weight_decay = float(weight_decay)

        # --- Diffusion time snapshots ---
        timestamps_all = [ts for _, _, ts in events]
        self.time_bins: List[int] = _build_time_bins(timestamps_all, self.time_step_split)
        diff_edges = _build_diffusion_edges_from_events(events)
        self.snap_adj: List[torch.Tensor] = _build_sparse_adj_snapshots(diff_edges, self.vocab_size, self.time_bins)

        # --- Snapshot GCN ---
        self.gcn = DiffusionGCN(self.vocab_size, self.d_model, dropout=dropout)

        # --- Time attention (fuse K snapshots) ---
        self.time_att = TimeAttention(self.time_step_split, self.d_model)

        # --- Option B: positional embedding + dual decoder blocks (causal mask) ---
        self.pos_embedding = nn.Embedding(max_pos_len, self.pos_dim)
        nn.init.xavier_normal_(self.pos_embedding.weight)

        d_total = self.d_model + self.pos_dim
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_total,
            nhead=n_heads,
            dim_feedforward=4 * d_total,
            dropout=dropout,
            batch_first=True,
        )
        # Two decoder blocks (two layers)
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=2)
        self.norm = nn.LayerNorm(d_total)

        # --- Classification head ---
        self.head = nn.Linear(d_total, self.vocab_size)
        self.criterion = nn.CrossEntropyLoss()

        # Evaluation metric configuration
        self.topk_list = tuple(topk) if isinstance(topk, (list, tuple)) else (5, 10, 20)

    # --- Helper: masking users activated in history during evaluation ---
    # Do not use prev_user masking; keep a simple, clear evaluation interface

    # --- Option B forward: build temporal features (B, T, D+pos_dim) ---
    def forward(self, batch: Dict[str, Any]) -> torch.Tensor:
        input_ids: torch.Tensor = batch["input_ids"]  # (B, T)
        key_padding_mask: torch.Tensor = batch["key_padding_mask"]  # (B, T)
        timestamps: Optional[torch.Tensor] = batch.get("timestamps")
        B, T = input_ids.size()

        # 1) Compute GCN full-user embeddings per snapshot
        device_type = "cuda" if input_ids.is_cuda else "cpu"
        node_emb_list: List[torch.Tensor] = []
        for A_hat in self.snap_adj:
            A_dev = A_hat.to(input_ids.device)
            with torch.autocast(device_type=device_type, enabled=torch.is_autocast_enabled()):
                H = self.gcn(A_dev)  # (V, D)
            node_emb_list.append(H)

        # 2) Index embeddings at each snapshot to obtain (B, T, K, D)
        dy_list: List[torch.Tensor] = []
        for H in node_emb_list:
            dy = F.embedding(input_ids, H)  # (B, T, D)
            dy_list.append(dy.unsqueeze(2))
        dyuser_emb = torch.cat(dy_list, dim=2)  # (B, T, K, D)

        # 3) Map timestamps to bin indices and fuse causally
        if timestamps is not None:
            T_idx = _map_ts_to_bin(timestamps.float(), self.time_bins)
        else:
            T_idx = torch.full((B, T), len(self.time_bins) - 1, device=input_ids.device, dtype=torch.long)
        dyemb = self.time_att(T_idx, dyuser_emb)  # (B, T, D)

        # 4) Option B: concatenate positional embeddings and pass through dual decoder blocks (causal + padding masks)
        pos_idx = torch.arange(T, device=input_ids.device).unsqueeze(0).expand(B, T)
        pos_feat = self.pos_embedding(pos_idx)  # (B, T, pos_dim)
        x = torch.cat([dyemb, pos_feat], dim=-1)  # (B, T, D+pos)

        causal_mask = torch.triu(torch.ones(T, T, dtype=torch.bool, device=input_ids.device), diagonal=1)
        with torch.autocast(device_type=device_type, enabled=torch.is_autocast_enabled()):
            x = self.encoder(x, mask=causal_mask, src_key_padding_mask=key_padding_mask)
        x = self.norm(x)
        return x

    # --- Training and evaluation ---
    def training_step(self, batch: Dict[str, Any], batch_idx: int):
        x = self.forward(batch)  # (B, T, D+pos)
        input_ids: torch.Tensor = batch["input_ids"]
        lengths: torch.Tensor = batch["lengths"]
        targets = input_ids[:, 1:]
        Tm1 = targets.size(1)
        pos_idx = torch.arange(Tm1, device=x.device).unsqueeze(0)
        valid = pos_idx < (lengths.unsqueeze(1) - 3)

        h = x[:, :-1, :]
        h_flat = h.reshape(-1, h.size(-1))
        targets_flat = targets.reshape(-1)
        mask_flat = valid.reshape(-1)

        if mask_flat.any():
            H = h_flat[mask_flat]
            P = targets_flat[mask_flat]
            logits = self.head(H)
            loss = F.cross_entropy(logits, P)
            acc = (logits.argmax(dim=-1) == P).float().mean()
        else:
            loss = h_flat.sum() * 0.0
            acc = torch.tensor(0.0, device=x.device)

        self.log("train_loss", loss, prog_bar=True, on_step=True, on_epoch=True)
        self.log("train_acc", acc, prog_bar=True, on_step=False, on_epoch=True)
        return loss

    def _eval_step(self, batch: Dict[str, Any], stage: str):
        x = self.forward(batch)
        target: torch.Tensor = batch["target"]
        lengths: torch.Tensor = batch["lengths"]
        B = x.size(0)
        last_idx = lengths - 1
        H_last = x[torch.arange(B, device=x.device), last_idx]
        logits = self.head(H_last)
        # Do not additionally mask previously-seen users; keep evaluation consistent with other models
        loss = self.criterion(logits, target)
        preds = logits.argmax(dim=-1)
        acc = (preds == target).float().mean()

        metrics = {}
        if isinstance(self.topk_list, (list, tuple)):
            m = compute_hit_map_at_k(logits, target, list(self.topk_list))
            for k in self.topk_list:
                metrics[f"{stage}_hit@{k}"] = m[f"hit@{k}"]
                metrics[f"{stage}_map@{k}"] = m[f"map@{k}"]

        self.log(f"{stage}_loss", loss, prog_bar=True, on_step=False, on_epoch=True)
        self.log(f"{stage}_acc", acc, prog_bar=True, on_step=False, on_epoch=True)
        for name, val in metrics.items():
            self.log(name, val, prog_bar=False, on_step=False, on_epoch=True)
        return loss

    def validation_step(self, batch: Dict[str, Any], batch_idx: int):
        return self._eval_step(batch, "val")

    def test_step(self, batch: Dict[str, Any], batch_idx: int):
        return self._eval_step(batch, "test")

    def configure_optimizers(self):
        opt = torch.optim.AdamW(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        return opt


def build_model(
    vocab_size: int,
    model_cfg: Dict[str, Any],
    optim_cfg: Dict[str, Any],
    events: Optional[List[Tuple[int, int, int]]] = None,
):
    """Unified entry point for dynamic loading by training scripts.

    If events are not provided (e.g., for profiling scripts), fall back to a single snapshot with self-loop adjacency; forward works but training is not recommended.
    """
    d_model = int(model_cfg.get("d_model", 256))
    n_heads = int(model_cfg.get("n_heads", 4))
    dropout = float(model_cfg.get("dropout", 0.1))
    time_step_cfg = model_cfg.get("time_step_split", 5)
    pos_dim = int(model_cfg.get("pos_dim", 8))
    topk = model_cfg.get("topk", [5, 10, 20])
    lr = float(optim_cfg.get("lr", 1e-3))
    weight_decay = float(optim_cfg.get("weight_decay", 0.01))

    # Automatically choose the number of time bins (same as DyHGCN)
    timestamps_all = [ts for _, _, ts in (events or [])]
    if isinstance(time_step_cfg, str) and time_step_cfg.strip().lower() == "auto":
        auto_min = int(model_cfg.get("auto_min_bins", 3))
        auto_max = int(model_cfg.get("auto_max_bins", 20))
        time_step_split = _choose_time_step_split(timestamps_all, min_bins=auto_min, max_bins=auto_max)
    else:
        try:
            time_step_split = int(time_step_cfg)
            if time_step_split <= 0:
                raise ValueError
        except Exception:
            time_step_split = 5

    # If events are not provided, build a placeholder model with empty events (for profiling scripts)
    safe_events: List[Tuple[int, int, int]]
    if events is None:
        safe_events = []
    else:
        safe_events = list(events)

    return MSHGAT_B(
        vocab_size=vocab_size,
        events=safe_events,
        d_model=d_model,
        n_heads=n_heads,
        dropout=dropout,
        time_step_split=time_step_split,
        pos_dim=pos_dim,
        lr=lr,
        weight_decay=weight_decay,
        topk=topk,
    )
