from typing import Dict, Any, List, Tuple, Optional

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl

from utils.metrics import compute_hit_map_at_k
from pl.layers.transformer import PositionalEncoding
from pl.layers.time_bins import _build_time_bins, _map_ts_to_bin, _choose_time_step_split


# Time bin utilities moved to pl.layers.time_bins


def _build_diffusion_edges_from_events(events: List[Tuple[int, int, int]]) -> List[Tuple[int, int, int]]:
    """From (u_idx, topic_idx, ts) events, reconstruct user-user diffusion edges
    within each topic cascade: (u_prev -> u_next, ts_next).
    """
    by_topic: Dict[int, List[Tuple[int, int]]] = {}
    for u, t, ts in events:
        by_topic.setdefault(t, []).append((u, ts))
    for t in by_topic:
        by_topic[t].sort(key=lambda x: x[1])
    edges: List[Tuple[int, int, int]] = []
    for t, seq in by_topic.items():
        for i in range(1, len(seq)):
            u_prev, _ = seq[i - 1]
            u_next, ts_next = seq[i]
            edges.append((u_prev, u_next, ts_next))
    return edges


def _percentile_int(ts_sorted: List[int], p: float) -> int:
    # deprecated local impl; kept for backward compatibility if referenced elsewhere
    L = len(ts_sorted)
    if L == 0:
        return 0
    if L == 1:
        return int(ts_sorted[0])
    pos = p * (L - 1)
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return int(ts_sorted[lo])
    frac = pos - lo
    return int(ts_sorted[lo] + frac * (ts_sorted[hi] - ts_sorted[lo]))


def _build_sparse_adj_snapshots(
    edges: List[Tuple[int, int, int]],
    vocab_size: int,
    time_bins: List[int],
) -> List[torch.Tensor]:
    """Build normalized adjacency A_hat for each time bin using only diffusion edges.

    - Symmetrize edges for GCN (add reverse edges).
    - Add self-loops with weight 1.
    - Normalize: D^{-1/2} (A) D^{-1/2}.
    Returns list of sparse COO tensors (float32) of shape (V, V).
    """
    # Group edges by bin (cumulative: ts <= bin)
    edges_sorted = sorted(edges, key=lambda x: x[2])
    snap_edges: List[List[Tuple[int, int, float]]] = []
    cur: List[Tuple[int, int, float]] = []
    j = 0
    for b in time_bins:
        while j < len(edges_sorted) and edges_sorted[j][2] <= b:
            u, v, _ = edges_sorted[j]
            # skip padding (0) if present
            if u == 0 or v == 0:
                j += 1
                continue
            cur.append((u, v, 1.0))
            cur.append((v, u, 1.0))  # symmetrize
            j += 1
        # Copy current cumulative edges
        snap_edges.append(list(cur))

    # Build sparse adjacency per snapshot
    snaps: List[torch.Tensor] = []
    V = int(vocab_size)
    for s_edges in snap_edges:
        # Add self-loops
        s_edges = s_edges + [(i, i, 1.0) for i in range(V)]
        if not s_edges:
            # fallback to identity
            i_idx = torch.arange(V, dtype=torch.long)
            idx = torch.stack([i_idx, i_idx], dim=0)
            val = torch.ones(V, dtype=torch.float32)
            A = torch.sparse_coo_tensor(idx, val, size=(V, V))
            snaps.append(A.coalesce())
            continue
        # Aggregate duplicate edges (sum weights)
        idx_rows = torch.tensor([e[0] for e in s_edges], dtype=torch.long)
        idx_cols = torch.tensor([e[1] for e in s_edges], dtype=torch.long)
        vals = torch.tensor([e[2] for e in s_edges], dtype=torch.float32)
        # coalesce by creating sparse then coalesce
        A = torch.sparse_coo_tensor(torch.stack([idx_rows, idx_cols], dim=0), vals, size=(V, V)).coalesce()
        # Degree
        deg = torch.sparse.sum(A, dim=1).to_dense()  # (V,)
        deg = torch.clamp(deg, min=1.0)
        inv_sqrt_deg = deg.pow(-0.5)
        # Normalize values: val * inv_sqrt_deg[row] * inv_sqrt_deg[col]
        rows = A.indices()[0]
        cols = A.indices()[1]
        norm_vals = A.values() * inv_sqrt_deg.index_select(0, rows) * inv_sqrt_deg.index_select(0, cols)
        A_hat = torch.sparse_coo_tensor(A.indices(), norm_vals, size=(V, V)).coalesce()
        snaps.append(A_hat)
    return snaps


class DiffusionGCN(nn.Module):
    """Two-layer GCN on sparse normalized adjacency for full user set.

    Produces node embeddings of shape (V, D).
    """

    def __init__(self, vocab_size: int, d_model: int, dropout: float = 0.1):
        super().__init__()
        self.vocab_size = int(vocab_size)
        self.d_model = int(d_model)
        self.embed = nn.Embedding(self.vocab_size, self.d_model)
        self.W0 = nn.Linear(self.d_model, self.d_model * 2)
        self.W1 = nn.Linear(self.d_model * 2, self.d_model)
        self.drop = nn.Dropout(dropout)
        nn.init.xavier_normal_(self.embed.weight)
        nn.init.xavier_normal_(self.W0.weight)
        nn.init.xavier_normal_(self.W1.weight)

    def forward(self, A_hat: torch.Tensor) -> torch.Tensor:
        # A_hat: sparse (V, V)
        # Sparse matmul does not support fp16 on CUDA; force float32 for this block.
        # This keeps AMP enabled for the rest of the model while ensuring correctness here.
        try:
            # Disable autocast for sparse ops; fall back silently on CPU
            autocast_disable = torch.cuda.amp.autocast(enabled=False)
        except Exception:
            # If amp not available, use a no-op context manager
            from contextlib import nullcontext
            autocast_disable = nullcontext()

        with autocast_disable:
            A_hat = A_hat.float()
            E = self.embed.weight.float()  # (V, D)
            H0 = E
            H1_lin = self.W0(H0)  # (V, 2D), float32
            H1_msg = torch.sparse.mm(A_hat, H1_lin)  # (V, 2D)
            H1 = F.relu(self.drop(H1_msg))
            H2_lin = self.W1(H1)  # (V, D)
            H2 = torch.sparse.mm(A_hat, H2_lin)  # (V, D)
        return H2  # (V, D)


class TimeAttention(nn.Module):
    def __init__(self, time_size: int, d_model: int):
        super().__init__()
        self.time_embedding = nn.Embedding(time_size, d_model)
        nn.init.xavier_normal_(self.time_embedding.weight)
        self.dropout = nn.Dropout(0.1)

    def forward(self, T_idx: torch.Tensor, Dy_U_embed: torch.Tensor) -> torch.Tensor:
        """Attention over time snapshots.

        - T_idx: (B, T) bin indices per position
        - Dy_U_embed: (B, T, K, D) user features from K snapshots
        Returns: (B, T, D)
        """
        temperature = Dy_U_embed.size(-1) ** 0.5
        T_embed = self.time_embedding(T_idx)  # (B, T, D)
        # Affine scores across snapshots K
        score = torch.einsum("btd,btkd->btk", T_embed, Dy_U_embed) / (temperature + 1e-6)
        # Causal mask: forbid attending to snapshots with index > T_idx
        K = Dy_U_embed.size(2)
        k_idx = torch.arange(K, device=T_idx.device).view(1, 1, K)
        causal_mask = k_idx > T_idx.unsqueeze(-1)  # (B, T, K)
        score = score.masked_fill(causal_mask, float("-inf"))
        alpha = F.softmax(score, dim=2)  # (B, T, K)
        alpha = self.dropout(alpha).unsqueeze(-1)  # (B, T, K, 1)
        att = (alpha * Dy_U_embed).sum(dim=2)  # (B, T, D)
        return att


class DyHGCN_SLM(pl.LightningModule):
    """DyHGCN_S variant integrated in our framework:
    - No social graph; diffusion-only snapshots per time bin
    - Within each snapshot, run two-layer sparse GCN to produce full-user embeddings
    - Fuse per-position embeddings across K snapshots via TimeAttention
    - Encode with TransformerEncoder (causal mask) and classify

    Chunyuan Yuan, Jiacheng Li, Wei Zhou, Yijun Lu, Xiaodan Zhang, Songlin Hu. 
    DyHGCN: A Dynamic Heterogeneous Graph Convolutional Network to Learn Users' Dynamic Preferences for Information Diffusion Prediction. In ECML-PKDD 2020.
    """

    def __init__(
        self,
        vocab_size: int,
        events: List[Tuple[int, int, int]],
        d_model: int = 256,
        n_heads: int = 4,
        n_layers: int = 2,
        dropout: float = 0.1,
        time_step_split: int = 5,
        lr: float = 1e-3,
        weight_decay: float = 0.01,
        topk: Optional[List[int]] = None,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.pad_idx = 0
        self.vocab_size = int(vocab_size)
        self.d_model = int(d_model)
        self.time_step_split = int(time_step_split)
        # Do not apply "previously seen user" masking; keep evaluation protocol simple and consistent

        # --- Diffusion snapshots ---
        timestamps_all = [ts for _, _, ts in events]
        self.time_bins: List[int] = _build_time_bins(timestamps_all, self.time_step_split)
        diff_edges = _build_diffusion_edges_from_events(events)
        self.snap_adj: List[torch.Tensor] = _build_sparse_adj_snapshots(diff_edges, self.vocab_size, self.time_bins)
        # Cache adjacency on device to avoid repeated large sparse copies each batch
        self._snap_adj_dev: Optional[List[torch.Tensor]] = None
        self._snap_adj_device: Optional[torch.device] = None

        # --- Snapshot GCN ---
        self.gcn = DiffusionGCN(self.vocab_size, self.d_model, dropout=dropout)

        # --- Fusion + Encoder ---
        self.time_att = TimeAttention(self.time_step_split, self.d_model)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=self.d_model,
            nhead=n_heads,
            dim_feedforward=4 * self.d_model,
            dropout=dropout,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=n_layers)
        self.pos = PositionalEncoding(self.d_model)
        self.norm = nn.LayerNorm(self.d_model)

        # --- Head and metrics ---
        self.head = nn.Linear(self.d_model, self.vocab_size)
        self.criterion = nn.CrossEntropyLoss(ignore_index=self.pad_idx)
        self.topk_list = list(topk or [5, 10, 20])

        # Optim
        self.lr = float(lr)
        self.weight_decay = float(weight_decay)

    def _build_features(self, batch: Dict[str, Any]) -> torch.Tensor:
        input_ids: torch.Tensor = batch["input_ids"]  # (B, T)
        key_padding_mask: torch.Tensor = batch["key_padding_mask"]  # (B, T)
        timestamps: Optional[torch.Tensor] = batch.get("timestamps")
        B, T = input_ids.size()

        # 1) Compute GCN embeddings for each snapshot (K = time_step_split)
        #    As weights update each forward, we recompute H2 per snapshot.
        device_type = "cuda" if input_ids.is_cuda else "cpu"
        # Prepare adjacency snapshots on current device (cached)
        if self._snap_adj_device != input_ids.device or self._snap_adj_dev is None:
            self._snap_adj_dev = [A.to(input_ids.device) for A in self.snap_adj]
            self._snap_adj_device = input_ids.device
        node_emb_list: List[torch.Tensor] = []
        for A_dev in self._snap_adj_dev:
            with torch.autocast(device_type=device_type, enabled=torch.is_autocast_enabled()):
                H = self.gcn(A_dev)  # (V, D)
            node_emb_list.append(H)

        # 2) For each snapshot, lookup embeddings for input_ids
        dyuser_list: List[torch.Tensor] = []
        for H in node_emb_list:
            dyuser = F.embedding(input_ids, H)  # (B, T, D)
            dyuser_list.append(dyuser.unsqueeze(2))  # (B, T, 1, D)
        dyuser_emb = torch.cat(dyuser_list, dim=2)  # (B, T, K, D)

        # 3) Map timestamps to bin indices per position
        if timestamps is not None:
            T_idx = _map_ts_to_bin(timestamps.float(), self.time_bins)
        else:
            # If no timestamps, use the last bin for all positions
            T_idx = torch.full((B, T), len(self.time_bins) - 1, device=input_ids.device, dtype=torch.long)

        # 4) Fuse across K snapshots via time attention
        dyemb = self.time_att(T_idx, dyuser_emb)  # (B, T, D)

        # 5) Positional encoding + causal transformer encoder
        x = self.pos(dyemb)
        causal_mask = torch.triu(torch.ones(T, T, dtype=torch.bool, device=input_ids.device), diagonal=1)
        with torch.autocast(device_type=device_type, enabled=torch.is_autocast_enabled()):
            x = self.encoder(x, mask=causal_mask, src_key_padding_mask=key_padding_mask)
        x = self.norm(x)
        return x  # (B, T, D)

    # --- Lightning hooks ---
    def forward(self, batch: Dict[str, Any]) -> torch.Tensor:
        return self._build_features(batch)

    def training_step(self, batch: Dict[str, Any], batch_idx: int):
        x = self.forward(batch)  # (B, T, D)
        input_ids: torch.Tensor = batch["input_ids"]
        lengths: torch.Tensor = batch["lengths"]
        # Predict next token at all valid positions except the last two events
        targets = input_ids[:, 1:]  # (B, T-1)
        Tm1 = targets.size(1)
        pos_idx = torch.arange(Tm1, device=x.device).unsqueeze(0)
        valid = pos_idx < (lengths.unsqueeze(1) - 3)

        h = x[:, :-1, :]  # (B, T-1, D)
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
        # Do not mask users that have already appeared; maintain unified evaluation logic
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
    if events is None:
        raise ValueError("DyHGCN requires events to build diffusion snapshots")
    d_model = int(model_cfg.get("d_model", 256))
    n_heads = int(model_cfg.get("n_heads", 4))
    n_layers = int(model_cfg.get("n_layers", 2))
    dropout = float(model_cfg.get("dropout", 0.1))
    time_step_cfg = model_cfg.get("time_step_split", 5)
    topk = model_cfg.get("topk", [5, 10, 20])
    lr = float(optim_cfg.get("lr", 1e-3))
    weight_decay = float(optim_cfg.get("weight_decay", 0.01))
    # Determine time_step_split automatically if requested
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
    return DyHGCN_SLM(
        vocab_size=vocab_size,
        events=list(events),
        d_model=d_model,
        n_heads=n_heads,
        n_layers=n_layers,
        dropout=dropout,
        time_step_split=time_step_split,
        lr=lr,
        weight_decay=weight_decay,
        topk=topk,
    )
    