from typing import Dict, Any, List, Tuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl

from utils.metrics import compute_hit_map_at_k
from pl.layers.transformer import PositionalEncoding
from pl.models.dyhgcn import TimeAttention
from pl.layers.time_bins import (
    _build_time_bins,
    _map_ts_to_bin,
    _choose_time_step_split,
)
from pl.layers.bipartite import build_bipartite_adj_snapshots




class SimpleTransformerBlock(nn.Module):
    """A minimal self-attention block to emulate reference SIEVE's TransformerBlock.

    Uses PyTorch's TransformerEncoderLayer with batch_first.
    """

    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        self.layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=4 * d_model,
            dropout=dropout,
            batch_first=True,
        )
        self.enc = nn.TransformerEncoder(self.layer, num_layers=1)

    def forward(self, x: torch.Tensor, src_key_padding_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        return self.enc(x, mask=None, src_key_padding_mask=src_key_padding_mask)


class SIEVELM(pl.LightningModule):
    """
    Components:
    - Bipartite cascade-user graph built from training events only (no leakage)
    - GCN propagation over bipartite graph to produce joint embeddings
    - Sequence encoder with self-attention; alignment attention via a second block
    - Contrastive learning (CL) loss between clean and perturbed embeddings

    Training/eval interface matches existing models: forward(batch)->(B,T,D) features, then classification.

    If you want the original version submitted to AAAI'26, 
    please visit: https://github.com/HeyWeCome/SIEVE
    """

    def __init__(
        self,
        vocab_size: int,
        num_topics: int,
        events: List[Tuple[int, int, int]],
        d_model: int = 256,
        n_heads: int = 4,
        gcn_layers: int = 2,
        dropout: float = 0.1,
        time_step_split: int = 5,
        lr: float = 1e-3,
        weight_decay: float = 0.01,
        cl_rate: float = 0.1,
        eps: float = 0.1,
        temperature: float = 0.1,
        temperature_uncertainty: float = 0.1,
        beta: float = 0.1,
        disable_ungsl_update: bool = False,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.pad_idx = 0
        self.vocab_size = int(vocab_size)
        self.num_topics = int(num_topics)
        self.d_model = int(d_model)
        self.n_heads = int(n_heads)
        self.gcn_layers = int(gcn_layers)
        self.drop = nn.Dropout(dropout)

        # Embeddings for users and cascades (topics)
        self.user_embedding = nn.Embedding(self.vocab_size, self.d_model)
        self.cas_embedding = nn.Embedding(self.num_topics, self.d_model)
        nn.init.xavier_uniform_(self.user_embedding.weight)
        nn.init.xavier_uniform_(self.cas_embedding.weight)

        # --- Time-binned bipartite adjacency snapshots ---
        timestamps_all = [ts for _, _, ts in events]
        # Build bins (fixed count)
        self.time_step_split = int(time_step_split)
        self.time_bins: List[int] = _build_time_bins(timestamps_all, self.time_step_split)
        self.snap_adj: List[torch.Tensor] = build_bipartite_adj_snapshots(self.num_topics, self.vocab_size, events, self.time_bins)
        self._snap_adj_device: Optional[torch.device] = None
        self._snap_adj_cpu: List[torch.Tensor] = self.snap_adj  # base snapshots on CPU

        # Simple GCN: linear transforms + sparse matmul per layer
        self.W_gcn0 = nn.Linear(self.d_model, self.d_model)
        self.W_gcn1 = nn.Linear(self.d_model, self.d_model)

        # Sequence encoders (two blocks)
        self.time_att = TimeAttention(self.time_step_split, self.d_model)
        self.seq_encoder = SimpleTransformerBlock(self.d_model, self.n_heads, dropout=dropout)
        self.align_attention = SimpleTransformerBlock(self.d_model, self.n_heads, dropout=dropout)
        self.pos_enc = PositionalEncoding(self.d_model)

        # Head
        self.head = nn.Linear(self.d_model, self.vocab_size)
        self.criterion = nn.CrossEntropyLoss(ignore_index=self.pad_idx)
        self.topk_list = self.hparams.get("topk", [5, 10, 20])

        # CL hyperparams
        self.cl_rate = float(cl_rate)
        self.eps = float(eps)
        self.temperature = float(temperature)
        self.temperature_uncertainty = float(temperature_uncertainty)
        self.disable_ungsl_update = bool(disable_ungsl_update)
        self.beta = float(beta)
        # UNGSL thresholds per node (cascade + user)
        self.thresholds = nn.Parameter(torch.full((self.num_topics + self.vocab_size, 1), 0.5, dtype=torch.float32))

        # Optim
        self.lr = float(lr)
        self.weight_decay = float(weight_decay)

        # Cache for training to avoid recomputation of node embeddings for CL
        self._cache_node_emb_list: Optional[List[torch.Tensor]] = None

    # --- Graph utilities ---
    def _compute_node_embeddings_for_adj(self, A_hat: torch.Tensor) -> torch.Tensor:
        """Run sparse GCN on provided adjacency to produce node embeddings (cascade + user)."""
        try:
            autocast_disable = torch.cuda.amp.autocast(enabled=False)
        except Exception:
            from contextlib import nullcontext
            autocast_disable = nullcontext()
        with autocast_disable:
            cas = self.cas_embedding.weight.float()
            usr = self.user_embedding.weight.float()
            X0 = torch.cat([cas, usr], dim=0)  # (V, D)
            A = A_hat.float()
            H = X0
            for _ in range(max(1, self.gcn_layers)):
                Z = self.W_gcn0(H)
                H_msg = torch.sparse.mm(A, Z)
                H = F.relu(self.drop(H_msg))
            Z2 = self.W_gcn1(H)
            H2 = torch.sparse.mm(A, Z2)
        return H2

    def _cl_loss_raw(self, x1: torch.Tensor, x2: torch.Tensor, temperature: float, reduce: str = "mean") -> torch.Tensor:
        x1 = F.normalize(x1, dim=-1)
        x2 = F.normalize(x2, dim=-1)
        pos = (x1 * x2).sum(dim=-1)
        pos = torch.exp(pos / temperature)
        neg = torch.exp(x1 @ x2.T / temperature).sum(dim=-1)
        loss_vec = -torch.log(pos / (neg + 1e-6))
        if reduce == "mean":
            return loss_vec.mean()
        else:
            return loss_vec

    def _update_adj_with_ungsl(self, A_hat: torch.Tensor) -> torch.Tensor:
        """Apply UNGSL reweighting to adjacency based on confidence from CL on ego embeddings."""
        if not self.training or self.disable_ungsl_update:
            return A_hat
        try:
            autocast_disable = torch.cuda.amp.autocast(enabled=False)
        except Exception:
            from contextlib import nullcontext
            autocast_disable = nullcontext()
        with autocast_disable:
            E0 = torch.cat([self.cas_embedding.weight, self.user_embedding.weight], dim=0).float()  # (V,D)
            cas_E0, user_E0 = torch.split(E0, [self.num_topics, self.vocab_size], dim=0)
            cas_noise = self._add_noise(cas_E0)
            user_noise = self._add_noise(user_E0)
            cas_u1 = self._cl_loss_raw(cas_E0, cas_noise, self.temperature_uncertainty, reduce="none")
            cas_u2 = self._cl_loss_raw(cas_noise, cas_E0, self.temperature_uncertainty, reduce="none")
            cas_uncertainty = 0.5 * (cas_u1 + cas_u2)
            user_u1 = self._cl_loss_raw(user_E0, user_noise, self.temperature_uncertainty, reduce="none")
            user_u2 = self._cl_loss_raw(user_noise, user_E0, self.temperature_uncertainty, reduce="none")
            user_uncertainty = 0.5 * (user_u1 + user_u2)

            cas_conf = torch.exp(-cas_uncertainty).unsqueeze(-1)
            user_conf = torch.exp(-user_uncertainty).unsqueeze(-1)
            conf_vec = torch.cat([cas_conf, user_conf], dim=0)  # (V,1)

            A_hat = A_hat.coalesce()
            idx = A_hat.indices()
            val = A_hat.values()
            src = idx[0]
            dst = idx[1]
            conf_src = conf_vec.index_select(0, src).squeeze(-1)
            thr_dst = self.thresholds.index_select(0, dst).squeeze(-1)
            x = conf_src - thr_dst
            if x.numel() == 0:
                return A_hat
            uncertainty_vec = torch.cat([cas_uncertainty, user_uncertainty], dim=0)
            unc_median = torch.median(uncertainty_vec)
            gamma_0 = 2.0
            gamma = gamma_0 * torch.exp(-unc_median)
            reliable = (x >= 0)
            unreliable = ~reliable
            weight = torch.zeros_like(x)
            weight[reliable] = gamma * torch.sigmoid(x[reliable])
            weight[unreliable] = self.beta
            new_val = val * weight
            if torch.isnan(new_val).any():
                return A_hat  # skip update
            A_new = torch.sparse_coo_tensor(idx, new_val, A_hat.shape).coalesce()
            return A_new

    def _add_noise(self, E: torch.Tensor) -> torch.Tensor:
        noise = torch.randn_like(E)
        signed = noise * torch.sign(E)
        norm = F.normalize(signed, p=2, dim=-1, eps=self.eps)
        return E + self.eps * norm

    def _cl_loss(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        x1 = F.normalize(x1, dim=-1)
        x2 = F.normalize(x2, dim=-1)
        pos = (x1 * x2).sum(dim=-1)
        pos = torch.exp(pos / self.temperature)
        neg = torch.exp(x1 @ x2.T / self.temperature).sum(dim=-1)
        return -torch.log(pos / (neg + 1e-6)).mean()

    # --- Feature builder ---
    def forward(self, batch: Dict[str, Any]) -> torch.Tensor:
        input_ids: torch.Tensor = batch["input_ids"]  # (B, T)
        key_padding_mask: torch.Tensor = batch["key_padding_mask"]  # (B, T)
        timestamps: Optional[torch.Tensor] = batch.get("timestamps")
        B, T = input_ids.size()

        # Move snapshots to device and optionally apply UNGSL updates
        if self._snap_adj_device != input_ids.device:
            self._snap_adj_cpu = [A.coalesce().cpu() for A in self._snap_adj_cpu]
            self._snap_adj_device = input_ids.device
        A_list_dev: List[torch.Tensor] = []
        for A in self._snap_adj_cpu:
            A_dev = A.to(input_ids.device)
            if self.training and not self.disable_ungsl_update:
                A_dev = self._update_adj_with_ungsl(A_dev)
            A_list_dev.append(A_dev)

        # Compute node embeddings per snapshot
        device_type = "cuda" if input_ids.is_cuda else "cpu"
        node_emb_list: List[torch.Tensor] = []
        for A_dev in A_list_dev:
            H = self._compute_node_embeddings_for_adj(A_dev)
            node_emb_list.append(H)  # (V, D)
        # Cache for CL in training
        self._cache_node_emb_list = node_emb_list if self.training else None

        # Lookup user embeddings for each snapshot and stack to (B, T, K, D)
        user_base = self.num_topics
        dy_list: List[torch.Tensor] = []
        for H in node_emb_list:
            user_emb = H[user_base : user_base + self.vocab_size]
            dy = F.embedding(input_ids, user_emb)  # (B, T, D)
            dy_list.append(dy.unsqueeze(2))
        dyuser_emb = torch.cat(dy_list, dim=2)  # (B, T, K, D)

        # Map timestamps to bins and fuse via time attention
        if timestamps is not None:
            T_idx = _map_ts_to_bin(timestamps.float(), self.time_bins)
        else:
            T_idx = torch.full((B, T), len(self.time_bins) - 1, device=input_ids.device, dtype=torch.long)
        dyemb = self.time_att(T_idx, dyuser_emb)  # (B, T, D)
        dyemb = self.pos_enc(dyemb)

        # Causal transformer encoding
        causal_mask = torch.triu(torch.ones(T, T, dtype=torch.bool, device=input_ids.device), diagonal=1)
        h1 = self.seq_encoder.enc(
            dyemb,
            mask=causal_mask,
            src_key_padding_mask=key_padding_mask,
        )
        h2 = self.align_attention.enc(
            h1,
            mask=causal_mask,
            src_key_padding_mask=key_padding_mask,
        )
        return h1

    # --- Lightning steps ---
    def _step(self, batch: Dict[str, Any], stage: str):
        x = self.forward(batch)  # (B, T, D)
        input_ids: torch.Tensor = batch["input_ids"]
        lengths: torch.Tensor = batch["lengths"]

        if stage == "train" and "target" not in batch:
            # train on all valid next positions except last two events
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
                loss_main = F.cross_entropy(logits, P)
                acc = (logits.argmax(dim=-1) == P).float().mean()
            else:
                loss_main = h_flat.sum() * 0.0
                acc = torch.tensor(0.0, device=x.device)

            # CL loss: mean across snapshot node embeddings (clean) vs perturbed
            if self._cache_node_emb_list is not None and len(self._cache_node_emb_list) > 0:
                joint_clean = torch.stack(self._cache_node_emb_list, dim=0).mean(dim=0)
            else:
                # fallback to ego embeddings
                joint_clean = torch.cat([self.cas_embedding.weight, self.user_embedding.weight], dim=0)
            joint_pert = self._add_noise(joint_clean.detach())
            user_base = self.num_topics
            cas_clean = joint_clean[0 : self.num_topics]
            cas_pert = joint_pert[0 : self.num_topics]
            user_clean = joint_clean[user_base : user_base + self.vocab_size]
            user_pert = joint_pert[user_base : user_base + self.vocab_size]
            cl_loss_users = self._cl_loss_raw(user_clean, user_pert, self.temperature, reduce="mean")
            cl_loss_cascades = self._cl_loss_raw(cas_clean, cas_pert, self.temperature, reduce="mean")
            loss = loss_main + self.cl_rate * (cl_loss_users + cl_loss_cascades)

            self.log(f"{stage}_loss", loss, prog_bar=True, on_step=True, on_epoch=True)
            self.log(f"{stage}_acc", acc, prog_bar=True, on_step=False, on_epoch=True)
            return loss

        # eval: last hidden for classification
        target: torch.Tensor = batch["target"]
        B = x.size(0)
        last_idx = lengths - 1
        last_hidden = x[torch.arange(B, device=x.device), last_idx]
        logits = self.head(last_hidden)
        loss = self.criterion(logits, target)
        acc = (logits.argmax(dim=-1) == target).float().mean()

        # top-k metrics
        metrics = {}
        if isinstance(self.topk_list, (list, tuple)):
            m = compute_hit_map_at_k(logits, target, topk=list(self.topk_list))
            for name, val in m.items():
                metrics[f"{stage}_{name}"] = val

        self.log(f"{stage}_loss", loss, prog_bar=True, on_step=False, on_epoch=True)
        self.log(f"{stage}_acc", acc, prog_bar=True, on_step=False, on_epoch=True)
        for name, val in metrics.items():
            self.log(name, val, prog_bar=False, on_step=False, on_epoch=True)
        return loss

    def training_step(self, batch: Dict[str, Any], batch_idx: int):
        return self._step(batch, stage="train")

    def validation_step(self, batch: Dict[str, Any], batch_idx: int):
        self._step(batch, stage="val")

    def test_step(self, batch: Dict[str, Any], batch_idx: int):
        self._step(batch, stage="test")

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)


def build_model(
    vocab_size: int,
    model_cfg: Dict[str, Any],
    optim_cfg: Dict[str, Any],
    events: Optional[List[Tuple[int, int, int]]] = None,
    num_topics: Optional[int] = None,
) -> pl.LightningModule:
    """Builder for SIEVELM to match training script dynamic import.

    Requires training events (first L-2 per cascade) and number of topics.
    """
    d_model = int(model_cfg.get("d_model", 256))
    n_heads = int(model_cfg.get("n_heads", 4))
    gcn_layers = int(model_cfg.get("gcn_layers", 2))
    dropout = float(model_cfg.get("dropout", 0.1))
    time_step_cfg = model_cfg.get("time_step_split", 5)
    lr = float(optim_cfg.get("lr", 1e-3))
    weight_decay = float(optim_cfg.get("weight_decay", 0.01))
    cl_rate = float(model_cfg.get("cl_rate", 0.1))
    eps = float(model_cfg.get("eps", 0.1))
    temperature = float(model_cfg.get("temperature", 0.1))
    temperature_uncertainty = float(model_cfg.get("temperature_uncertainty", 0.1))
    beta = float(model_cfg.get("beta", 0.1))
    disable_ungsl_update = bool(model_cfg.get("disable_ungsl_update", False))

    if events is None or num_topics is None:
        # minimal viable fallback: empty graph with self-loops only
        events = []
        if num_topics is None:
            num_topics = 1
    # Determine time_step_split (support 'auto')
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

    m = SIEVELM(
        vocab_size=vocab_size,
        num_topics=int(num_topics),
        events=list(events),
        d_model=d_model,
        n_heads=n_heads,
        gcn_layers=gcn_layers,
        dropout=dropout,
        time_step_split=time_step_split,
        lr=lr,
        weight_decay=weight_decay,
        cl_rate=cl_rate,
        eps=eps,
        temperature=temperature,
        temperature_uncertainty=temperature_uncertainty,
        beta=beta,
        disable_ungsl_update=disable_ungsl_update,
    )
    m.hparams.topk = model_cfg.get("topk", [5, 10, 20])
    return m
