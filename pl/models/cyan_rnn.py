from typing import Dict, Any

import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl

from utils.metrics import compute_hit_map_at_k


class CYANRNNLM(pl.LightningModule):
    """CYAN-RNN: Cascade next-user prediction with RNN + coverage attention.

    Composition:
    - Base RNN (GRU by default) encodes user-embedding sequences to capture temporal patterns;
    - Attention mechanism (look-back over hidden states) computes weights over {1..t} when predicting t+1;
    - Coverage strategy: maintain coverage vector cov_t (accumulated past attention) and add a penalty term in attention scoring to encourage under-attended nodes.

    Interface consistent with training scripts: forward returns temporal features (B, T, D);
    Training uses candidate-set negative sampling; validation/test classify at the last step and report hit/MAP metrics.

    Reference:
    - Wang, Yongqing et al. (2017). "Cascade Dynamics Modeling with Attention-based Recurrent Neural Network." IJCAI, pp. 2985–2991.
    """

    def __init__(
        self,
        vocab_size: int,
        d_model: int = 256,
        n_layers: int = 1,
        dropout: float = 0.1,
        lr: float = 1e-3,
        weight_decay: float = 0.01,
        rnn_type: str = "gru",
    ):
        super().__init__()
        self.save_hyperparameters()

        self.pad_idx = 0
        # User embeddings
        self.embed = nn.Embedding(vocab_size, d_model, padding_idx=self.pad_idx)
        # Simple time-delta features (consistent with TransformerBackbone style)
        self.time_mlp = nn.Sequential(
            nn.Linear(1, d_model), nn.ReLU(), nn.Linear(d_model, d_model)
        )

        # RNN encoder
        self.hidden_dim = d_model
        self.n_layers = int(n_layers)
        self.rnn_type = str(rnn_type).lower()
        if self.rnn_type == "lstm":
            self.rnn = nn.LSTM(d_model, d_model, num_layers=self.n_layers, batch_first=True)
        else:
            self.rnn = nn.GRU(d_model, d_model, num_layers=self.n_layers, batch_first=True)

        # Coverage attention parameters: e_{t,i} = v^T tanh(Wa h_t + Ua h_i + Ca cov_{t,i})
        self.Wa = nn.Linear(d_model, d_model, bias=False)
        self.Ua = nn.Linear(d_model, d_model, bias=False)
        self.Ca = nn.Linear(1, d_model, bias=False)  # Map coverage scalar to feature dimension
        self.v = nn.Linear(d_model, 1, bias=False)

        # Fuse h_t with attention context c_t
        self.W_tilde = nn.Linear(2 * d_model, d_model)
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

        # Classification head
        self.head = nn.Linear(d_model, vocab_size)
        self.criterion = nn.CrossEntropyLoss(ignore_index=self.pad_idx)
        self.topk_list = self.hparams.get("topk", [5, 10, 20])
        self.neg_k = int(self.hparams.get("neg_k", 100))

        # Pure sequence task: no external cache or topic embeddings

    # --- Forward: return fused temporal features (B, T, D) ---
    def forward(self, batch: Dict[str, Any]) -> torch.Tensor:
        input_ids: torch.Tensor = batch["input_ids"]  # (B, T)
        key_padding_mask: torch.Tensor = batch["key_padding_mask"]  # (B, T)
        lengths: torch.Tensor = batch["lengths"]  # (B,)

        x = self.embed(input_ids)  # (B, T, D)
        # Time-delta features (log1p(delta_t)) to enhance temporal signal
        if "timestamps" in batch:
            timestamps: torch.Tensor = batch["timestamps"].float()
            deltas = timestamps.clone()
            deltas[:, 1:] = timestamps[:, 1:] - timestamps[:, :-1]
            deltas[:, 0] = 0.0
            deltas = torch.log1p(torch.clamp(deltas, min=0.0))
            t_feat = self.time_mlp(deltas.unsqueeze(-1))
            x = x + t_feat

        # RNN encoding (use pack to prevent PAD from affecting hidden state)
        # enforce_sorted=False avoids manual sorting
        packed = nn.utils.rnn.pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=False)
        out_packed, _ = self.rnn(packed)
        H, _ = nn.utils.rnn.pad_packed_sequence(out_packed, batch_first=True, total_length=input_ids.size(1))  # (B, T, D)

        B, T, D = H.size()
        # Coverage vector: accumulate past attention (position-wise)
        cov = torch.zeros(B, T, device=H.device, dtype=H.dtype)
        H_tilde = torch.zeros_like(H)

        # Stepwise attention (with coverage term), fuse to obtain h~_t
        UaH = self.Ua(H)  # Precompute Ua h_i, shape (B, T, D)
        for t in range(T):
            h_t = H[:, t, :]  # (B, D)
            # Attend only to valid positions: i <= t and i < lengths
            pos_idx = torch.arange(T, device=H.device).unsqueeze(0).expand(B, T)
            valid_i = (pos_idx <= t) & (pos_idx < lengths.unsqueeze(1))

            # Coverage term: Ca * cov_{:, i}
            cov_i = cov  # (B, T)
            cov_proj = self.Ca(cov_i.unsqueeze(-1))  # (B, T, D)

            # Attention score e_{t,i}
            Waht = self.Wa(h_t).unsqueeze(1)  # (B, 1, D)
            energy = self.v(torch.tanh(Waht + UaH + cov_proj)).squeeze(-1)  # (B, T)
            energy = energy.masked_fill(~valid_i, float('-inf'))

            # Normalize to obtain α_t
            alpha_t = torch.softmax(energy, dim=-1)  # (B, T)
            alpha_t = alpha_t.masked_fill(~valid_i, 0.0)

            # Context vector c_t
            c_t = torch.einsum("bt, btd -> bd", alpha_t, H)  # (B, D)

            # Fuse h_t and c_t
            hc = torch.cat([h_t, c_t], dim=-1)
            h_tilde = torch.tanh(self.W_tilde(hc))
            h_tilde = self.dropout(self.norm(h_tilde))
            H_tilde[:, t, :] = h_tilde

            # Update coverage: cov_t = sum_{j=1..t} α_j (used in next step)
            cov = cov + alpha_t

        return H_tilde  # (B, T, D)

    # --- Unified train/val/test step ---
    def _step(self, batch: Dict[str, Any], stage: str):
        x = self.forward(batch)  # (B, T, D)
        lengths: torch.Tensor = batch["lengths"]

        if stage == "train" and "target" not in batch:
            # Full-vocabulary softmax training
            targets = batch["input_ids"][:, 1:]
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

            self.log(f"{stage}_loss", loss, prog_bar=True, on_step=True, on_epoch=True)
            self.log(f"{stage}_acc", acc, prog_bar=True, on_step=False, on_epoch=True)
            return loss

        target: torch.Tensor = batch.get("target")
        B = x.size(0)
        last_idx = lengths - 1
        last_hidden = x[torch.arange(B, device=x.device), last_idx]

        if stage == "train" and target is not None:
            vocab_size = self.head.out_features
            neg_k = int(self.neg_k)
            device = last_hidden.device
            candidates = torch.empty(B, 1 + neg_k, dtype=torch.long, device=device)
            candidates[:, 0] = target

            for i in range(B):
                cnt = 0
                while cnt < neg_k:
                    samp = torch.randint(1, vocab_size, (neg_k,), device=device)
                    samp = samp[~samp.eq(target[i])]
                    take = min(neg_k - cnt, samp.numel())
                    if take > 0:
                        candidates[i, 1 + cnt : 1 + cnt + take] = samp[:take]
                        cnt += take

            weight = self.head.weight[candidates]
            bias = self.head.bias[candidates] if self.head.bias is not None else None
            logits = torch.einsum("bkd,bd->bk", weight, last_hidden)
            if bias is not None:
                logits = logits + bias
            labels = torch.zeros(B, dtype=torch.long, device=device)
            loss = F.cross_entropy(logits, labels)
            acc = (logits.argmax(dim=-1) == labels).float().mean()

            self.log(f"{stage}_loss", loss, prog_bar=True, on_step=True, on_epoch=True)
            self.log(f"{stage}_acc", acc, prog_bar=True, on_step=False, on_epoch=True)
            return loss

        # Validation/test: classify using the last fused hidden representation
        target = batch["target"]
        B = x.size(0)
        last_idx = lengths - 1
        last_hidden = x[torch.arange(B, device=x.device), last_idx]
        logits = self.head(last_hidden)
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

    def training_step(self, batch: Dict[str, Any], batch_idx: int):
        return self._step(batch, stage="train")

    def validation_step(self, batch: Dict[str, Any], batch_idx: int):
        self._step(batch, stage="val")

    def test_step(self, batch: Dict[str, Any], batch_idx: int):
        self._step(batch, stage="test")

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.hparams.lr, weight_decay=self.hparams.weight_decay)

    # Pure sequence task: no external cache/topic embedding interfaces


def build_model(
    vocab_size: int,
    model_cfg: Dict[str, Any],
    optim_cfg: Dict[str, Any],
) -> pl.LightningModule:
    """Build the CYAN-RNN model; interface consistent with training scripts.

    - `n_layers` is the number of RNN layers; `n_heads` is unused in this model (kept for config alignment).
    """
    m = CYANRNNLM(
        vocab_size=vocab_size,
        d_model=int(model_cfg.get("d_model", 256)),
        n_layers=int(model_cfg.get("n_layers", 1)),
        dropout=float(model_cfg.get("dropout", 0.1)),
        lr=float(optim_cfg.get("lr", 1e-3)),
        weight_decay=float(optim_cfg.get("weight_decay", 0.01)),
        rnn_type=str(model_cfg.get("rnn_type", "gru")),
    )
    m.hparams.topk = model_cfg.get("topk", [5, 10, 20])
    m.hparams.neg_k = int(model_cfg.get("neg_k", 100))
    return m
    
