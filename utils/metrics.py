from typing import Dict, List

import torch


def compute_hit_map_at_k(logits: torch.Tensor, targets: torch.Tensor, topk: List[int]) -> Dict[str, torch.Tensor]:
    """Compute Hit@k and MAP@k for next-item classification.

    Args:
        logits: (B, V) scores for each class.
        targets: (B,) ground-truth class indices.
        topk: list of k values.

    Returns:
        dict mapping metric names to scalars (Torch tensors).
    """
    metrics: Dict[str, torch.Tensor] = {}
    if logits.ndim != 2:
        raise ValueError("logits must be shape (B, V)")
    if targets.ndim != 1:
        raise ValueError("targets must be shape (B,)")
    B = logits.size(0)

    # argsort for rank computation
    sorted_idx = torch.argsort(logits, dim=-1, descending=True)
    eq = (sorted_idx == targets.unsqueeze(1))  # (B, V)
    pos = torch.argmax(eq.int(), dim=1)  # (B,) position where target appears
    found = eq.any(dim=1)  # (B,)

    for k in topk:
        # Top-k indices
        topk_idx = sorted_idx[:, :k]
        hit = (topk_idx == targets.unsqueeze(1)).any(dim=1).float().mean()
        metrics[f"hit@{k}"] = hit
        # AP@k for single relevant item: 1/(rank) if rank<=k else 0
        ap = torch.where(found & (pos < k), 1.0 / (pos.float() + 1.0), torch.zeros_like(pos, dtype=torch.float))
        metrics[f"map@{k}"] = ap.mean()

    return metrics


def compute_accuracy(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """Compute top-1 classification accuracy.

    Args:
        logits: (B, V)
        targets: (B,)
    Returns:
        scalar tensor accuracy
    """
    preds = logits.argmax(dim=-1)
    return (preds == targets).float().mean()


def compute_accuracy(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """Compute top-1 classification accuracy.

    Args:
        logits: (B, V)
        targets: (B,)
    Returns:
        scalar tensor accuracy
    """
    preds = logits.argmax(dim=-1)
    return (preds == targets).float().mean()


def compute_accuracy(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """Compute top-1 classification accuracy.

    Args:
        logits: (B, V)
        targets: (B,)
    Returns:
        scalar tensor accuracy
    """
    preds = logits.argmax(dim=-1)
    return (preds == targets).float().mean()
    