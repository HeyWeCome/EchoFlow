from typing import List

import math
import torch


def _build_time_bins(timestamps: List[int], time_step_split: int) -> List[int]:
    """Build monotonically non-decreasing bin thresholds from sorted timestamps.

    If `time_step_split` <= 1, returns a single threshold (max ts or 0).
    """
    if time_step_split <= 1:
        return [max(timestamps) if timestamps else 0]
    ts_sorted = sorted(int(x) for x in timestamps)
    if not ts_sorted:
        return [0] * time_step_split
    L = len(ts_sorted)
    bins: List[int] = []
    for k in range(1, time_step_split + 1):
        idx = min(L - 1, max(0, math.floor(L * k / time_step_split) - 1))
        bins.append(int(ts_sorted[idx]))
    # ensure non-decreasing
    for i in range(1, len(bins)):
        if bins[i] < bins[i - 1]:
            bins[i] = bins[i - 1]
    return bins


def _map_ts_to_bin(ts: torch.Tensor, bins: List[int]) -> torch.Tensor:
    """Map each timestamp to the last bin with threshold <= ts.

    Returns LongTensor of indices in [0, len(bins)-1].
    """
    if ts.numel() == 0:
        return torch.zeros_like(ts, dtype=torch.long)
    device = ts.device
    thr = torch.tensor(bins, device=device, dtype=ts.dtype)
    idx = torch.bucketize(ts, thr, right=True) - 1
    idx = idx.clamp(min=0, max=len(bins) - 1)
    return idx.long()


def _choose_time_step_split(
    timestamps: List[int],
    min_bins: int = 3,
    max_bins: int = 20,
) -> int:
    """Choose an automatic number of time bins based on timestamp distribution.

    Heuristic combines Freedman–Diaconis rule and Sturges' formula, clamped.
    """
    ts_sorted = sorted(int(x) for x in timestamps)
    n = len(ts_sorted)
    if n == 0:
        return max(min_bins, 1)
    rng = max(0, ts_sorted[-1] - ts_sorted[0])
    q1 = _percentile_int(ts_sorted, 0.25)
    q3 = _percentile_int(ts_sorted, 0.75)
    iqr = max(1, q3 - q1)
    width = 2.0 * iqr / (n ** (1.0 / 3.0))
    k_fd = int(math.ceil(rng / width)) if (width > 0 and rng > 0) else 0
    k_st = int(math.ceil(math.log2(n))) + 1 if n > 1 else 1
    k = k_fd if k_fd > 0 else k_st
    k = max(min_bins, min(k, max_bins))
    return k


def _percentile_int(ts_sorted: List[int], p: float) -> int:
    """Integer percentile using linear interpolation over a pre-sorted list."""
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
    