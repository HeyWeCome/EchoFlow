import csv
import os
from typing import Dict, List, Tuple, Optional, Any

import torch
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl


def _detect_header(row: List[str]) -> bool:
    if not row:
        return False
    fields = [c.strip().lower() for c in row]
    return (
        len(fields) >= 3
        and fields[0] == "user_id"
        and fields[1] == "topic_id"
        and fields[2] == "timestamp"
    )


def read_interactions(
    path: str,
) -> List[Tuple[str, str, int]]:
    """Reads interactions.inter returning list of (user_id, topic_id, timestamp).

    Requires header with 'timestamp'. Optionally limit the number of rows for sampling.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Interactions file not found: {path}")

    rows: List[Tuple[str, str, int]] = []
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        first = next(reader, None)
        if first is not None and not _detect_header(first):
            # Treat as data
            if len(first) >= 3:
                try:
                    rows.append((first[0].strip(), first[1].strip(), int(first[2].strip())))
                except Exception:
                    pass
        for row in reader:
            if not row or len(row) < 3:
                continue
            user_id, topic_id, ts_str = row[0].strip(), row[1].strip(), row[2].strip()
            try:
                ts = int(ts_str)
            except Exception:
                continue
            rows.append((user_id, topic_id, ts))
    return rows


def build_cascades(rows: List[Tuple[str, str, int]]) -> Dict[str, List[Tuple[str, int]]]:
    """Group rows by topic_id into cascades and sort by timestamp ascending.

    Returns dict: topic_id -> list of (user_id, timestamp)
    """
    cascades: Dict[str, List[Tuple[str, int]]] = {}
    for u, t, ts in rows:
        cascades.setdefault(t, []).append((u, ts))
    for t in cascades:
        cascades[t].sort(key=lambda x: x[1])
    return cascades


def build_user_vocab(cascades: Dict[str, List[Tuple[str, int]]]) -> Dict[str, int]:
    """Build user_id -> index vocabulary. Index 0 is reserved for PAD."""
    users = set()
    for seq in cascades.values():
        for u, _ in seq:
            users.add(u)
    # PAD = 0; actual users start from 1
    user2idx = {u: i + 1 for i, u in enumerate(sorted(users))}
    return user2idx


def build_topic_vocab(cascades: Dict[str, List[Tuple[str, int]]]) -> Dict[str, int]:
    """Build topic_id -> index vocabulary. Index 0 is reserved for PAD/unused to align with embedding interface."""
    topics = sorted(list(cascades.keys()))
    topic2idx = {t: i + 1 for i, t in enumerate(topics)}
    return topic2idx


def encode_cascades(
    cascades: Dict[str, List[Tuple[str, int]]],
    user2idx: Dict[str, int],
    topic2idx: Dict[str, int],
    min_len: int = 4,
) -> List[Dict[str, Any]]:
    """Encode cascades to structured samples including topic idx, user idx sequence, and timestamp sequence.

    Each item is a dict: {"topic": int, "users": List[int], "times": List[int]}.
    - min_len default 4 ensures at least one train pair and val/test exist.
    """
    items: List[Dict[str, Any]] = []
    for topic_id, seq in cascades.items():
        if len(seq) < min_len:
            continue
        topic_idx = topic2idx.get(topic_id)
        if topic_idx is None:
            continue
        users_idx: List[int] = []
        times_seq: List[int] = []
        for u, ts in seq:
            if u in user2idx:
                users_idx.append(user2idx[u])
                times_seq.append(ts)
        if len(users_idx) < min_len:
            continue
        items.append({"topic": topic_idx, "users": users_idx, "times": times_seq})
    return items


class CascadeTrainDataset(Dataset):
    """Training dataset: from each cascade, create multiple (context, target) pairs
    for t in [1 .. L-3], so last two events are held out for val/test (leave-one-out)."""

    def __init__(self, sequences: List[List[int]]):
        self.samples: List[Tuple[List[int], int]] = []
        for seq in sequences:
            L = len(seq)
            # create pairs predicting token at position t, given [0..t-1]
            for t in range(1, L - 2):
                context = seq[:t]
                target = seq[t]
                self.samples.append((context, target))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[List[int], int]:
        return self.samples[idx]


class CascadeTrainSeqDataset(Dataset):
    """Sequence-level training dataset: one sample per cascade.
    Provides users and timestamps; excludes last two tokens when computing loss in model.

    To support time-aware chunking without leakage, samples are sorted by their last timestamp ascending.
    """

    def __init__(self, items: List[Dict[str, Any]]):
        # Sort by last event time to ensure monotonic t_checkpoint when no shuffle
        self.items = sorted(items, key=lambda x: x["times"][-1])

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        return self.items[idx]


class CascadeEvalDataset(Dataset):
    """Evaluation dataset: one sample per cascade.
    - For validation: context = seq[:L-2], target = seq[L-2]
    - For test: context = seq[:L-1], target = seq[L-1]

    Returns dicts including topic, context users, context times, and target.
    """

    def __init__(self, items: List[Dict[str, Any]], use_last: bool = False):
        self.items: List[Dict[str, Any]] = []
        for obj in items:
            users = obj["users"]
            times = obj["times"]
            topic = obj["topic"]
            L = len(users)
            if use_last:
                context_users = users[: L - 1]
                context_times = times[: L - 1]
                target = users[L - 1]
            else:
                context_users = users[: L - 2]
                context_times = times[: L - 2]
                target = users[L - 2]
            self.items.append({
                "topic": topic,
                "context_users": context_users,
                "context_times": context_times,
                "target": target,
            })

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        return self.items[idx]


def collate_eval(
    batch: List[Dict[str, Any]], pad_idx: int = 0, max_len: Optional[int] = None
):
    contexts_users = []
    contexts_times = []
    topics = []
    for obj in batch:
        cu = obj["context_users"]
        ct = obj["context_times"]
        if max_len is not None and len(cu) > max_len:
            cu = cu[-max_len:]
            ct = ct[-max_len:]
        contexts_users.append(torch.tensor(cu, dtype=torch.long))
        contexts_times.append(torch.tensor(ct, dtype=torch.long))
        topics.append(obj["topic"])
    targets = torch.tensor([obj["target"] for obj in batch], dtype=torch.long)
    lengths = torch.tensor([len(c) for c in contexts_users], dtype=torch.long)
    max_len = int(lengths.max().item())
    padded_users = torch.full((len(contexts_users), max_len), pad_idx, dtype=torch.long)
    padded_times = torch.zeros((len(contexts_times), max_len), dtype=torch.long)
    for i, (cu, ct) in enumerate(zip(contexts_users, contexts_times)):
        padded_users[i, : len(cu)] = cu
        padded_times[i, : len(ct)] = ct
    key_padding_mask = padded_users.eq(pad_idx)  # shape (B, T)
    topic_ids = torch.tensor(topics, dtype=torch.long)
    return {
        "input_ids": padded_users,  # (B, T)
        "timestamps": padded_times,  # (B, T)
        "topic_ids": topic_ids,  # (B,)
        "key_padding_mask": key_padding_mask,  # (B, T)
        "lengths": lengths,  # (B,)
        "target": targets,  # (B,)
    }


def collate_train_pairs(
    batch: List[Tuple[List[int], int]], pad_idx: int = 0, max_len: Optional[int] = None
):
    contexts = []
    targets = []
    for ctx, tgt in batch:
        if max_len is not None and len(ctx) > max_len:
            ctx = ctx[-max_len:]
        contexts.append(torch.tensor(ctx, dtype=torch.long))
        targets.append(tgt)
    targets_t = torch.tensor(targets, dtype=torch.long)
    lengths = torch.tensor([len(c) for c in contexts], dtype=torch.long)
    max_len_b = int(lengths.max().item())
    padded_users = torch.full((len(contexts), max_len_b), pad_idx, dtype=torch.long)
    for i, cu in enumerate(contexts):
        padded_users[i, : len(cu)] = cu
    key_padding_mask = padded_users.eq(pad_idx)
    return {
        "input_ids": padded_users,
        "key_padding_mask": key_padding_mask,
        "lengths": lengths,
        "target": targets_t,
    }


def collate_train_seq(batch: List[Dict[str, Any]], pad_idx: int = 0, max_len: Optional[int] = None):
    """Collate full sequences for sequence-level training with timestamps and topic ids.
    Returns padded input (users), timestamps, topic_ids, key padding mask, and lengths.
    Targets will be derived in the model by shifting input and masking last two.
    """
    users_list = []
    times_list = []
    topic_ids = []
    for obj in batch:
        users = obj["users"]
        times = obj["times"]
        if max_len is not None and len(users) > max_len:
            users = users[-max_len:]
            times = times[-max_len:]
        users_list.append(torch.tensor(users, dtype=torch.long))
        times_list.append(torch.tensor(times, dtype=torch.long))
        topic_ids.append(obj["topic"])
    lengths = torch.tensor([len(s) for s in users_list], dtype=torch.long)
    max_len = int(lengths.max().item())
    padded_users = torch.full((len(users_list), max_len), pad_idx, dtype=torch.long)
    padded_times = torch.zeros((len(times_list), max_len), dtype=torch.long)
    for i, (u, t) in enumerate(zip(users_list, times_list)):
        padded_users[i, : len(u)] = u
        padded_times[i, : len(t)] = t
    key_padding_mask = padded_users.eq(pad_idx)
    topic_ids = torch.tensor(topic_ids, dtype=torch.long)
    return {
        "input_ids": padded_users,  # (B, T)
        "timestamps": padded_times,  # (B, T)
        "topic_ids": topic_ids,  # (B,)
        "key_padding_mask": key_padding_mask,  # (B, T)
        "lengths": lengths,  # (B,)
    }


class CascadesDataModule(pl.LightningDataModule):
    def __init__(
        self,
        interactions_path: Optional[str] = None,
        dataset: Optional[str] = None,
        root: str = "dataset",
        min_len: int = 4,
        batch_size: int = 64,
        num_workers: int = 0,
        pin_memory: bool = True,
        persistent_workers: bool = True,
        prefetch_factor: Optional[int] = 2,
        train_max_len: Optional[int] = None,
        eval_max_len: Optional[int] = None,
        train_style: str = "seq",
    ):
        super().__init__()
        self.interactions_path = interactions_path
        self.dataset = dataset
        self.root = root
        self.min_len = min_len
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.persistent_workers = persistent_workers if num_workers > 0 else False
        self.prefetch_factor = prefetch_factor if num_workers > 0 else None
        self.train_max_len = train_max_len
        self.eval_max_len = eval_max_len
        self.train_style = train_style
        self._stats_printed = False

        # To be set in setup()
        self.user2idx: Dict[str, int] = {}
        self.topic2idx: Dict[str, int] = {}
        self.num_users: int = 0
        self.num_topics: int = 0
        self.rows: List[Tuple[str, str, int]] = []
        self.train_ds: Optional[Dataset] = None
        self.val_ds: Optional[Dataset] = None
        self.test_ds: Optional[Dataset] = None
        # When using time-aware training, disable shuffle to avoid leakage
        self.time_order: bool = False

    def _resolve_path(self) -> str:
        if self.interactions_path:
            return self.interactions_path
        if not self.dataset:
            raise ValueError("Either interactions_path or dataset must be provided")
        base = os.path.abspath(self.root)
        return os.path.join(base, self.dataset, "interactions.inter")

    def setup(self, stage: Optional[str] = None):
        path = self._resolve_path()
        rows = read_interactions(path)
        self.rows = rows
        cascades = build_cascades(rows)
        user2idx = build_user_vocab(cascades)
        topic2idx = build_topic_vocab(cascades)
        items = encode_cascades(cascades, user2idx, topic2idx, min_len=self.min_len)
        self.user2idx = user2idx
        self.topic2idx = topic2idx
        self.num_users = len(user2idx) + 1  # include PAD at 0
        self.num_topics = len(topic2idx) + 1

        try:
            if not self._stats_printed:
                n_users = len(user2idx)
                n_topics = len(topic2idx)
                n_inter = len(rows)
                density = float(n_inter) / float(max(1, n_users * n_topics))
                ts = [r[2] for r in rows]
                if ts:
                    t = torch.tensor(ts, dtype=torch.float32)
                    q = torch.quantile(t, torch.tensor([0.25, 0.5, 0.75]))
                    t_min = int(t.min().item())
                    t_max = int(t.max().item())
                    t_p25 = int(q[0].item())
                    t_p50 = int(q[1].item())
                    t_p75 = int(q[2].item())
                else:
                    t_min = t_max = t_p25 = t_p50 = t_p75 = 0
                lens = []
                for seq in cascades.values():
                    s = set(u for u, _ in seq)
                    lens.append(len(s))
                if lens:
                    med_len = int(torch.tensor(lens, dtype=torch.float32).median().item())
                else:
                    med_len = 0
                print(
                    f"[DataStats] users={n_users}, topics={n_topics}, interactions={n_inter}, density={density:.6f}"
                )
                print(
                    f"[DataStats] timestamps: min={t_min}, p25={t_p25}, median={t_p50}, p75={t_p75}, max={t_max}"
                )
                print(f"[DataStats] per-topic unique users median={med_len}")
                self._stats_printed = True
        except Exception:
            pass

        # Build datasets
        if str(self.train_style).lower() == "pairs":
            seqs = [obj["users"] for obj in items]
            self.train_ds = CascadeTrainDataset(seqs)
        else:
            self.train_ds = CascadeTrainSeqDataset(items)
        self.val_ds = CascadeEvalDataset(items, use_last=False)
        self.test_ds = CascadeEvalDataset(items, use_last=True)

    def train_dataloader(self):
        assert self.train_ds is not None
        kwargs = {
            "batch_size": self.batch_size,
            "shuffle": not self.time_order,
            "num_workers": self.num_workers,
            "pin_memory": self.pin_memory,
            "persistent_workers": self.persistent_workers,
            "collate_fn": (
                (lambda b: collate_train_pairs(b, pad_idx=0, max_len=self.train_max_len))
                if str(self.train_style).lower() == "pairs"
                else (lambda b: collate_train_seq(b, pad_idx=0, max_len=self.train_max_len))
            ),
        }
        if self.prefetch_factor is not None:
            kwargs["prefetch_factor"] = int(self.prefetch_factor)
        return DataLoader(self.train_ds, **kwargs)

    def val_dataloader(self):
        assert self.val_ds is not None
        kwargs = {
            "batch_size": self.batch_size,
            "shuffle": False,
            "num_workers": self.num_workers,
            "pin_memory": self.pin_memory,
            "persistent_workers": self.persistent_workers,
            "collate_fn": lambda b: collate_eval(b, pad_idx=0, max_len=self.eval_max_len),
        }
        if self.prefetch_factor is not None:
            kwargs["prefetch_factor"] = int(self.prefetch_factor)
        return DataLoader(self.val_ds, **kwargs)

    def test_dataloader(self):
        assert self.test_ds is not None
        kwargs = {
            "batch_size": self.batch_size,
            "shuffle": False,
            "num_workers": self.num_workers,
            "pin_memory": self.pin_memory,
            "persistent_workers": self.persistent_workers,
            "collate_fn": lambda b: collate_eval(b, pad_idx=0, max_len=self.eval_max_len),
        }
        if self.prefetch_factor is not None:
            kwargs["prefetch_factor"] = int(self.prefetch_factor)
        return DataLoader(self.test_ds, **kwargs)


def encode_events(rows: List[Tuple[str, str, int]], user2idx: Dict[str, int], topic2idx: Dict[str, int]) -> List[Tuple[int, int, int]]:
    """Convert raw (user_id, topic_id, ts) rows into indexed events list sorted by timestamp ascending.

    Returns list of (u_idx, topic_idx, ts).
    """
    events: List[Tuple[int, int, int]] = []
    for u_id, t_id, ts in rows:
        u_idx = user2idx.get(u_id)
        t_idx = topic2idx.get(t_id)
        if u_idx is None or t_idx is None:
            continue
        events.append((u_idx, t_idx, ts))
    events.sort(key=lambda x: x[2])
    return events
    
