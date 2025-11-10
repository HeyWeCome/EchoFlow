from typing import Any, Dict, List, Tuple


def build_training_events_from_items(items: List[Dict[str, Any]]) -> List[Tuple[int, int, int]]:
    """
    Build diffusion events strictly from training contexts of encoded cascades.

    Each item is {"topic": int, "users": List[int], "times": List[int]}.
    We use only the first L-2 events of each cascade (leave last two out), to
    avoid leakage into validation/test.

    Returns list of (u_idx, topic_idx, ts) sorted by timestamp ascending.
    """
    events: List[Tuple[int, int, int]] = []
    for obj in items:
        topic_idx = int(obj["topic"])
        users: List[int] = list(obj["users"])  # already encoded indices
        times: List[int] = list(obj["times"])  # raw timestamps ascending
        L = len(users)
        if L < 3:
            # require at least 3 to leave last two out and keep >=1 event
            continue
        cut = max(0, L - 2)
        for i in range(cut):
            events.append((int(users[i]), topic_idx, int(times[i])))
    events.sort(key=lambda x: x[2])
    return events


def build_training_events_from_dm(dm) -> List[Tuple[int, int, int]]:
    """Helper to build training diffusion events from a CascadesDataModule.

    Uses dm.train_ds.items which are encoded cascades from encode_cascades.
    """
    # Lazy import to avoid circular dependency at module import time
    try:
        from pl.data_module import CascadesDataModule  # type: ignore
    except Exception:
        CascadesDataModule = None  # noqa: N806

    items = getattr(dm.train_ds, "items", None)
    if items is None:
        raise ValueError("DataModule has no train items for building training events")
    return build_training_events_from_items(items)
    