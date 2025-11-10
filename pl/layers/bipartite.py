from typing import List, Tuple

import numpy as np
import scipy.sparse as sp
import torch


def build_bipartite_adj_static(
    cas_size: int,
    user_size: int,
    events: List[Tuple[int, int, int]],
) -> torch.Tensor:
    """Build normalized static bipartite adjacency between cascades and users.

    Layout: cascades [0..cas_size-1], users [cas_size..cas_size+user_size-1].
    Events use indices with PAD=0 reserved; we skip zeros and shift to 0-based.
    Returns a symmetric normalized sparse COO tensor (float32).
    """
    Vc = int(cas_size)
    Vu = int(user_size)
    V = Vc + Vu

    # Build bipartite incidence R (Vc x Vu) with undirected edges
    R = sp.dok_matrix((Vc, Vu), dtype=np.float32)
    for u_idx, c_idx, _ in events:
        if int(u_idx) <= 0 or int(c_idx) <= 0:
            continue
        u = int(u_idx) - 1
        c = int(c_idx) - 1
        R[c, u] = 1.0
    R = R.tolil()

    # Assemble full adjacency and symmetrize
    A = sp.dok_matrix((V, V), dtype=np.float32)
    A = A.tolil()
    A[:Vc, Vc:] = R
    A[Vc:, :Vc] = R.T
    # add self-loops
    for i in range(V):
        A[i, i] = 1.0
    A = A.tocsr()

    # Symmetric normalize: D^{-1/2} A D^{-1/2}
    rowsum = np.array(A.sum(1)).flatten()
    rowsum = np.clip(rowsum, a_min=1e-6, a_max=None)
    D_inv_sqrt = sp.diags(np.power(rowsum, -0.5))
    A_hat = D_inv_sqrt.dot(A).dot(D_inv_sqrt)
    coo = A_hat.tocoo()
    idx = torch.tensor(np.vstack([coo.row, coo.col]), dtype=torch.long)
    val = torch.tensor(coo.data, dtype=torch.float32)
    return torch.sparse_coo_tensor(idx, val, size=(V, V)).coalesce()


def build_bipartite_adj_snapshots(
    cas_size: int,
    user_size: int,
    events: List[Tuple[int, int, int]],
    time_bins: List[int],
) -> List[torch.Tensor]:
    """Build normalized bipartite adjacency snapshots between cascades and users.

    Layout: cascades [0..cas_size-1], users [cas_size..cas_size+user_size-1].
    Edges added cumulatively up to each bin threshold using training events.
    Each snapshot adds self-loops and applies symmetric normalization.
    Returns list of sparse COO tensors (float32) of shape (V, V).
    """
    Vc = int(cas_size)
    Vu = int(user_size)
    V = Vc + Vu

    events_sorted = sorted(events, key=lambda x: int(x[2]))
    snaps_edges: List[List[Tuple[int, int, float]]] = []
    cur: List[Tuple[int, int, float]] = []
    j = 0
    for b in time_bins:
        b_int = int(b)
        while j < len(events_sorted) and int(events_sorted[j][2]) <= b_int:
            u_idx, c_idx, _ = events_sorted[j]
            if int(u_idx) <= 0 or int(c_idx) <= 0:
                j += 1
                continue
            u_node = Vc + (int(u_idx) - 1)
            c_node = int(c_idx) - 1
            cur.append((c_node, u_node, 1.0))
            cur.append((u_node, c_node, 1.0))
            j += 1
        snaps_edges.append(list(cur))

    snaps: List[torch.Tensor] = []
    for s_edges in snaps_edges:
        # add self-loops
        s_all = s_edges + [(i, i, 1.0) for i in range(V)]
        if not s_all:
            idx = torch.arange(V, dtype=torch.long)
            A = torch.sparse_coo_tensor(torch.stack([idx, idx], dim=0), torch.ones(V, dtype=torch.float32), size=(V, V))
            snaps.append(A.coalesce())
            continue
        rows = torch.tensor([e[0] for e in s_all], dtype=torch.long)
        cols = torch.tensor([e[1] for e in s_all], dtype=torch.long)
        vals = torch.tensor([e[2] for e in s_all], dtype=torch.float32)
        A = torch.sparse_coo_tensor(torch.stack([rows, cols], dim=0), vals, size=(V, V)).coalesce()
        deg = torch.sparse.sum(A, dim=1).to_dense()
        deg = torch.clamp(deg, min=1.0)
        inv_sqrt = deg.pow(-0.5)
        r = A.indices()[0]
        c = A.indices()[1]
        norm_vals = A.values() * inv_sqrt.index_select(0, r) * inv_sqrt.index_select(0, c)
        A_hat = torch.sparse_coo_tensor(A.indices(), norm_vals, size=(V, V)).coalesce()
        snaps.append(A_hat)
    return snaps
