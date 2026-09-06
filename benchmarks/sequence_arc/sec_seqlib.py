"""Shared primitives for the sequence-of-dyfs experiments (SEC 10-Q corpus).

These back the "when does the global fit go stale?" section of POSTGRES_NOTES.md.

Frozen-partition routing is verified in sec_frozen_probe.py:
    bucket_id = sum_i( (x @ H.T > 0)_i << i )
100% agreement with dyf-rs, and independent of which other points are present.

Config via env:
    SEC_FILINGS_DYF  path to filings.dyf   (default: ~/Projects/sec10quant/data/filings.dyf)
    DYF_SEQ_CACHE    npz cache dir         (default: ~/.cache/dyf_seq)
"""

import os

import numpy as np

CACHE = os.environ.get("DYF_SEQ_CACHE", os.path.expanduser("~/.cache/dyf_seq"))
FILINGS_DYF = os.environ.get("SEC_FILINGS_DYF", os.path.expanduser("~/Projects/sec10quant/data/filings.dyf"))
NPZ = os.path.join(CACHE, "filings.npz")

NUM_BITS = 4
MAX_DEPTH = 4
MIN_LEAF = 16
SEED = 42


def load():
    """Return (E unit-normalised f32, dates, tickers, sections, quarters), date-sorted."""
    if not os.path.exists(NPZ):
        raise SystemExit(f"missing {NPZ} -- run sec_extract.py first")
    d = np.load(NPZ, allow_pickle=True)
    E = d["embeddings"].astype(np.float32)
    D = d["dates"].astype(str)
    T = d["tickers"].astype(str)
    S = d["sections"].astype(str)
    order = np.argsort(D, kind="stable")
    E, D, T, S = E[order], D[order], T[order], S[order]
    E /= np.linalg.norm(E, axis=1, keepdims=True) + 1e-12
    Q = np.array([f"{x[:4]}Q{(int(x[5:7]) - 1) // 3 + 1}" for x in D])
    return E, D, T, S, Q


def build(E, idxs=None, num_bits=NUM_BITS, max_depth=MAX_DEPTH, min_leaf=None, seed=SEED):
    """Build a dyf tree over E[idxs]; return the flattened node list."""
    from dyf.dyf_tree import build_dyf_tree

    sub = E if idxs is None else E[idxs]
    tree = build_dyf_tree(
        sub,
        max_depth=max_depth,
        num_bits=num_bits,
        min_leaf_size=MIN_LEAF if min_leaf is None else min_leaf,
        seed=seed,
    )
    return flatten(tree, sub)


def flatten(tree, E):
    """Nested-dict tree -> flat node list with centroids, hyperplanes, bucket maps."""
    nodes = []

    def rec(t):
        nid = len(nodes)
        nodes.append(None)
        idxs = np.asarray(t["indices"], dtype=np.int64)
        cen = E[idxs].mean(0) if len(idxs) else np.zeros(E.shape[1], np.float32)
        cen = cen / (np.linalg.norm(cen) + 1e-12)
        kids = [rec(c) for c in t["children"]] if t["children"] else []
        hp = t.get("hyperplanes")
        hp = np.asarray(hp, dtype=np.float32) if hp is not None and kids else None
        nodes[nid] = {
            "children": kids,
            "hp": hp,
            "bmap": t.get("bucket_id_to_child") if kids else None,
            "centroid": cen.astype(np.float32),
            "n": len(idxs),
            "indices": idxs,
            "leaf_id": -1,
        }
        return nid

    rec(tree)
    lid = 0
    for n in nodes:
        if not n["children"]:
            n["leaf_id"] = lid
            lid += 1
    return nodes


def n_leaves(flat):
    return sum(1 for n in flat if n["leaf_id"] >= 0)


def route(E, flat):
    """Route points through a FROZEN partition.

    Returns (leaf_id per point, n_unseen_bucket). A point whose bucket had no members
    at build time has no child to receive it; it falls back to the nearest child
    centroid. That fallback count is the drift signal used throughout.
    """
    out = np.full(len(E), -1, dtype=np.int32)
    unseen = 0
    stack = [(0, np.arange(len(E), dtype=np.int64))]
    while stack:
        nid, idxs = stack.pop()
        node = flat[nid]
        if not len(idxs):
            continue
        if node["leaf_id"] >= 0:
            out[idxs] = node["leaf_id"]
            continue
        H, bmap, kids = node["hp"], node["bmap"], node["children"]
        if H is None or bmap is None:
            stack.append((kids[0], idxs))
            continue
        proj = E[idxs] @ H.T
        bid = ((proj > 0).astype(np.int64) << np.arange(H.shape[0])).sum(1)
        lut = np.full(1 << H.shape[0], -1, dtype=np.int64)
        for b, c in bmap.items():
            lut[int(b)] = int(c)
        child_of = lut[bid]
        miss = child_of < 0
        if miss.any():
            unseen += int(miss.sum())
            kc = np.stack([flat[k]["centroid"] for k in kids])
            child_of[miss] = (E[idxs[miss]] @ kc.T).argmax(1)
        for ci, k in enumerate(kids):
            sel = child_of == ci
            if sel.any():
                stack.append((k, idxs[sel]))
    return out, unseen


def leaf_centroids(E, assign, nl):
    """Recompute leaf centroids from current membership -- what a delta frame stores."""
    C = np.zeros((nl, E.shape[1]), np.float32)
    cnt = np.zeros(nl, np.int64)
    np.add.at(C, assign, E)
    np.add.at(cnt, assign, 1)
    C /= np.linalg.norm(C, axis=1, keepdims=True) + 1e-12
    return C, cnt


def fresh_assign(flat_fresh, n):
    a = np.full(n, -1, dtype=np.int32)
    for node in flat_fresh:
        if node["leaf_id"] >= 0:
            a[node["indices"]] = node["leaf_id"]
    return a


def ivf_search(Q, E, assign, C, probe, k=10):
    """Probe top-`probe` leaves by centroid sim, scan members exactly.

    Identical procedure for frozen and fresh trees so the comparison is fair.
    """
    nl = C.shape[0]
    probe = min(probe, nl)
    order = np.argsort(assign, kind="stable")
    starts = np.searchsorted(assign[order], np.arange(nl + 1))
    sims = Q @ C.T
    top = np.argpartition(-sims, probe - 1, axis=1)[:, :probe]
    res = np.zeros((len(Q), k), dtype=np.int64)
    for qi in range(len(Q)):
        cand = np.concatenate([order[starts[lf] : starts[lf + 1]] for lf in top[qi]])
        if len(cand) == 0:
            res[qi] = -1
            continue
        s = E[cand] @ Q[qi]
        kk = min(k, len(cand))
        best = cand[np.argpartition(-s, kk - 1)[:kk]]
        best = best[np.argsort(-(E[best] @ Q[qi]))]
        res[qi, :kk] = best
        res[qi, kk:] = -1
    return res


def scan_cost(Q, assign, C, probe):
    """Mean candidates scanned -- for equal-WORK rather than equal-probe comparison."""
    nl = C.shape[0]
    probe = min(probe, nl)
    cnt = np.bincount(assign, minlength=nl)
    top = np.argpartition(-(Q @ C.T), probe - 1, axis=1)[:, :probe]
    return float(cnt[top].sum(1).mean())


def exact_knn(Q, E, k=10, chunk=20000):
    out = np.zeros((len(Q), k), dtype=np.int64)
    for s in range(0, len(Q), 64):
        q = Q[s : s + 64]
        sims = np.empty((len(q), len(E)), np.float32)
        for c in range(0, len(E), chunk):
            sims[:, c : c + chunk] = q @ E[c : c + chunk].T
        idx = np.argpartition(-sims, k - 1, axis=1)[:, :k]
        for i in range(len(q)):
            out[s + i] = idx[i][np.argsort(-sims[i, idx[i]])]
    return out


def recall_at_k(got, truth):
    return float(np.mean([len(set(g.tolist()) & set(t.tolist())) / len(t) for g, t in zip(got, truth)]))
