"""Haxe localization re-run WITH A NOISE FLOOR.

The first pass reported real-vs-null H/b gaps of -0.005 (global) and -0.009/-0.015
(per-region) with no variance estimate, so they were uninterpretable.

Noise floor = NULL-vs-NULL. Each null seed is scored against the mean of the
others exactly as `real` is. Under no effect, real-vs-null should look like
null-vs-null. That is the only way to know whether -0.015 means anything.
"""

import numpy as np
import polars as pl

from dyf.lazy_index import LazyIndex

DYF = "/Users/jdonaldson/Projects/haxe/src.dyf"
NSEED = 6
BITS = 4
MIN_N = 200


def unit(x, axis=-1):
    n = np.linalg.norm(x, axis=axis, keepdims=True)
    return x / np.where(n > 0, n, 1)


def load():
    li = LazyIndex(DYF)
    f = li.extract_all_fields()
    emb = unit(np.asarray(f["embeddings"], dtype=np.float32))
    fl = f["fields"]
    df = pl.DataFrame({"idx": np.arange(len(emb)), "file": fl["file"], "line": [int(x) for x in fl["line"]]})
    tree = li.get_tree_structure()
    by_id = {n["node_id"]: n for n in tree}
    item_leaf = np.full(len(emb), -1, dtype=np.int64)
    for n in tree:
        if n["is_leaf"] and n["batch_index"] >= 0:
            rb = li.get_leaf(n["batch_index"])
            item_leaf[np.asarray(rb.column("item_index"))] = n["node_id"]
    return emb, df, by_id, item_leaf


def anc(node_id, by_id, height):
    cur = node_id
    while cur is not None and by_id[cur]["depth"] < height:
        cur = by_id[cur]["parent_id"]
    return cur if cur is not None else node_id


def groups_of(df):
    out = []
    for _, sub in df.sort(["file", "line"]).group_by(["file"], maintain_order=True):
        idx = sub["idx"].to_numpy()
        if len(idx) >= 2:
            out.append(idx)
    return out


def build(groups, emb, rng=None):
    S, D = [], []
    for idx in groups:
        ii = rng.permutation(idx) if rng is not None else idx
        S.append(ii[:-1])
        D.append(emb[ii[1:]] - emb[ii[:-1]])
    return np.concatenate(S), np.concatenate(D, axis=0)


def hnorm(X, b):
    Xu = unit(X)
    Xc = Xu - Xu.mean(0, keepdims=True)
    _, _, vt = np.linalg.svd(Xc, full_matrices=False)
    hp = vt[:b]
    codes = (((Xu @ hp.T) >= 0) @ (2 ** np.arange(hp.shape[0]))).astype(np.int64)
    c = np.bincount(codes, minlength=2**b).astype(float)
    p = c[c > 0] / c.sum()
    return float(-(p * np.log2(p)).sum() / b)


def cells(D_sets, reg_sets, edges, per_region, rng):
    """For each (region, band), return matched-n H/b for every set."""
    mags = [np.linalg.norm(D, axis=1) for D in D_sets]
    regs = sorted(set(reg_sets[0].tolist())) if per_region else [None]
    rows = []
    for gi in regs:
        for lo, hi in edges:
            idxs = []
            for D, reg, m in zip(D_sets, reg_sets, mags):
                sel = (m >= lo) & (m < hi)
                if gi is not None:
                    sel &= reg == gi
                idxs.append(np.flatnonzero(sel))
            n = min(len(i) for i in idxs)
            if n < MIN_N:
                continue
            hs = [hnorm(D[rng.choice(i, n, replace=False)], BITS) for D, i in zip(D_sets, idxs)]
            rows.append((n, hs))
    return rows


def summarize(tag, rows):
    if not rows:
        print(f"  {tag:<26} (no cell met MIN_N)")
        return
    w = np.array([n for n, _ in rows], float)
    w /= w.sum()
    H = np.array([hs for _, hs in rows])  # (cells, 1+NSEED)
    real = H[:, 0]
    nulls = H[:, 1:]
    d_real = float(np.dot(w, real - nulls.mean(1)))
    # noise floor: each null scored against the mean of the OTHER nulls
    d_null = []
    for k in range(nulls.shape[1]):
        others = np.delete(nulls, k, axis=1).mean(1)
        d_null.append(float(np.dot(w, nulls[:, k] - others)))
    mu, sd = float(np.mean(d_null)), float(np.std(d_null))
    z = (d_real - mu) / sd if sd > 0 else float("nan")
    print(
        f"  {tag:<26} cells {len(rows):3d}  real-vs-null {d_real:+.4f}   "
        f"null-vs-null {mu:+.4f} +/- {sd:.4f}   z {z:+.2f}"
    )


def main():
    emb, df, by_id, item_leaf = load()
    g = groups_of(df)
    rng = np.random.default_rng(11)
    S_r, D_r = build(g, emb)
    nulls = [build(g, emb, np.random.default_rng(600 + s)) for s in range(NSEED)]
    print(f"deltas {len(D_r)}  nulls {NSEED}  bits {BITS}  MIN_N {MIN_N}")

    m = np.linalg.norm(D_r, axis=1)
    qs = np.quantile(m, [0.4, 0.65, 0.85, 1.0])
    edges = [(qs[0], qs[1]), (qs[1], qs[2]), (qs[2], qs[3])]

    D_sets = [D_r] + [d for _, d in nulls]
    S_sets = [S_r] + [s for s, _ in nulls]

    for height in (3, 2):
        reg_item = np.array([anc(int(l), by_id, height) if l >= 0 else -1 for l in item_leaf])
        reg_sets = [reg_item[s] for s in S_sets]
        nreg = len(np.unique(reg_item))
        print(f"\n--- tree height {height}  ({nreg} source regions) ---")
        summarize("GLOBAL basis", cells(D_sets, reg_sets, edges, False, rng))
        summarize("PER-REGION basis", cells(D_sets, reg_sets, edges, True, rng))


if __name__ == "__main__":
    main()
