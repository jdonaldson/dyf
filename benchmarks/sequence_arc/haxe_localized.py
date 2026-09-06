"""Haxe re-run with LOCALIZATION — the generic version of chroma root-relative framing.

The music result showed absolute-frame differencing destroys a vocabulary that
provably exists, and that re-expressing each delta in a frame determined by its
SOURCE recovers it. Chroma had a group structure (C12) making that easy. Here we
test the two domain-free analogues the notes actually propose:

  GLOBAL      one LSH basis fit over all deltas          (what the first run did)
  PER-REGION  a basis fit separately within each source-state region of the dyf
              tree  -- literally "cluster deltas within a state-leaf"
  SRC-ALIGNED Householder-rotate each delta into a frame where its source vector
              points at a fixed reference -- "the frame follows the source"

Controls carried over from the music run, both essential:
  * MAGNITUDE BANDING with one absolute threshold applied to real and null, since
    small deltas are direction-noise and null deltas are systematically larger.
  * n-MATCHING per (region, band). Fitting a b-bit basis on few points trivially
    concentrates the histogram, so real and null must be compared at equal n.
"""

import numpy as np
import polars as pl

from dyf.lazy_index import LazyIndex

DYF = "/Users/jdonaldson/Projects/haxe/src.dyf"
NSEED = 3
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
    df = pl.DataFrame(
        {
            "idx": np.arange(len(emb)),
            "file": fl["file"],
            "line": [int(x) for x in fl["line"]],
        }
    )
    # item -> leaf node_id
    tree = li.get_tree_structure()
    by_id = {n["node_id"]: n for n in tree}
    item_leaf = np.full(len(emb), -1, dtype=np.int64)
    for n in tree:
        if n["is_leaf"] and n["batch_index"] >= 0:
            rb = li.get_leaf(n["batch_index"])
            ii = np.asarray(rb.column("item_index"))
            item_leaf[ii] = n["node_id"]
    return emb, df, by_id, item_leaf


def ancestor_at_height(node_id, by_id, height):
    """Walk up until node height >= target (root has the largest height)."""
    cur = node_id
    while cur is not None and by_id[cur]["depth"] < height:
        cur = by_id[cur]["parent_id"]
    return cur if cur is not None else node_id


def pairs_per_file(df):
    out = []
    for _, sub in df.sort(["file", "line"]).group_by(["file"], maintain_order=True):
        idx = sub["idx"].to_numpy()
        if len(idx) >= 2:
            out.append(idx)
    return out


def build(groups, emb, rng=None):
    """Return (src_idx, delta) for consecutive pairs."""
    S, D = [], []
    for idx in groups:
        ii = rng.permutation(idx) if rng is not None else idx
        S.append(ii[:-1])
        D.append(emb[ii[1:]] - emb[ii[:-1]])
    return np.concatenate(S), np.concatenate(D, axis=0)


def householder_align(src, delta, ref):
    """Rotate each delta by the Householder reflection taking src -> ref."""
    v = src - ref
    nv = np.linalg.norm(v, axis=1, keepdims=True)
    v = v / np.where(nv > 1e-8, nv, 1)
    return delta - 2 * np.sum(v * delta, axis=1, keepdims=True) * v


def hnorm(X, b):
    if len(X) < 50:
        return np.nan
    Xu = unit(X)
    Xc = Xu - Xu.mean(0, keepdims=True)
    _, _, vt = np.linalg.svd(Xc, full_matrices=False)
    hp = vt[:b]
    codes = (((Xu @ hp.T) >= 0) @ (2 ** np.arange(hp.shape[0]))).astype(np.int64)
    counts = np.bincount(codes, minlength=2**b).astype(float)
    p = counts[counts > 0] / counts.sum()
    return float(-(p * np.log2(p)).sum() / b)


def banded(mags, edges):
    return [(mags >= lo) & (mags < hi) for lo, hi in edges]


def evaluate(tag, Dr, Dn_list, reg_r, reg_n_list, edges, per_region, rng):
    """Weighted mean H/b for real vs null, n-matched within (region, band)."""
    mr = np.linalg.norm(Dr, axis=1)
    real_h, null_h, wts = [], [], []
    groups_r = [None] if not per_region else sorted(set(reg_r.tolist()))
    for gi in groups_r:
        selg_r = np.ones(len(Dr), bool) if gi is None else (reg_r == gi)
        for bi, (lo, hi) in enumerate(edges):
            sr = selg_r & (mr >= lo) & (mr < hi)
            nr = int(sr.sum())
            if nr < MIN_N:
                continue
            cand = []
            for Dn, reg_n in zip(Dn_list, reg_n_list):
                mn = np.linalg.norm(Dn, axis=1)
                selg_n = np.ones(len(Dn), bool) if gi is None else (reg_n == gi)
                sn = selg_n & (mn >= lo) & (mn < hi)
                if int(sn.sum()) >= MIN_N:
                    cand.append(np.flatnonzero(sn))
            if not cand:
                continue
            n = min([nr] + [len(c) for c in cand])
            ri = rng.choice(np.flatnonzero(sr), n, replace=False)
            hr = hnorm(Dr[ri], BITS)
            hns = [hnorm(Dn_list[k][rng.choice(c, n, replace=False)], BITS) for k, c in enumerate(cand)]
            if np.isnan(hr) or not len(hns):
                continue
            real_h.append(hr)
            null_h.append(np.nanmean(hns))
            wts.append(n)
    if not wts:
        print(f"  {tag:<28} (no cell met MIN_N)")
        return
    w = np.array(wts, float)
    w /= w.sum()
    R = float(np.dot(w, real_h))
    N = float(np.dot(w, null_h))
    print(f"  {tag:<28} cells {len(wts):3d}  H/b real {R:.4f}  null {N:.4f}  delta {R - N:+.4f}")


def main():
    emb, df, by_id, item_leaf = load()
    groups = pairs_per_file(df)
    rng = np.random.default_rng(7)

    S_r, D_r = build(groups, emb)
    nulls = [build(groups, emb, np.random.default_rng(500 + s)) for s in range(NSEED)]
    print(f"deltas real {len(D_r)}  null sets {len(nulls)}")

    m = np.linalg.norm(D_r, axis=1)
    qs = np.quantile(m, [0.4, 0.65, 0.85, 1.0])
    edges = [(qs[0], qs[1]), (qs[1], qs[2]), (qs[2], qs[3])]
    print("magnitude bands:", [f"{lo:.3f}-{hi:.3f}" for lo, hi in edges])

    for height in (3, 2):
        reg_of_item = np.array([ancestor_at_height(int(l), by_id, height) if l >= 0 else -1 for l in item_leaf])
        sizes = np.bincount(np.searchsorted(np.unique(reg_of_item), reg_of_item))
        print(
            f"\n--- source-region granularity: tree height {height} "
            f"({len(np.unique(reg_of_item))} regions, median {np.median(sizes):.0f} items) ---"
        )

        reg_r = reg_of_item[S_r]
        reg_n_list = [reg_of_item[s] for s, _ in nulls]
        Dn_list = [d for _, d in nulls]

        evaluate("GLOBAL basis", D_r, Dn_list, reg_r, reg_n_list, edges, False, rng)
        evaluate("PER-REGION basis", D_r, Dn_list, reg_r, reg_n_list, edges, True, rng)

        ref = np.zeros(emb.shape[1], np.float32)
        ref[0] = 1.0
        Hr = householder_align(emb[S_r], D_r, ref)
        Hn = [householder_align(emb[s], d, ref) for s, d in nulls]
        evaluate("SRC-ALIGNED (global)", Hr, Hn, reg_r, reg_n_list, edges, False, rng)
        evaluate("SRC-ALIGNED + PER-REGION", Hr, Hn, reg_r, reg_n_list, edges, True, rng)


if __name__ == "__main__":
    main()
