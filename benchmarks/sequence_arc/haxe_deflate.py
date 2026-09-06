"""Does ANY order structure survive after removing the identifier-length axis?

Buckets track len(tgt)-len(src) at r=0.52, so the Haxe "move vocabulary" is
substantially a chunking artifact: tree-sitter emitted both short `let e = ...`
bindings and long top-level functions, and the detected move is largely
binding -> function.

Test: fit the best linear length-axis on the REAL deltas, project every delta
(real and null) orthogonal to it, then re-run the localized real-vs-null H/b
comparison with the same magnitude banding, n-matching and null-vs-null floor.

If the effect vanishes, the Haxe result is entirely a chunk-kind artifact.
If it shrinks but survives, something else is there.
"""

import numpy as np
import polars as pl

from dyf.lazy_index import LazyIndex

DYF = "/Users/jdonaldson/Projects/haxe/src.dyf"
NSEED, BITS, MIN_N = 6, 4, 200


def unit(x, axis=-1):
    n = np.linalg.norm(x, axis=axis, keepdims=True)
    return x / np.where(n > 0, n, 1)


def load():
    li = LazyIndex(DYF)
    f = li.extract_all_fields()
    emb = unit(np.asarray(f["embeddings"], dtype=np.float32))
    fl = f["fields"]
    df = pl.DataFrame(
        {"idx": np.arange(len(emb)), "file": fl["file"], "line": [int(x) for x in fl["line"]], "title": fl["title"]}
    )
    tree = li.get_tree_structure()
    by_id = {n["node_id"]: n for n in tree}
    item_leaf = np.full(len(emb), -1, dtype=np.int64)
    for n in tree:
        if n["is_leaf"] and n["batch_index"] >= 0:
            rb = li.get_leaf(n["batch_index"])
            item_leaf[np.asarray(rb.column("item_index"))] = n["node_id"]
    return emb, df, by_id, item_leaf


def anc(nid, by_id, h):
    cur = nid
    while cur is not None and by_id[cur]["depth"] < h:
        cur = by_id[cur]["parent_id"]
    return cur if cur is not None else nid


def hnorm(X, b):
    Xu = unit(X)
    Xc = Xu - Xu.mean(0, keepdims=True)
    _, _, vt = np.linalg.svd(Xc, full_matrices=False)
    hp = vt[:b]
    codes = (((Xu @ hp.T) >= 0) @ (2 ** np.arange(b))).astype(np.int64)
    c = np.bincount(codes, minlength=2**b).astype(float)
    p = c[c > 0] / c.sum()
    return float(-(p * np.log2(p)).sum() / b)


def summarize(tag, rows):
    if not rows:
        print(f"  {tag:<34} (no cells)")
        return
    w = np.array([n for n, _ in rows], float)
    w /= w.sum()
    H = np.array([h for _, h in rows])
    real, nulls = H[:, 0], H[:, 1:]
    d_real = float(np.dot(w, real - nulls.mean(1)))
    dn = []
    for k in range(nulls.shape[1]):
        others = np.delete(nulls, k, axis=1).mean(1)
        dn.append(float(np.dot(w, nulls[:, k] - others)))
    mu, sd = float(np.mean(dn)), float(np.std(dn))
    z = (d_real - mu) / sd if sd > 0 else float("nan")
    print(f"  {tag:<34} cells {len(rows):3d}  real-vs-null {d_real:+.4f}  floor {mu:+.4f} +/- {sd:.4f}  z {z:+.2f}")


def main():
    emb, df, by_id, item_leaf = load()
    dfs = df.sort(["file", "line"])
    idx_s = dfs["idx"].to_numpy()
    files = np.array(dfs["file"].to_list())
    title = np.array(dfs["title"].to_list())
    same = files[:-1] == files[1:]
    ps, pt = np.flatnonzero(same), np.flatnonzero(same) + 1

    rng = np.random.default_rng(13)
    D_real = emb[idx_s[pt]] - emb[idx_s[ps]]
    S_real = idx_s[ps]

    # nulls: shuffle within file
    groups = []
    for _, sub in dfs.group_by(["file"], maintain_order=True):
        g = sub["idx"].to_numpy()
        if len(g) >= 2:
            groups.append(g)

    def build_null(r):
        S, D = [], []
        for g in groups:
            ii = r.permutation(g)
            S.append(ii[:-1])
            D.append(emb[ii[1:]] - emb[ii[:-1]])
        return np.concatenate(S), np.concatenate(D, axis=0)

    nulls = [build_null(np.random.default_rng(700 + s)) for s in range(NSEED)]

    # length axis, fit on real
    L = np.array([len(t.split(".")[-1]) for t in title])
    dlen = (L[pt] - L[ps]).astype(float)
    y = (dlen - dlen.mean()) / (dlen.std() + 1e-9)
    w_len = D_real.T @ y / len(y)
    w_len /= np.linalg.norm(w_len)
    print(f"length axis fitted on {len(D_real)} real deltas")

    def deflate(X):
        return X - np.outer(X @ w_len, w_len)

    reg_item = np.array([anc(int(l), by_id, 2) if l >= 0 else -1 for l in item_leaf])
    mags = np.linalg.norm(D_real, axis=1)
    qs = np.quantile(mags, [0.4, 0.65, 0.85, 1.0])
    edges = [(qs[0], qs[1]), (qs[1], qs[2]), (qs[2], qs[3])]

    for defl in (False, True):
        D_sets = [D_real] + [d for _, d in nulls]
        S_sets = [S_real] + [s for s, _ in nulls]
        if defl:
            D_sets = [deflate(d) for d in D_sets]
        regs = [reg_item[s] for s in S_sets]
        ms = [np.linalg.norm(d, axis=1) for d in D_sets]

        rows_g, rows_r = [], []
        for gi in [None] + sorted(set(regs[0].tolist())):
            for lo, hi in edges:
                idxs = []
                for reg, m in zip(regs, ms):
                    sel = (m >= lo) & (m < hi)
                    if gi is not None:
                        sel = sel & (reg == gi)
                    idxs.append(np.flatnonzero(sel))
                n = min(len(i) for i in idxs)
                if n < MIN_N:
                    continue
                hs = [hnorm(D[rng.choice(i, n, replace=False)], BITS) for D, i in zip(D_sets, idxs)]
                (rows_g if gi is None else rows_r).append((n, hs))

        print(f"\n--- length axis {'REMOVED' if defl else 'intact'} ---")
        summarize("GLOBAL basis", rows_g)
        summarize("PER-REGION basis", rows_r)


if __name__ == "__main__":
    main()
