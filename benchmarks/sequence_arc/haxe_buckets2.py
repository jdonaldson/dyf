"""Is the bucket structure carrying IDENTIFIER LENGTH rather than semantics?

The token-overlap test came back flat, but the example pairs suggested a
different confound: source names are short local bindings (e, a, m, e1, v')
and targets are long top-level functions (find_array_write_access_raise).
Tree-sitter chunked both `let e = ...` bindings and full function definitions,
so "the move" may just be binding -> function, a property of the STATES.

Measure per bucket: leaf-name length of src and tgt (after stripping the module
prefix), the signed difference, and the rate at which src is a short (<=3 char)
identifier. Compare to the region baseline.
"""

import numpy as np
import polars as pl

from dyf.lazy_index import LazyIndex

DYF = "/Users/jdonaldson/Projects/haxe/src.dyf"
BITS = 4


def unit(x, axis=-1):
    n = np.linalg.norm(x, axis=axis, keepdims=True)
    return x / np.where(n > 0, n, 1)


def leaf_name(t):
    return t.split(".")[-1]


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
            "title": fl["title"],
            "kind": fl["kind"],
        }
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


def main():
    emb, df, by_id, item_leaf = load()
    dfs = df.sort(["file", "line"])
    title = np.array(dfs["title"].to_list())
    idx_sorted = dfs["idx"].to_numpy()
    files = np.array(dfs["file"].to_list())
    same = files[:-1] == files[1:]
    p_src, p_tgt = np.flatnonzero(same), np.flatnonzero(same) + 1
    D = emb[idx_sorted[p_tgt]] - emb[idx_sorted[p_src]]
    mags = np.linalg.norm(D, axis=1)
    reg_item = np.array([anc(int(l), by_id, 2) if l >= 0 else -1 for l in item_leaf])
    reg = reg_item[idx_sorted[p_src]]

    lo = np.quantile(mags, 0.65)
    band = mags >= lo

    L = np.array([len(leaf_name(t)) for t in title])
    Ls, Lt = L[p_src], L[p_tgt]
    short_src = Ls <= 3

    # global correlation first: does delta direction track length change at all?
    sel_all = np.flatnonzero(band)
    dlen = (Lt - Ls)[sel_all]
    Du_all = unit(D[sel_all])
    # single best linear direction for length-change
    y = (dlen - dlen.mean()) / (dlen.std() + 1e-9)
    w = Du_all.T @ y / len(y)
    proj = Du_all @ (w / np.linalg.norm(w))
    r = np.corrcoef(proj, dlen)[0, 1]
    print(f"band n={len(sel_all)}  corr(delta projected on best length-axis, len(tgt)-len(src)) = {r:.3f}")
    print(
        f"  mean len src {Ls[sel_all].mean():.1f}  tgt {Lt[sel_all].mean():.1f}  "
        f"short-src rate {short_src[sel_all].mean():.2f}\n"
    )

    sizes = sorted(((int(((reg == rr) & band).sum()), rr) for rr in np.unique(reg)), reverse=True)
    for n_in, rr in sizes[:2]:
        sel = np.flatnonzero((reg == rr) & band)
        if len(sel) < 200:
            continue
        Du = unit(D[sel])
        Dc = Du - Du.mean(0, keepdims=True)
        _, _, vt = np.linalg.svd(Dc, full_matrices=False)
        hp = vt[:BITS]
        codes = (((Du @ hp.T) >= 0) @ (2 ** np.arange(BITS))).astype(int)
        counts = np.bincount(codes, minlength=2**BITS)
        s, t, sh = Ls[sel], Lt[sel], short_src[sel]
        print(
            f"=== region {rr}: n={len(sel)} | baseline  len_src {s.mean():5.1f}  "
            f"len_tgt {t.mean():5.1f}  d {np.mean(t - s):+5.1f}  short_src {sh.mean():.2f} ==="
        )
        for b in np.argsort(counts)[::-1][:5]:
            m = codes == b
            if m.sum() < 10:
                continue
            print(
                f"  bucket {b:2d} n={int(m.sum()):4d}  len_src {s[m].mean():5.1f}  "
                f"len_tgt {t[m].mean():5.1f}  d {np.mean(t[m] - s[m]):+5.1f}  "
                f"short_src {sh[m].mean():.2f}"
            )
        print()


if __name__ == "__main__":
    main()
