"""What is actually IN the concentrated delta buckets on Haxe?

The localized run found a real but small order effect. Open question: is the
structure SEMANTIC (a genuine move vocabulary) or LEXICAL (adjacent OCaml
functions share naming prefixes, e.g. codegen.has_properties -> codegen.get_properties,
so the "move" is just a naming-convention direction)?

Bucket ids are per-region (each region fits its own basis), so we inspect the
largest regions separately. For each top bucket we report example pairs plus
lexical enrichment against that region's own baseline:

  same_module     src and tgt share the token before the first '.'
  tok_jaccard     Jaccard overlap of identifier tokens in the two titles
  same_kind       identical `kind` field

If the top buckets are just high-overlap same-module pairs, the effect is lexical.
"""

import re

import numpy as np
import polars as pl

from dyf.lazy_index import LazyIndex

DYF = "/Users/jdonaldson/Projects/haxe/src.dyf"
BITS = 4
TOP_REGIONS = 2
TOP_BUCKETS = 4


def unit(x, axis=-1):
    n = np.linalg.norm(x, axis=axis, keepdims=True)
    return x / np.where(n > 0, n, 1)


def toks(s):
    return set(t for t in re.split(r"[._\W]+", s.lower()) if t)


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
    kind = np.array(dfs["kind"].to_list())
    print("kind distribution:", dict(zip(*np.unique(kind, return_counts=True))))

    idx_sorted = dfs["idx"].to_numpy()
    files = np.array(dfs["file"].to_list())
    # consecutive pairs within file, in sorted position space
    same = files[:-1] == files[1:]
    p_src = np.flatnonzero(same)
    p_tgt = p_src + 1
    D = emb[idx_sorted[p_tgt]] - emb[idx_sorted[p_src]]
    mags = np.linalg.norm(D, axis=1)
    print(f"pairs: {len(D)}")

    reg_item = np.array([anc(int(l), by_id, 2) if l >= 0 else -1 for l in item_leaf])
    reg = reg_item[idx_sorted[p_src]]

    # magnitude band = the band where the effect was strongest (upper range)
    lo, hi = np.quantile(mags, 0.65), np.quantile(mags, 1.0)
    band = (mags >= lo) & (mags <= hi)
    print(f"band |d| {lo:.3f}-{hi:.3f}: {int(band.sum())} pairs\n")

    sizes = [(int((reg == r) & band.sum() if False else int(((reg == r) & band).sum())), r) for r in np.unique(reg)]
    sizes.sort(reverse=True)

    for n_in, r in sizes[:TOP_REGIONS]:
        sel = np.flatnonzero((reg == r) & band)
        if len(sel) < 200:
            continue
        Du = unit(D[sel])
        Dc = Du - Du.mean(0, keepdims=True)
        _, _, vt = np.linalg.svd(Dc, full_matrices=False)
        hp = vt[:BITS]
        codes = (((Du @ hp.T) >= 0) @ (2 ** np.arange(BITS))).astype(int)
        counts = np.bincount(codes, minlength=2**BITS)

        # region baseline
        st, tt = title[p_src[sel]], title[p_tgt[sel]]
        sk, tk = kind[p_src[sel]], kind[p_tgt[sel]]
        mod = np.array([a.split(".")[0] == b.split(".")[0] for a, b in zip(st, tt)])
        jac = np.array([len(toks(a) & toks(b)) / max(len(toks(a) | toks(b)), 1) for a, b in zip(st, tt)])
        kk = sk == tk
        print(
            f"=== region {r}: {len(sel)} pairs | baseline "
            f"same_module {mod.mean():.2f}  tok_jaccard {jac.mean():.3f}  "
            f"same_kind {kk.mean():.2f} ==="
        )

        for b in np.argsort(counts)[::-1][:TOP_BUCKETS]:
            m = codes == b
            if m.sum() < 10:
                continue
            print(
                f"\n  bucket {b:2d}  n={int(m.sum()):4d} ({m.mean():.1%})  "
                f"same_module {mod[m].mean():.2f}  tok_jaccard {jac[m].mean():.3f}  "
                f"same_kind {kk[m].mean():.2f}"
            )
            for j in np.flatnonzero(m)[:6]:
                print(f"      {st[j][:44]:<44} ->  {tt[j][:44]}")
        print()


if __name__ == "__main__":
    main()
