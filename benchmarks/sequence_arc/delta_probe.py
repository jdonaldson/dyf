"""Delta-space probe on the Haxe compiler src.dyf corpus.

Question: do consecutive code chunks within a file exhibit a *move vocabulary* —
a small set of recurring transition directions — or are deltas near-random?

Null: shuffle chunk order WITHIN each file and recompute deltas. This preserves
the embedding set, the file grouping, file lengths, and the state distribution;
it breaks ONLY sequential adjacency. So any real-vs-null gap is attributable to
order, not to anisotropy of the corpus.
"""

import numpy as np
import polars as pl

from dyf.lazy_index import LazyIndex

RNG = np.random.default_rng(0)
DYF_PATH = "/Users/jdonaldson/Projects/haxe/src.dyf"


def load():
    li = LazyIndex(DYF_PATH)
    f = li.extract_all_fields()
    emb = np.asarray(f["embeddings"], dtype=np.float32)
    fl = f["fields"]
    df = pl.DataFrame(
        {
            "idx": np.arange(len(emb)),
            "file": fl["file"],
            "line": [int(x) for x in fl["line"]],
            "kind": fl["kind"],
            "title": fl["title"],
        }
    )
    return emb, df


def unit(x, axis=-1):
    n = np.linalg.norm(x, axis=axis, keepdims=True)
    return x / np.where(n > 0, n, 1)


def seq_order(df):
    """Index arrays grouped per file, sorted by line."""
    out = []
    for (fname,), sub in df.sort(["file", "line"]).group_by(["file"], maintain_order=True):
        idx = sub["idx"].to_numpy()
        if len(idx) >= 2:
            out.append(idx)
    return out


def deltas_from(groups, emb, shuffle_rng=None):
    ds = []
    for idx in groups:
        if shuffle_rng is not None:
            idx = shuffle_rng.permutation(idx)
        e = emb[idx]
        ds.append(np.diff(e, axis=0))
    return np.concatenate(ds, axis=0)


def spectrum(X, center=True):
    """Return eigenvalue spectrum of the covariance of X (rows = samples)."""
    Xc = X - X.mean(0, keepdims=True) if center else X
    # economical: use SVD on the (n,d) matrix
    s = np.linalg.svd(Xc, compute_uv=False)
    lam = (s**2) / max(len(Xc) - 1, 1)
    return lam


def eff_rank(lam):
    """Participation ratio + components to reach 90% variance."""
    lam = lam[lam > 0]
    pr = (lam.sum() ** 2) / (lam**2).sum()
    c = np.cumsum(lam) / lam.sum()
    k90 = int(np.searchsorted(c, 0.90) + 1)
    return pr, k90


def sign_lsh(X, num_bits, fit_on=None):
    """dyf-style sign LSH: top-`num_bits` PCA directions as origin-passing
    hyperplanes, sign of projection, bit-packed. Matches classifier.py:830-832.
    `fit_on` lets you fit the basis on one set and apply to another."""
    src = X if fit_on is None else fit_on
    srcc = src - src.mean(0, keepdims=True)
    # top components via SVD
    _, _, vt = np.linalg.svd(srcc, full_matrices=False)
    hp = vt[:num_bits]
    signs = (X @ hp.T) >= 0
    powers = 2 ** np.arange(num_bits)
    return (signs @ powers).astype(np.int64)


def bucket_stats(codes, num_bits):
    counts = np.bincount(codes, minlength=2**num_bits).astype(np.float64)
    p = counts[counts > 0] / counts.sum()
    H = -(p * np.log2(p)).sum()
    occ = int((counts > 0).sum())
    srt = np.sort(counts)[::-1]
    top10 = srt[:10].sum() / counts.sum()
    return {
        "H_norm": H / num_bits,
        "occupied": occ,
        "top10_mass": top10,
    }


def main():
    emb, df = load()
    print(f"corpus: {emb.shape[0]:,} chunks x {emb.shape[1]}d")
    groups = seq_order(df)
    lens = np.array([len(g) for g in groups])
    print(
        f"files with >=2 chunks: {len(groups):,}  |  chunks/file: "
        f"median {np.median(lens):.0f}, mean {lens.mean():.1f}, max {lens.max()}"
    )

    embn = unit(emb)
    D_real = deltas_from(groups, embn)
    print(f"deltas (real): {D_real.shape[0]:,}")

    # ---- (2) delta magnitude -------------------------------------------------
    dn = np.linalg.norm(D_real, axis=1)
    # adjacent cosine, since states are unit norm: |d|^2 = 2(1-cos)
    cos_adj = 1 - (dn**2) / 2
    print("\n== consecutive-chunk geometry ==")
    print(f"  ||d||/||e||   median {np.median(dn):.3f}   mean {dn.mean():.3f}")
    print(f"  cos(adjacent) median {np.median(cos_adj):.3f}   mean {cos_adj.mean():.3f}")

    # free drift statistic: ||mean(d)||
    print(f"  ||mean(d)||   {np.linalg.norm(D_real.mean(0)):.4f}  (vs mean ||d|| {dn.mean():.3f})")

    # ---- nulls ---------------------------------------------------------------
    NSEED = 5
    D_null = [deltas_from(groups, embn, shuffle_rng=np.random.default_rng(100 + s)) for s in range(NSEED)]

    # ---- (1) spectra ---------------------------------------------------------
    print("\n== effective rank (unit-normalized directions) ==")
    Un_states = embn
    Un_real = unit(D_real)
    pr_s, k90_s = eff_rank(spectrum(Un_states))
    pr_r, k90_r = eff_rank(spectrum(Un_real))
    prs_n, k90s_n = zip(*[eff_rank(spectrum(unit(d))) for d in D_null])
    print(f"  states        PR {pr_s:7.1f}   k90 {k90_s:4d}")
    print(f"  deltas real   PR {pr_r:7.1f}   k90 {k90_r:4d}")
    print(
        f"  deltas null   PR {np.mean(prs_n):7.1f} +/- {np.std(prs_n):.1f}   "
        f"k90 {np.mean(k90s_n):.0f} +/- {np.std(k90s_n):.1f}"
    )

    # ---- (3) Zipf / bucket concentration ------------------------------------
    print("\n== sign-LSH bucket concentration (each fit in its OWN basis) ==")
    print(f"{'bits':>5} {'set':>12} {'H/bits':>8} {'occupied':>10} {'top10':>8}")
    for b in (4, 6, 8, 10):
        cs = bucket_stats(sign_lsh(Un_states, b), b)
        cr = bucket_stats(sign_lsh(Un_real, b), b)
        cn = [bucket_stats(sign_lsh(unit(d), b), b) for d in D_null]
        print(f"{b:>5} {'states':>12} {cs['H_norm']:8.4f} {cs['occupied']:10d} {cs['top10_mass']:8.3f}")
        print(f"{b:>5} {'delta real':>12} {cr['H_norm']:8.4f} {cr['occupied']:10d} {cr['top10_mass']:8.3f}")
        print(
            f"{b:>5} {'delta null':>12} {np.mean([c['H_norm'] for c in cn]):8.4f} "
            f"{np.mean([c['occupied'] for c in cn]):10.0f} "
            f"{np.mean([c['top10_mass'] for c in cn]):8.3f}"
        )

    # ---- (4) transition structure over move codes ---------------------------
    # Recompute per-file so we can build bigrams without crossing file boundaries.
    print("\n== move-code transition structure (b=6) ==")
    b = 6
    hpfit = unit(D_real)
    srcc = hpfit - hpfit.mean(0, keepdims=True)
    _, _, vt = np.linalg.svd(srcc, full_matrices=False)
    hp = vt[:b]
    powers = 2 ** np.arange(b)

    def codes_per_file(shuffle_rng=None):
        seqs = []
        for idx in groups:
            ii = shuffle_rng.permutation(idx) if shuffle_rng is not None else idx
            e = embn[ii]
            d = unit(np.diff(e, axis=0))
            if len(d) >= 2:
                seqs.append(((d @ hp.T) >= 0) @ powers)
        return seqs

    def bigram_mi(seqs):
        M = np.zeros((2**b, 2**b))
        for s in seqs:
            np.add.at(M, (s[:-1], s[1:]), 1)
        tot = M.sum()
        if tot == 0:
            return 0.0
        P = M / tot
        pi, pj = P.sum(1, keepdims=True), P.sum(0, keepdims=True)
        nz = P > 0
        return float((P[nz] * np.log2(P[nz] / (pi @ pj)[nz])).sum())

    mi_real = bigram_mi(codes_per_file())
    mi_null = [bigram_mi(codes_per_file(np.random.default_rng(200 + s))) for s in range(NSEED)]
    print(f"  bigram MI real {mi_real:.4f} bits")
    print(f"  bigram MI null {np.mean(mi_null):.4f} +/- {np.std(mi_null):.4f} bits")


if __name__ == "__main__":
    main()
