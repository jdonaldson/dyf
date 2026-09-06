"""Move-vocabulary probe on beat-synchronous chroma (Spotify previews).

Same metrics and same null as delta_probe.py (Haxe), so the numbers are
directly comparable.

Null: shuffle beat order WITHIN each track. Preserves the chroma distribution,
track grouping and track lengths; breaks only temporal adjacency.

Addition vs the Haxe probe: a magnitude split. Sign-LSH is scale-invariant,
which is a LIABILITY here -- during a sustained chord the delta is ~0 and hashes
by its noise direction, spreading uniformly and diluting real transitions. So we
report all deltas and, separately, deltas above the median magnitude (i.e. the
actual chord changes).
"""

import os

import numpy as np

CACHE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "chroma_beats.npz")
NSEED = 5


def unit(x, axis=-1):
    n = np.linalg.norm(x, axis=axis, keepdims=True)
    return x / np.where(n > 0, n, 1)


def load_seqs():
    z = np.load(CACHE, allow_pickle=True)
    lens, stacked = z["lengths"], z["stacked"]
    out, off = [], 0
    for L in lens:
        out.append(stacked[off : off + L])
        off += L
    return out


def deltas_from(seqs, rng=None):
    ds = []
    for s in seqs:
        e = unit(s)
        if rng is not None:
            e = rng.permutation(e)
        ds.append(np.diff(e, axis=0))
    return np.concatenate(ds, axis=0)


def spectrum(X):
    Xc = X - X.mean(0, keepdims=True)
    s = np.linalg.svd(Xc, compute_uv=False)
    return (s**2) / max(len(Xc) - 1, 1)


def eff_rank(lam):
    lam = lam[lam > 1e-12]
    pr = (lam.sum() ** 2) / (lam**2).sum()
    c = np.cumsum(lam) / lam.sum()
    return pr, int(np.searchsorted(c, 0.90) + 1)


def sign_lsh(X, b):
    Xc = X - X.mean(0, keepdims=True)
    _, _, vt = np.linalg.svd(Xc, full_matrices=False)
    hp = vt[:b]
    return (((X @ hp.T) >= 0) @ (2 ** np.arange(hp.shape[0]))).astype(np.int64)


def bstats(codes, b):
    counts = np.bincount(codes, minlength=2**b).astype(float)
    p = counts[counts > 0] / counts.sum()
    H = -(p * np.log2(p)).sum()
    srt = np.sort(counts)[::-1]
    return H / b, int((counts > 0).sum()), srt[:10].sum() / counts.sum()


def report(tag, D, b_list):
    Un = unit(D)
    pr, k90 = eff_rank(spectrum(Un))
    print(f"  {tag:<16} n={len(D):>6}  PR {pr:6.2f}  k90 {k90:3d}", end="")
    for b in b_list:
        h, occ, t10 = bstats(sign_lsh(Un, b), b)
        print(f"   | b={b}: H/b {h:.4f} occ {occ:4d} top10 {t10:.3f}", end="")
    print()
    return pr


def main():
    seqs = load_seqs()
    lens = np.array([len(s) for s in seqs])
    dim = seqs[0].shape[1]
    print(f"tracks {len(seqs)} | beats/track median {np.median(lens):.0f} total {lens.sum()} | dim {dim}")

    D_real = deltas_from(seqs)
    dn = np.linalg.norm(D_real, axis=1)
    cos_adj = 1 - (dn**2) / 2
    print("\n== consecutive-beat geometry ==")
    print(f"  ||d||/||e||   median {np.median(dn):.3f}  mean {dn.mean():.3f}")
    print(f"  cos(adjacent) median {np.median(cos_adj):.3f}  mean {cos_adj.mean():.3f}")
    print(f"  ||mean(d)||   {np.linalg.norm(D_real.mean(0)):.4f}")

    D_null = [deltas_from(seqs, np.random.default_rng(100 + s)) for s in range(NSEED)]

    B = [4, 6, 8]
    states = np.concatenate([unit(s) for s in seqs], axis=0)

    print("\n== ALL deltas ==")
    report("states", states, B)
    report("delta real", D_real, B)
    prs = [report(f"delta null[{i}]", d, B) for i, d in enumerate(D_null)]
    print(f"  {'null PR mean':<16} {np.mean(prs):6.2f} +/- {np.std(prs):.2f}")

    # magnitude split: real chord changes only
    thr = np.median(dn)
    print(f"\n== deltas above median magnitude (thr={thr:.3f}) ==")
    report("delta real>med", D_real[dn > thr], B)
    prs2 = []
    for i, d in enumerate(D_null):
        m = np.linalg.norm(d, axis=1)
        prs2.append(report(f"delta null>med[{i}]", d[m > np.median(m)], B))
    print(f"  {'null PR mean':<16} {np.mean(prs2):6.2f} +/- {np.std(prs2):.2f}")

    # ---- bigram MI over move codes -----------------------------------------
    print("\n== move-code transition structure (b=6) ==")
    b = 6
    Un = unit(D_real)
    Xc = Un - Un.mean(0, keepdims=True)
    _, _, vt = np.linalg.svd(Xc, full_matrices=False)
    hp = vt[:b]
    powers = 2 ** np.arange(hp.shape[0])

    def codes(rng=None):
        out = []
        for s in seqs:
            e = unit(s)
            if rng is not None:
                e = rng.permutation(e)
            d = unit(np.diff(e, axis=0))
            if len(d) >= 2:
                out.append((((d @ hp.T) >= 0) @ powers).astype(np.int64))
        return out

    def mi(cs):
        M = np.zeros((2**b, 2**b))
        for s in cs:
            np.add.at(M, (s[:-1], s[1:]), 1)
        if M.sum() == 0:
            return 0.0
        P = M / M.sum()
        pi, pj = P.sum(1, keepdims=True), P.sum(0, keepdims=True)
        nz = P > 0
        return float((P[nz] * np.log2(P[nz] / (pi @ pj)[nz])).sum())

    m_real = mi(codes())
    m_null = [mi(codes(np.random.default_rng(200 + s))) for s in range(NSEED)]
    print(f"  bigram MI real {m_real:.4f} bits")
    print(f"  bigram MI null {np.mean(m_null):.4f} +/- {np.std(m_null):.4f} bits")


if __name__ == "__main__":
    main()
