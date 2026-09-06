"""Does the INTERVAL structure survive deflating the entropy axis?

chroma_audit.py showed buckets concentrating on real intervals (P4 x7.1, P5 x5.4)
but also a strong d_entropy correlation (r=0.79) that, once deflated, removed the
real-vs-null significance. Bucket CONTENT was measured with the entropy axis
still intact, so it may have been riding on it -- exactly the Haxe failure.

Re-hash after deflation and re-measure bucket -> interval enrichment. If P4/P5
buckets persist, the harmonic vocabulary is independent of the confound.
"""

import os

import numpy as np

CACHE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "chroma_beats.npz")
BITS = 6
NAMES = {
    0: "unison",
    1: "m2",
    2: "M2",
    3: "m3",
    4: "M3",
    5: "P4",
    6: "tritone",
    7: "P5",
    8: "m6",
    9: "M6",
    10: "m7",
    11: "M7",
}


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


def ent(X):
    P = np.clip(X, 0, None)
    P = P / np.clip(P.sum(1, keepdims=True), 1e-9, None)
    return -(P * np.log2(np.clip(P, 1e-12, None))).sum(1)


def rootrel(seqs):
    Ds, Iv, dH = [], [], []
    for s in seqs:
        e = unit(s)
        src, tgt = e[:-1], e[1:]
        roots = np.argmax(src, axis=1)
        cols = (np.arange(12)[None, :] + roots[:, None]) % 12
        rs = np.take_along_axis(src, cols, axis=1)
        rt = np.take_along_axis(tgt, cols, axis=1)
        Ds.append(rt - rs)
        Iv.append(np.argmax(rt, axis=1))
        dH.append(ent(rt) - ent(rs))
    return np.concatenate(Ds, 0), np.concatenate(Iv), np.concatenate(dH)


def codes_of(X, b):
    Xu = unit(X)
    Xc = Xu - Xu.mean(0, keepdims=True)
    _, _, vt = np.linalg.svd(Xc, full_matrices=False)
    hp = vt[:b]
    return (((Xu @ hp.T) >= 0) @ (2 ** np.arange(hp.shape[0]))).astype(np.int64)


def mi_bits(codes, iv):
    """Mutual information between bucket id and interval, in bits."""
    cu, ci = np.unique(codes, return_inverse=True)
    M = np.zeros((len(cu), 12))
    np.add.at(M, (ci, iv), 1)
    P = M / M.sum()
    pi, pj = P.sum(1, keepdims=True), P.sum(0, keepdims=True)
    nz = P > 0
    return float((P[nz] * np.log2(P[nz] / (pi @ pj)[nz])).sum())


def report(tag, D, IV, base):
    cd = codes_of(D, BITS)
    counts = np.bincount(cd, minlength=2**BITS)
    print(f"\n=== {tag} ===   MI(bucket; interval) = {mi_bits(cd, IV):.3f} bits")
    for b in np.argsort(counts)[::-1][:5]:
        m = cd == b
        if m.sum() < 20:
            continue
        di = np.bincount(IV[m], minlength=12) / m.sum()
        top = np.argsort(di)[::-1][:2]
        s = ", ".join(f"{NAMES[i]} {di[i]:.2f} (x{di[i] / max(base[i], 1e-9):.1f})" for i in top)
        print(f"  bucket {b:2d} n={int(m.sum()):4d}  | {s}")


def main():
    seqs = load_seqs()
    D, IV, DH = rootrel(seqs)
    mags = np.linalg.norm(D, axis=1)
    lo, hi = np.quantile(mags, 0.65), np.quantile(mags, 0.95)
    sel = np.flatnonzero((mags >= lo) & (mags < hi))
    Db, IVb, DHb = D[sel], IV[sel], DH[sel]
    base = np.bincount(IVb, minlength=12) / len(IVb)
    print(f"n={len(sel)}  baseline: " + "  ".join(f"{NAMES[i]}:{base[i]:.2f}" for i in np.argsort(base)[::-1][:5]))

    y = (DHb - DHb.mean()) / (DHb.std() + 1e-9)
    w = Db.T @ y / len(y)
    w /= np.linalg.norm(w)

    report("entropy axis INTACT", Db, IVb, base)
    report("entropy axis REMOVED", Db - np.outer(Db @ w, w), IVb, base)

    # how much of the interval information did the entropy axis alone carry?
    proj = unit(Db) @ w
    qs = np.quantile(proj, np.linspace(0, 1, 9))
    binned = np.clip(np.searchsorted(qs, proj) - 1, 0, 7)
    print(f"\nMI(entropy-axis octile; interval) = {mi_bits(binned, IVb):.3f} bits (vs bucket MI above)")


if __name__ == "__main__":
    main()
