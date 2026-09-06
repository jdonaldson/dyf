"""Diagnostic: is the flat delta result a property of music, or of SUBTRACTION?

Chroma transitions are a GROUP ACTION (cyclic group C12 on pitch classes), not a
translation. "Up a fourth" from C is chroma(F)-chroma(C); from D it is
chroma(G)-chroma(D). Those are different vectors related by a cyclic rotation,
so plain subtraction entangles the move with its starting position -- exactly
the anchor-dependence failure predicted in SEQUENCE_NOTES.md.

Test: re-express each delta in a ROOT-RELATIVE frame. Roll both the source and
target chroma so the source frame's root sits at index 0, then subtract. Now
"up a fourth" is the same vector from every starting chord.

If a vocabulary appears root-relative but not absolute, the diagnosis is proven:
the moves exist, and subtraction in the wrong frame hides them.
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


def deltas(seqs, root_relative, rng=None):
    ds = []
    for s in seqs:
        e = unit(s)
        if rng is not None:
            e = rng.permutation(e)
        if not root_relative:
            ds.append(np.diff(e, axis=0))
            continue
        # roll each consecutive pair into the source frame's root
        src, tgt = e[:-1], e[1:]
        roots = np.argmax(src, axis=1)
        cols = (np.arange(12)[None, :] + roots[:, None]) % 12
        rs = np.take_along_axis(src, cols, axis=1)
        rt = np.take_along_axis(tgt, cols, axis=1)
        ds.append(rt - rs)
    return np.concatenate(ds, axis=0)


def spectrum(X):
    Xc = X - X.mean(0, keepdims=True)
    s = np.linalg.svd(Xc, compute_uv=False)
    return (s**2) / max(len(Xc) - 1, 1)


def eff_rank(lam):
    lam = lam[lam > 1e-12]
    return (lam.sum() ** 2) / (lam**2).sum()


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
    return H / b, srt[:10].sum() / counts.sum()


def run(tag, seqs, root_relative, mag_filter):
    D = deltas(seqs, root_relative)
    if mag_filter:
        m = np.linalg.norm(D, axis=1)
        D = D[m > np.median(m)]
    Un = unit(D)
    pr = eff_rank(spectrum(Un))
    h6, t6 = bstats(sign_lsh(Un, 6), 6)
    h8, t8 = bstats(sign_lsh(Un, 8), 8)

    hs6, ts6, hs8, ts8, prs = [], [], [], [], []
    for s in range(NSEED):
        Dn = deltas(seqs, root_relative, rng=np.random.default_rng(300 + s))
        if mag_filter:
            mn = np.linalg.norm(Dn, axis=1)
            Dn = Dn[mn > np.median(mn)]
        Unn = unit(Dn)
        prs.append(eff_rank(spectrum(Unn)))
        a, b_ = bstats(sign_lsh(Unn, 6), 6)
        c, d_ = bstats(sign_lsh(Unn, 8), 8)
        hs6.append(a)
        ts6.append(b_)
        hs8.append(c)
        ts8.append(d_)

    print(f"\n{tag}  (n={len(D)})")
    print(f"  PR         real {pr:6.2f}   null {np.mean(prs):6.2f} +/- {np.std(prs):.2f}")
    print(f"  H/b  b=6   real {h6:.4f}   null {np.mean(hs6):.4f} +/- {np.std(hs6):.4f}")
    print(f"  top10 b=6  real {t6:.4f}   null {np.mean(ts6):.4f} +/- {np.std(ts6):.4f}")
    print(f"  H/b  b=8   real {h8:.4f}   null {np.mean(hs8):.4f} +/- {np.std(hs8):.4f}")
    print(f"  top10 b=8  real {t8:.4f}   null {np.mean(ts8):.4f} +/- {np.std(ts8):.4f}")


def main():
    seqs = load_seqs()
    print(f"tracks {len(seqs)}  beats {sum(len(s) for s in seqs)}")
    run("ABSOLUTE frame, all deltas", seqs, False, False)
    run("ROOT-RELATIVE frame, all deltas", seqs, True, False)
    run("ABSOLUTE frame, |d|>median", seqs, False, True)
    run("ROOT-RELATIVE frame, |d|>median", seqs, True, True)


if __name__ == "__main__":
    main()
