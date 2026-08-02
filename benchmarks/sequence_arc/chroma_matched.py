"""Magnitude-matched comparison — removes the SNR confound.

Real (adjacent) deltas are small: cos(adjacent)=0.938. Null (shuffled) deltas
pair distant chords, so they are large. Direction of a SMALL difference between
two similar vectors is noise-dominated and spreads across buckets; a LARGE
difference is signal-dominated and concentrates. Filtering each condition at its
OWN median therefore compares them at different absolute magnitudes, which can
manufacture the entire entropy gap.

Fix: apply ONE absolute threshold to both, and additionally compare inside
matched magnitude bands. The question becomes the right one:

    given two chroma frames THIS different, does temporal adjacency
    make the transition direction any more predictable?
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
        src, tgt = e[:-1], e[1:]
        if root_relative:
            roots = np.argmax(src, axis=1)
            cols = (np.arange(12)[None, :] + roots[:, None]) % 12
            src = np.take_along_axis(src, cols, axis=1)
            tgt = np.take_along_axis(tgt, cols, axis=1)
        ds.append(tgt - src)
    return np.concatenate(ds, axis=0)


def sign_lsh(X, b):
    Xc = X - X.mean(0, keepdims=True)
    _, _, vt = np.linalg.svd(Xc, full_matrices=False)
    hp = vt[:b]
    return (((X @ hp.T) >= 0) @ (2 ** np.arange(hp.shape[0]))).astype(np.int64)


def hnorm(X, b):
    if len(X) < 50:
        return np.nan
    counts = np.bincount(sign_lsh(unit(X), b), minlength=2**b).astype(float)
    p = counts[counts > 0] / counts.sum()
    return float(-(p * np.log2(p)).sum() / b)


def band_compare(seqs, root_relative, b=6):
    D = deltas(seqs, root_relative)
    m = np.linalg.norm(D, axis=1)
    Dn = [deltas(seqs, root_relative, np.random.default_rng(400 + s)) for s in range(NSEED)]
    mn = [np.linalg.norm(d, axis=1) for d in Dn]

    # bands defined on the REAL magnitude distribution, applied to both
    qs = np.quantile(m, [0.5, 0.7, 0.85, 0.95])
    edges = [(qs[0], qs[1]), (qs[1], qs[2]), (qs[2], qs[3]), (qs[3], m.max())]

    frame = "ROOT-RELATIVE" if root_relative else "ABSOLUTE"
    print(f"\n=== {frame} frame, magnitude-matched bands (b={b}) ===")
    print(f"{'band |d|':>16} {'n_real':>7} {'H/b real':>9} {'n_null':>7} {'H/b null':>16} {'delta':>8}")
    for lo, hi in edges:
        sel = (m >= lo) & (m < hi)
        hr = hnorm(D[sel], b)
        hs, ns = [], []
        for d, mm in zip(Dn, mn):
            s2 = (mm >= lo) & (mm < hi)
            ns.append(int(s2.sum()))
            hs.append(hnorm(d[s2], b))
        hs = np.array(hs, dtype=float)
        print(
            f"{lo:6.3f}-{hi:6.3f}  {int(sel.sum()):7d} {hr:9.4f} {int(np.mean(ns)):7d} "
            f"{np.nanmean(hs):9.4f} +/-{np.nanstd(hs):.4f} {hr - np.nanmean(hs):+8.4f}"
        )


def main():
    seqs = load_seqs()
    print(f"tracks {len(seqs)}  beats {sum(len(s) for s in seqs)}")
    for rr in (False, True):
        band_compare(seqs, rr, b=6)


if __name__ == "__main__":
    main()
