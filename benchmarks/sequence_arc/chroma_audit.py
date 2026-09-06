"""Artifact audit on the chroma move-buckets -- the same audit that dissolved Haxe.

Haxe looked significant (z -15.5) until its top buckets turned out to encode
identifier length; deflating one axis killed it. Chroma has had no such audit.

Two parts:

  (A) CONTENT. In the root-relative frame the source root sits at index 0, so
      the target's argmax IS the transition interval in semitones. If buckets
      carry harmony they should concentrate on recognizable intervals (5=fourth,
      7=fifth, 0=same root) rather than reproducing the baseline distribution.

  (B) CONFOUND. Per-frame loudness is already gone (chroma frames are unit
      normalized), so the analogue of Haxe's "identifier length" is frame
      PEAKINESS: a clear chord is a low-entropy chroma frame, a percussive or
      transitional beat is diffuse. If consecutive beats systematically run
      tonal->diffuse, the delta direction encodes that, not harmony.
      Test: correlate delta against its best d(entropy) axis, then deflate it
      and re-run real-vs-null with a null-vs-null floor.
"""

import os

import numpy as np

CACHE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "chroma_beats.npz")
NSEED, BITS, MIN_N = 6, 6, 200


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
    """Shannon entropy (bits) of each chroma frame treated as a distribution."""
    P = np.clip(X, 0, None)
    P = P / np.clip(P.sum(1, keepdims=True), 1e-9, None)
    return -(P * np.log2(np.clip(P, 1e-12, None))).sum(1)


def rootrel(seqs, rng=None):
    """Root-relative consecutive pairs. Returns delta, interval, d_entropy."""
    Ds, Iv, dH = [], [], []
    for s in seqs:
        e = unit(s)
        if rng is not None:
            e = rng.permutation(e)
        src, tgt = e[:-1], e[1:]
        roots = np.argmax(src, axis=1)
        cols = (np.arange(12)[None, :] + roots[:, None]) % 12
        rs = np.take_along_axis(src, cols, axis=1)
        rt = np.take_along_axis(tgt, cols, axis=1)
        Ds.append(rt - rs)
        Iv.append(np.argmax(rt, axis=1))
        dH.append(ent(rt) - ent(rs))
    return (np.concatenate(Ds, 0), np.concatenate(Iv), np.concatenate(dH))


def hnorm(X, b):
    Xu = unit(X)
    Xc = Xu - Xu.mean(0, keepdims=True)
    _, _, vt = np.linalg.svd(Xc, full_matrices=False)
    hp = vt[:b]
    codes = (((Xu @ hp.T) >= 0) @ (2 ** np.arange(hp.shape[0]))).astype(np.int64)
    c = np.bincount(codes, minlength=2**b).astype(float)
    p = c[c > 0] / c.sum()
    return float(-(p * np.log2(p)).sum() / b)


def codes_of(X, b):
    Xu = unit(X)
    Xc = Xu - Xu.mean(0, keepdims=True)
    _, _, vt = np.linalg.svd(Xc, full_matrices=False)
    hp = vt[:b]
    return (((Xu @ hp.T) >= 0) @ (2 ** np.arange(hp.shape[0]))).astype(np.int64)


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


def main():
    seqs = load_seqs()
    D, IV, DH = rootrel(seqs)
    mags = np.linalg.norm(D, axis=1)
    lo, hi = np.quantile(mags, 0.65), np.quantile(mags, 0.95)
    band = (mags >= lo) & (mags < hi)
    sel = np.flatnonzero(band)
    print(f"deltas {len(D)} | audit band |d| {lo:.3f}-{hi:.3f} -> n={len(sel)}")

    # ---------- (A) content ----------
    Db, IVb, DHb, Mb = D[sel], IV[sel], DH[sel], mags[sel]
    cd = codes_of(Db, BITS)
    counts = np.bincount(cd, minlength=2**BITS)
    base = np.bincount(IVb, minlength=12) / len(IVb)
    print("\nbaseline interval distribution:")
    print("  " + "  ".join(f"{NAMES[i]}:{base[i]:.2f}" for i in np.argsort(base)[::-1][:6]))
    print(f"baseline d_entropy {DHb.mean():+.3f}   mean |d| {Mb.mean():.3f}")

    print("\ntop buckets:")
    for b in np.argsort(counts)[::-1][:6]:
        m = cd == b
        if m.sum() < 20:
            continue
        di = np.bincount(IVb[m], minlength=12) / m.sum()
        top = np.argsort(di)[::-1][:3]
        lift = ", ".join(f"{NAMES[i]} {di[i]:.2f} (x{di[i] / max(base[i], 1e-9):.1f})" for i in top)
        print(f"  bucket {b:2d} n={int(m.sum()):4d}  dH {DHb[m].mean():+.3f}  |d| {Mb[m].mean():.3f}  | {lift}")

    # ---------- (B) confound ----------
    y = (DHb - DHb.mean()) / (DHb.std() + 1e-9)
    w = Db.T @ y / len(y)
    w /= np.linalg.norm(w)
    r = np.corrcoef(unit(Db) @ w, DHb)[0, 1]
    print(f"\ncorr(delta on best d_entropy axis, d_entropy) = {r:.3f}")

    nulls = [rootrel(seqs, np.random.default_rng(800 + s)) for s in range(NSEED)]
    rng = np.random.default_rng(17)
    qs = np.quantile(mags, [0.5, 0.7, 0.85, 0.95])
    edges = [(qs[0], qs[1]), (qs[1], qs[2]), (qs[2], qs[3])]

    for defl in (False, True):
        sets = [D] + [d for d, _, _ in nulls]
        if defl:
            sets = [X - np.outer(X @ w, w) for X in sets]
        ms = [np.linalg.norm(X, axis=1) for X in sets]
        rows = []
        for lo2, hi2 in edges:
            idxs = [np.flatnonzero((m >= lo2) & (m < hi2)) for m in ms]
            n = min(len(i) for i in idxs)
            if n < MIN_N:
                continue
            hs = [hnorm(X[rng.choice(i, n, replace=False)], BITS) for X, i in zip(sets, idxs)]
            rows.append((n, hs))
        H = np.array([h for _, h in rows])
        wt = np.array([n for n, _ in rows], float)
        wt /= wt.sum()
        real, nl = H[:, 0], H[:, 1:]
        d_real = float(np.dot(wt, real - nl.mean(1)))
        dn = [float(np.dot(wt, nl[:, k] - np.delete(nl, k, 1).mean(1))) for k in range(nl.shape[1])]
        mu, sd = float(np.mean(dn)), float(np.std(dn))
        print(
            f"  entropy axis {'REMOVED':>8}" if defl else f"  entropy axis {'intact':>8}",
            f" cells {len(rows)}  real-vs-null {d_real:+.4f}  "
            f"floor {mu:+.4f} +/- {sd:.4f}  z {(d_real - mu) / sd:+.2f}",
        )


if __name__ == "__main__":
    main()
