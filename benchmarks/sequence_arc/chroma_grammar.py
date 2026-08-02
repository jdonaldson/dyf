"""Isolate GRAMMAR from walk-consistency.

The sequence-permutation null preserved each track's interval multiset, but a
permuted interval sequence generally does not correspond to any actual chord
walk. Real intervals chain through shared roots -- interval_i ends on the root
that interval_{i+1} starts from -- so consecutive intervals are dependent even
with zero harmonic grammar. That mechanical dependency inflates the lag-1 excess.

Proper null: resample each track's ROOT sequence i.i.d. from that track's own
empirical root distribution, then derive intervals. This preserves the chord
inventory, its frequencies, AND walk-consistency (it is a genuine walk), while
destroying any ordering preference. Real-vs-this isolates grammar.

Reported alongside the weaker permutation null so the two are comparable.
"""

import os

import numpy as np

CACHE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "chroma_beats.npz")
NSEED = 8


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


def mi(a, b, card):
    if len(a) < 30:
        return np.nan
    M = np.zeros((card, card))
    np.add.at(M, (a, b), 1)
    P = M / M.sum()
    pi, pj = P.sum(1, keepdims=True), P.sum(0, keepdims=True)
    nz = P > 0
    return float((P[nz] * np.log2(P[nz] / (pi @ pj)[nz])).sum())


def lag_mi(seqs, k, card=12):
    a, b = [], []
    for s in seqs:
        if len(s) > k:
            a.append(s[:-k])
            b.append(s[k:])
    if not a:
        return np.nan
    return mi(np.concatenate(a), np.concatenate(b), card)


def intervals_from_roots(roots, collapse):
    iv = (roots[1:] - roots[:-1]) % 12
    return iv[iv != 0] if collapse else iv


def main():
    seqs = load_seqs()
    roots = [np.argmax(unit(s), axis=1) for s in seqs]

    for collapse in (False, True):
        tag = "unisons collapsed (progression)" if collapse else "all beats"
        real = [intervals_from_roots(r, collapse) for r in roots]
        real = [r for r in real if len(r) > 5]
        n = sum(len(r) for r in real)
        print(f"\n=== INTERVAL symbols, {tag} ===  n={n}")
        print(
            f"{'lag':>4} {'MI real':>9} | {'perm null (multiset)':>22} {'z':>7} | "
            f"{'WALK null (inventory)':>22} {'z':>7}"
        )

        for k in (1, 2, 3):
            r = lag_mi(real, k)

            perm = []
            for s in range(NSEED):
                rg = np.random.default_rng(1000 + s)
                perm.append(lag_mi([rg.permutation(x) for x in real], k))
            pm, ps = np.nanmean(perm), np.nanstd(perm)

            walk = []
            for s in range(NSEED):
                rg = np.random.default_rng(2000 + s)
                w = []
                for rt in roots:
                    rr = rg.choice(rt, size=len(rt), replace=True)  # i.i.d. from track marginal
                    iv = intervals_from_roots(rr, collapse)
                    if len(iv) > 5:
                        w.append(iv)
                walk.append(lag_mi(w, k))
            wm, ws = np.nanmean(walk), np.nanstd(walk)

            zp = (r - pm) / ps if ps > 0 else np.nan
            zw = (r - wm) / ws if ws > 0 else np.nan
            print(f"{k:>4} {r:9.4f} | {pm:12.4f} +/-{ps:.4f} {zp:+7.2f} | {wm:12.4f} +/-{ws:.4f} {zw:+7.2f}")

    # what does the grammar look like? top transitions vs walk expectation
    print("\n=== most over-represented interval bigrams (progression) ===")
    real = [intervals_from_roots(r, True) for r in roots]
    real = [r for r in real if len(r) > 5]
    A = np.concatenate([x[:-1] for x in real])
    B = np.concatenate([x[1:] for x in real])
    M = np.zeros((12, 12))
    np.add.at(M, (A, B), 1)
    P = M / M.sum()
    exp = P.sum(1, keepdims=True) @ P.sum(0, keepdims=True)
    with np.errstate(divide="ignore", invalid="ignore"):
        lift = np.where(exp > 0, P / exp, 0)
    NAMES = {
        0: "uni",
        1: "m2",
        2: "M2",
        3: "m3",
        4: "M3",
        5: "P4",
        6: "TT",
        7: "P5",
        8: "m6",
        9: "M6",
        10: "m7",
        11: "M7",
    }
    flat = [(lift[i, j], M[i, j], i, j) for i in range(12) for j in range(12) if M[i, j] >= 25]
    for lf, cnt, i, j in sorted(flat, reverse=True)[:8]:
        print(f"  {NAMES[i]:>3} -> {NAMES[j]:<3}  n={int(cnt):4d}  lift x{lf:.2f}")


if __name__ == "__main__":
    main()
