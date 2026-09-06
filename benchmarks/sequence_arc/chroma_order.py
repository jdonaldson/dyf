"""Does ORDER carry information beyond the track's transition inventory?

Every previous null shuffled BEATS, which changes which transitions exist and so
conflates "a vocabulary exists" with "order matters". This null permutes the
TRANSITION SEQUENCE within each track: the multiset of moves is preserved
exactly, per track, and only their arrangement is destroyed. Any real-vs-null
gap is therefore pure ordering.

Two symbol alphabets:
  interval  the transition interval in semitones (12 symbols) -- interpretable,
            and low-cardinality enough to estimate bigram MI from ~5k samples
  bucket    the b=6 sign-LSH code (64 symbols) -- what dyf would actually give
            you; comparing the two says whether the hash PRESERVES the
            sequential information the interval carries

Two sequence definitions:
  all       every beat transition (dominated by unisons = sustained chords)
  collapsed unisons dropped -- the actual chord progression / harmonic rhythm

MI is measured at lags 1..4. A genuine progression grammar (ii-V-I) should show
elevated MI at lag 1 AND decaying structure at lag 2-3. The shared-endpoint
telescoping artifact (d_i and d_{i+1} share a frame) can only inflate lag 1, so
the lag profile separates grammar from artifact.
"""

import os

import numpy as np

CACHE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "chroma_beats.npz")
NSEED, BITS = 8, 6


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


def per_track(seqs):
    """Root-relative deltas, intervals and d_entropy, kept PER TRACK in order."""
    out = []
    for s in seqs:
        e = unit(s)
        src, tgt = e[:-1], e[1:]
        roots = np.argmax(src, axis=1)
        cols = (np.arange(12)[None, :] + roots[:, None]) % 12
        rs = np.take_along_axis(src, cols, axis=1)
        rt = np.take_along_axis(tgt, cols, axis=1)
        out.append(
            {
                "D": rt - rs,
                "iv": np.argmax(rt, axis=1),
                "dH": ent(rt) - ent(rs),
            }
        )
    return out


def mi(pairs_a, pairs_b, na, nb):
    if len(pairs_a) < 30:
        return np.nan
    M = np.zeros((na, nb))
    np.add.at(M, (pairs_a, pairs_b), 1)
    P = M / M.sum()
    pi, pj = P.sum(1, keepdims=True), P.sum(0, keepdims=True)
    nz = P > 0
    return float((P[nz] * np.log2(P[nz] / (pi @ pj)[nz])).sum())


def lag_mi(seq_list, k, card):
    a, b = [], []
    for s in seq_list:
        if len(s) > k:
            a.append(s[:-k])
            b.append(s[k:])
    if not a:
        return np.nan
    return mi(np.concatenate(a), np.concatenate(b), card, card)


def run(tag, seq_list, card, rng):
    print(f"\n=== {tag} ===  tracks {len(seq_list)}  symbols {sum(len(s) for s in seq_list)}  alphabet {card}")
    print(f"{'lag':>4} {'MI real':>9} {'MI null':>18} {'excess':>9} {'z':>7}")
    for k in (1, 2, 3, 4):
        r = lag_mi(seq_list, k, card)
        nulls = []
        for s in range(NSEED):
            rg = np.random.default_rng(900 + s)
            perm = [rg.permutation(x) for x in seq_list]
            nulls.append(lag_mi(perm, k, card))
        nulls = np.array(nulls, float)
        mu, sd = np.nanmean(nulls), np.nanstd(nulls)
        z = (r - mu) / sd if sd > 0 else np.nan
        print(f"{k:>4} {r:9.4f} {mu:9.4f} +/-{sd:.4f} {r - mu:+9.4f} {z:+7.2f}")


def main():
    seqs = load_seqs()
    T = per_track(seqs)
    rng = np.random.default_rng(23)

    # shared LSH basis fit on all deltas, entropy axis deflated (audit showed it
    # is a confound that carries only 0.27 bits of interval info)
    Dall = np.concatenate([t["D"] for t in T], 0)
    dH = np.concatenate([t["dH"] for t in T])
    y = (dH - dH.mean()) / (dH.std() + 1e-9)
    w = Dall.T @ y / len(y)
    w /= np.linalg.norm(w)
    Dd = Dall - np.outer(Dall @ w, w)
    Du = unit(Dd)
    Dc = Du - Du.mean(0, keepdims=True)
    _, _, vt = np.linalg.svd(Dc, full_matrices=False)
    hp = vt[:BITS]

    off = 0
    for t in T:
        n = len(t["D"])
        d = Du[off : off + n]
        t["code"] = (((d @ hp.T) >= 0) @ (2 ** np.arange(BITS))).astype(np.int64)
        off += n

    iv_all = [t["iv"] for t in T]
    cd_all = [t["code"] for t in T]
    keep = [t["iv"] != 0 for t in T]
    iv_col = [t["iv"][m] for t, m in zip(T, keep)]
    cd_col = [t["code"][m] for t, m in zip(T, keep)]

    frac = np.mean(np.concatenate([t["iv"] for t in T]) == 0)
    print(f"unison (sustained-chord) fraction: {frac:.2f}")

    run("INTERVAL symbols, all beats", iv_all, 12, rng)
    run("INTERVAL symbols, unisons collapsed (progression)", iv_col, 12, rng)
    run("BUCKET symbols (b=6, entropy-deflated), all beats", cd_all, 2**BITS, rng)
    run("BUCKET symbols, unisons collapsed", cd_col, 2**BITS, rng)


if __name__ == "__main__":
    main()
