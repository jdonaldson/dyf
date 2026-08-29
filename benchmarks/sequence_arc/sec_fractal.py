"""Is the structure dyf recovers actually FRACTAL? Proper measurements, not proxies.

The spectral work in this arc used `alpha`, an OLS slope of log(eigenvalue) on log(rank).
That is not evidence of a power law -- fitting a slope to anything yields a slope. And the
depth profiles were computed at MATCHED sample size (n=300) to kill a sampling artifact,
which also deliberately removed the scale dependence that fractality is about. So none of
it actually tested self-similarity. This does, three ways:

1. CORRELATION DIMENSION (Grassberger-Procaccia). C(r) = fraction of point pairs closer
   than r; if C(r) ~ r^D2 then D2 is the correlation dimension. The measurement that
   matters is not the number but whether the local slope d log C / d log r PLATEAUS over a
   range of r -- a real fractal has a scaling window. Reported with the width of the window
   in decades, so "no plateau" is a visible outcome rather than a hidden one.

2. POWER LAW vs EXPONENTIAL on the spectrum. A power-law spectrum (lambda_i ~ i^-a) is
   scale-free; an exponential one (lambda_i ~ e^-bi) has a characteristic scale. Fit both,
   compare R^2 on the same points. This is the test `alpha` never had.

3. GENERALISED DIMENSIONS D_q. Box-counting is hopeless in 768d, so the fixed-radius
   correlation form is used: p_i(r) = fraction of points within r of point i,
   Z_q(r) = mean(p_i^(q-1)), D_q = (1/(q-1)) d log Z_q / d log r. D_q constant in q means
   monofractal; D_q decreasing in q means MULTIfractal -- different regions carrying
   different local dimensions. eff_rank ranged 9.1-85.3 across depth-2 cells and was
   regionally organised (ICC 0.466), which is what a multifractal would look like.

Also measures D2 INSIDE depth-1 and depth-2 cells, which is the scale-invariance test the
n-matched depth profile could not do: if D2 is the same at the corpus level and within
cells, the structure repeats across tree levels.

Corpora chosen because the depth profiles already disagree: SEC alpha is flat with depth
(0.874 -> 0.834) while CMU MoCap steepens monotonically (2.548 -> 3.780). If that means
anything, SEC should show a scaling window and MoCap should not.
"""

import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import sec_seqlib as S  # noqa: E402
from sec_cell_volume import cells_at_depth, flatten_ev  # noqa: E402

PAPER = os.path.expanduser("~/Projects/semantic-proprioception-paper/experiments/data")
N_PAIR = 6000  # points per correlation-integral estimate
N_R = 40  # radii sampled log-spaced
QS = (1.5, 2.0, 3.0, 4.0, 5.0)
MIN_PAIR_FRAC = 1e-4  # ignore radii with too few pairs to be stable
SEED = 42


def corr_curve(X, rng, n=N_PAIR):
    """Return (radii, C(r), per-point neighbour fractions). True Euclidean, so this works
    on RAW (non-unit-norm) vectors as well as normalised ones."""
    if len(X) > n:
        X = X[rng.choice(len(X), n, replace=False)]
    X = X.astype(np.float32)
    sq = (X * X).sum(1)
    Dm = sq[:, None] + sq[None, :] - 2.0 * (X @ X.T)
    np.maximum(Dm, 0.0, out=Dm)
    Dm = np.sqrt(Dm)
    iu = np.triu_indices(len(X), k=1)
    d = Dm[iu]
    lo, hi = np.percentile(d, [0.5, 99.5])
    if not np.isfinite(lo) or lo <= 0:
        lo = max(d[d > 0].min(), 1e-6)
    radii = np.geomspace(lo, hi, N_R)
    C = np.array([(d < r).mean() for r in radii])
    P = np.array([(Dm < r).mean(1) for r in radii])  # (N_R, n) per-point mass
    return radii, C, P


def local_slope(x, y):
    """Centred finite-difference slope of log y vs log x."""
    lx, ly = np.log(x), np.log(np.maximum(y, 1e-300))
    s = np.full(len(x), np.nan)
    s[1:-1] = (ly[2:] - ly[:-2]) / (lx[2:] - lx[:-2])
    return s


def scaling_window(radii, C, slopes):
    """Widest run of radii where the local slope is flat; returns (D2, decades, lo, hi)."""
    ok = np.isfinite(slopes) & (C > MIN_PAIR_FRAC) & (C < 0.99)
    idx = np.where(ok)[0]
    if len(idx) < 5:
        return float("nan"), 0.0, float("nan"), float("nan")
    best = None
    for i in range(len(idx)):
        for j in range(i + 4, len(idx)):
            seg = slopes[idx[i] : idx[j] + 1]
            seg = seg[np.isfinite(seg)]
            if len(seg) < 5:
                continue
            spread = float(seg.max() - seg.min())
            dec = float(np.log10(radii[idx[j]] / radii[idx[i]]))
            # prefer wide windows that stay flat: score = decades penalised by spread
            score = dec - 2.0 * spread
            if best is None or score > best[0]:
                best = (score, float(np.mean(seg)), dec, radii[idx[i]], radii[idx[j]], spread)
    if best is None:
        return float("nan"), 0.0, float("nan"), float("nan")
    return best[1], best[2], best[3], best[4]


def fit_compare(spec):
    """Power law (log-log) vs exponential (log-linear) on a spectrum. Returns R^2 each."""
    y = np.asarray(spec, float)
    y = y[y > 0]
    k = min(60, len(y))
    y = y[:k]
    i = np.arange(1, k + 1)
    ly = np.log(y)

    def r2(x, yy):
        A = np.vstack([x, np.ones_like(x)]).T
        b = np.linalg.lstsq(A, yy, rcond=None)[0]
        pred = A @ b
        ss = ((yy - pred) ** 2).sum()
        tot = ((yy - yy.mean()) ** 2).sum()
        return float(1 - ss / max(tot, 1e-30)), float(-b[0])

    r2_pl, a_pl = r2(np.log(i), ly)  # power law
    r2_ex, b_ex = r2(i.astype(float), ly)  # exponential
    return {"r2_powerlaw": r2_pl, "alpha": a_pl, "r2_exponential": r2_ex, "beta": b_ex}


def spectrum_of(X, rng, n_sub=300, draws=4):
    if len(X) < n_sub:
        return None
    acc = None
    for _ in range(draws):
        Z = X[rng.choice(len(X), n_sub, replace=False)].astype(np.float64)
        Z -= Z.mean(0)
        lam = np.linalg.eigvalsh((Z @ Z.T) / (n_sub - 1))[::-1]
        lam = np.clip(lam[: n_sub - 1], 1e-14, None)
        lam /= lam.sum()
        acc = lam if acc is None else acc + lam
    return acc / draws


def analyse(X, name, out, rng, tree_cells=None):
    print(f"\n=== {name}: {X.shape} ===", flush=True)
    t0 = time.time()
    radii, C, P = corr_curve(X, rng)
    sl = local_slope(radii, C)
    D2, dec, rlo, rhi = scaling_window(radii, C, sl)
    print(f"  correlation dimension D2 = {D2:.2f} over {dec:.2f} decades  [{time.time() - t0:.0f}s]")
    print(f"  {'r':>9}{'C(r)':>11}{'local slope':>13}")
    for i in range(0, len(radii), max(1, len(radii) // 10)):
        print(f"  {radii[i]:>9.4f}{C[i]:>11.5f}{sl[i]:>13.2f}")

    # generalised dimensions
    dq = {}
    for q in QS:
        Zq = np.array([np.mean(np.maximum(P[i], 1e-12) ** (q - 1.0)) for i in range(len(radii))])
        s = local_slope(radii, Zq) / (q - 1.0)
        ok = np.isfinite(s) & (C > MIN_PAIR_FRAC) & (C < 0.99)
        dq[q] = float(np.nanmean(s[ok])) if ok.sum() >= 3 else float("nan")
    print("  generalised dimensions: " + "  ".join(f"D{q:g}={dq[q]:.2f}" for q in QS))
    spread = max(dq.values()) - min(dq.values())
    print(f"  D_q spread = {spread:.2f}  ({'MULTIfractal' if spread > 0.5 else 'monofractal-ish'})")

    sp = spectrum_of(X, rng)
    fit = fit_compare(sp) if sp is not None else {}
    if fit:
        verdict = "power law" if fit["r2_powerlaw"] > fit["r2_exponential"] else "EXPONENTIAL"
        print(
            f"  spectrum fit: R2 powerlaw={fit['r2_powerlaw']:.3f} (alpha={fit['alpha']:.2f})  "
            f"R2 exponential={fit['r2_exponential']:.3f} (beta={fit['beta']:.3f})  -> {verdict}"
        )

    rec = {
        "D2": D2,
        "decades": dec,
        "r_lo": rlo,
        "r_hi": rhi,
        "dq": {str(k): v for k, v in dq.items()},
        "dq_spread": float(spread),
        **fit,
    }

    # D2 inside cells -- the scale-invariance test the n-matched profile could not do
    if tree_cells:
        for depth, cells in tree_cells.items():
            vals = []
            for nid, idx in cells[:12]:
                if len(idx) < 1500:
                    continue
                r2_, c2, _ = corr_curve(X[idx], rng, n=min(len(idx), 3000))
                s2 = local_slope(r2_, c2)
                d2c, dc, _, _ = scaling_window(r2_, c2, s2)
                if np.isfinite(d2c):
                    vals.append((d2c, dc))
            if vals:
                a = np.array([v[0] for v in vals])
                print(
                    f"  D2 within depth-{depth} cells: {a.mean():.2f} +- {a.std():.2f} "
                    f"(n={len(vals)}, mean window {np.mean([v[1] for v in vals]):.2f} decades)"
                )
                rec[f"D2_depth{depth}"] = {"mean": float(a.mean()), "sd": float(a.std()), "n": len(vals)}
    out[name] = rec


def main():
    rng = np.random.default_rng(SEED)
    out = {}

    E, *_ = S.load()
    from dyf.dyf_tree import build_dyf_tree

    flat = flatten_ev(
        build_dyf_tree(E, max_depth=S.MAX_DEPTH, num_bits=S.NUM_BITS, min_leaf_size=S.MIN_LEAF, seed=SEED), E
    )
    cells = {d: [(nid, flat[nid]["indices"]) for nid, _ in cells_at_depth(flat, d)] for d in (1, 2)}
    analyse(E, "sec_768", out, rng, tree_cells=cells)

    for nm, f in [("cmu_mocap_62", "cmu_mocap_features.npy"), ("wikipedia_384", "wikipedia_embeddings.npy")]:
        p = os.path.join(PAPER, f)
        if not os.path.exists(p):
            print(f"SKIP {nm}")
            continue
        X = np.load(p).astype(np.float32)
        if len(X) > 100000:
            X = X[rng.choice(len(X), 100000, replace=False)]
        X = np.ascontiguousarray(X)
        X /= np.linalg.norm(X, axis=1, keepdims=True) + 1e-12
        analyse(X, nm, out, rng)

    print("\n" + "=" * 82)
    print("SUMMARY")
    print("=" * 82)
    print(f"{'corpus':<16}{'D2':>7}{'window':>9}{'Dq spread':>11}{'R2 power':>10}{'R2 exp':>9}   verdict")
    for k, v in out.items():
        verd = "power law" if v.get("r2_powerlaw", 0) > v.get("r2_exponential", 1) else "exponential"
        mf = "multifractal" if v["dq_spread"] > 0.5 else "monofractal"
        print(
            f"{k:<16}{v['D2']:>7.2f}{v['decades']:>8.2f}d{v['dq_spread']:>11.2f}"
            f"{v.get('r2_powerlaw', float('nan')):>10.3f}{v.get('r2_exponential', float('nan')):>9.3f}   {verd}, {mf}"
        )

    path = os.path.join(S.CACHE, "fractal_results.json")
    with open(path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nsaved -> {path}")


def raw_vs_normalised():
    """Does dropping unit-normalisation recover the dynamic range a scaling window needs?

    The normalised run produced windows of 0.01-0.11 decades, and unit vectors bound all
    pairwise distances to [0, 2] -- under one decade of range even in principle, where a
    credible fractal claim wants two or more. Only corpora whose SOURCE vectors are not
    already unit-norm can test this; the MiniLM text sets and the SEC filings are
    normalised upstream (norm sd = 0.000) so nothing can be recovered there.
    """
    rng = np.random.default_rng(SEED)
    sets = [
        ("cmu_mocap_62", "cmu_mocap_features.npy"),  # norm 113-430, ratio 3.81 -- best case
        ("wikipedia_nomic_768", "wikipedia_nomic_embeddings.npy"),  # ratio 1.27
        ("arxiv_nomic_768", "arxiv_nomic_embeddings.npy"),  # ratio 1.22
        ("news_nomic_768", "news_nomic_embeddings.npy"),  # ratio 1.17
    ]
    print(f"{'corpus':<22}{'mode':>7}{'norm ratio':>12}{'dist range':>12}{'D2':>8}{'window':>9}{'Dq spread':>11}")
    out = {}
    for nm, f in sets:
        p = os.path.join(PAPER, f)
        if not os.path.exists(p):
            print(f"{nm:<22}  (missing)")
            continue
        X0 = np.load(p).astype(np.float32)
        if len(X0) > 100000:
            X0 = X0[rng.choice(len(X0), 100000, replace=False)]
        X0 = np.ascontiguousarray(X0)
        nrm = np.linalg.norm(X0, axis=1)
        ratio = float(nrm.max() / max(nrm.min(), 1e-9))
        for mode in ("raw", "normed"):
            X = X0 if mode == "raw" else X0 / (nrm[:, None] + 1e-12)
            radii, C, P = corr_curve(X, rng)
            sl = local_slope(radii, C)
            D2, dec, _, _ = scaling_window(radii, C, sl)
            dyn = float(np.log10(radii[-1] / max(radii[0], 1e-12)))
            dq = {}
            for q in QS:
                Zq = np.array([np.mean(np.maximum(P[i], 1e-12) ** (q - 1.0)) for i in range(len(radii))])
                s = local_slope(radii, Zq) / (q - 1.0)
                ok = np.isfinite(s) & (C > MIN_PAIR_FRAC) & (C < 0.99)
                dq[q] = float(np.nanmean(s[ok])) if ok.sum() >= 3 else float("nan")
            spread = max(dq.values()) - min(dq.values())
            print(f"{nm:<22}{mode:>7}{ratio:>12.2f}{dyn:>11.2f}d{D2:>8.2f}{dec:>8.2f}d{spread:>11.2f}")
            out[f"{nm}:{mode}"] = {
                "norm_ratio": ratio,
                "dist_range_decades": dyn,
                "D2": D2,
                "window_decades": dec,
                "dq_spread": float(spread),
            }
    print(
        "\n'dist range' is the log10 span of measured pairwise distances -- the ceiling on any\n"
        "scaling window. If raw does not widen it, unit-normalisation was not the binding limit."
    )
    with open(os.path.join(S.CACHE, "fractal_raw_vs_normed.json"), "w") as fh:
        json.dump(out, fh, indent=2)


if __name__ == "__main__":
    if "--raw" in sys.argv:
        raw_vs_normalised()
    else:
        main()
