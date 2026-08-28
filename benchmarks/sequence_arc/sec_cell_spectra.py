"""Do dyf cells have distinct PCA SPECTRAL SHAPES, and does shape mean anything?

Follow-on to sec_cell_volume.py, which found per-cell log-VOLUME (sum log lambda) flat
and uninformative. Volume is a scalar collapse of the spectrum: a filament, a pancake and
an isotropic blob can share a log-determinant. This probe asks the un-collapsed question:

  (a) Do cells differ in spectral SHAPE beyond what equal-size random subsets differ by?
  (b) Are the shapes organised -- siblings alike, gradients across the map?
  (c) Does shape predict anything -- content composition, retrieval difficulty?

TWO CONTROLS THAT THE QUESTION CANNOT BE ASKED WITHOUT:

1. n-MATCHED SUBSAMPLING. In the n < d regime the sample spectrum spreads as n shrinks
   (Marchenko-Pastur), so raw per-cell spectra encode occupancy, not geometry. Every cell
   is subsampled to a common N_SUB before its spectrum is taken, averaged over draws.
   Consequence worth stating: at min_leaf=16 a LEAF spectrum has rank <= 15 out of 768 --
   effective dimension there is capped by leaf size and cannot be a geometric measure.
   `--leafdemo` shows this directly.

2. RANDOM-SUBSET NULL. Equal-size random draws from the whole corpus have their own
   spectral spread. "Cells differ from each other" is only interesting if they differ MORE
   than random draws of the same size do. Reported as a variance ratio, per descriptor.

Descriptors, all on the n-matched spectrum (p_i = lambda_i / sum lambda):
  eff_rank   exp(-sum p log p)          -- effective dimensionality
  pr         (sum l)^2 / sum l^2        -- participation ratio, same family
  top1       p_1                        -- dominance of the leading direction
  alpha      -OLS slope of log l on log rank  -- power-law decay speed
  aniso      lambda_1 / lambda_2        -- filament vs pancake

Retrieval difficulty is measured directly (recall@10 of IVF probe-32 vs exact kNN, for
queries drawn from each cell) rather than reusing the frozen/fresh gap, so this probe is
self-contained and needs no shift conditions.
"""

import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import sec_seqlib as S  # noqa: E402
from sec_cell_volume import cells_at_depth, flatten_ev, spearman  # noqa: E402

N_SUB = 200  # common sample size for depth 1-2 spectra
N_SUB_DEEP = 50  # smaller, so depth 3 can join the comparison
N_DRAWS = 5  # subsample draws averaged per cell
DEPTHS = (1, 2, 3)
NQ_PER_CELL = 60  # queries per cell; 12 left the per-cell target noise-dominated
PROBE = 32
K = 10
SEED = 42


# ------------------------------------------------------------------------- spectral shape
def spectrum(X, n_sub, rng, n_draws=N_DRAWS):
    """Mean normalised eigenvalue spectrum of X over `n_draws` subsamples of size n_sub.

    Decomposes whichever side is smaller -- Gram (n x n) when n_sub <= d, covariance
    (d x d) otherwise -- since both carry the same nonzero spectrum. Returns a descending
    spectrum of length min(n_sub - 1, d) normalised to sum 1, or None if X is too small.
    """
    if len(X) < n_sub:
        return None
    d = X.shape[1]
    rank = min(n_sub - 1, d)
    acc = None
    for _ in range(n_draws):
        idx = rng.choice(len(X), n_sub, replace=False)
        Z = X[idx].astype(np.float64)
        Z -= Z.mean(0)
        M = (Z @ Z.T) if n_sub <= d else (Z.T @ Z)
        lam = np.linalg.eigvalsh(M / (n_sub - 1))[::-1]
        lam = np.clip(lam[:rank], 1e-14, None)
        lam = lam / lam.sum()
        acc = lam if acc is None else acc + lam
    return acc / n_draws


def descriptors(p):
    """Shape descriptors from a normalised descending spectrum."""
    p = np.clip(np.asarray(p, float), 1e-14, None)
    p = p / p.sum()
    H = -float((p * np.log(p)).sum())
    r = min(50, len(p))
    x, y = np.log(np.arange(1, r + 1)), np.log(p[:r])
    alpha = -float(np.polyfit(x, y, 1)[0])
    return {
        "eff_rank": float(np.exp(H)),
        "pr": float(1.0 / (p**2).sum()),
        "top1": float(p[0]),
        "alpha": alpha,
        "aniso": float(p[0] / p[1]) if len(p) > 1 else np.nan,
    }


DESCS = ["eff_rank", "pr", "top1", "alpha", "aniso"]


# ------------------------------------------------------------------------------- controls
def leafdemo(E):
    """Effective dimension at small n is n, not geometry. Shows why leaves are unusable."""
    rng = np.random.default_rng(SEED)
    print("Effective rank of RANDOM corpus subsets vs sample size n (768 dims available):")
    print(f"{'n':>7}{'eff_rank':>11}{'pr':>9}{'top1':>8}{'rank ceiling':>14}")
    for n in [16, 32, 64, 128, 256, 512, 1024, 4096, 16384]:
        p = spectrum(E, n, rng, n_draws=3)
        d = descriptors(p)
        print(f"{n:>7}{d['eff_rank']:>11.1f}{d['pr']:>9.1f}{d['top1']:>8.3f}{min(n - 1, E.shape[1]):>14}")
    print(
        "\n-> eff_rank tracks n across the whole usable range, so at leaf sizes (16-31)\n"
        "   it IS occupancy. Any 'leaf intrinsic dimension' is capped by min_leaf."
    )


# ----------------------------------------------------------------------------------- main
def main():
    E, D, T, SEC, Q = S.load()
    rng = np.random.default_rng(SEED)
    print(f"corpus {E.shape}  N_SUB={N_SUB}  draws={N_DRAWS}", flush=True)

    t0 = time.time()
    tree_flat = flatten_ev(
        __import__("dyf.dyf_tree", fromlist=["build_dyf_tree"]).build_dyf_tree(
            E, max_depth=S.MAX_DEPTH, num_bits=S.NUM_BITS, min_leaf_size=S.MIN_LEAF, seed=SEED
        ),
        E,
    )
    NL = S.n_leaves(tree_flat)
    print(f"full-corpus tree: {len(tree_flat)} nodes, {NL} leaves [{time.time() - t0:.0f}s]", flush=True)

    out = {}
    for depth in DEPTHS:
        n_sub = N_SUB if depth <= 2 else N_SUB_DEEP
        cells = cells_at_depth(tree_flat, depth)
        rows = []
        for nid, lids in cells:
            idxs = tree_flat[nid]["indices"]
            p = spectrum(E[idxs], n_sub, rng)
            if p is None:
                continue
            r = descriptors(p)
            r["nid"] = int(nid)
            r["n"] = int(len(idxs))
            r["leaf_ids"] = [int(x) for x in lids]
            # content composition
            sec = SEC[idxs]
            vals, cnt = np.unique(sec, return_counts=True)
            r["sec_purity"] = float(cnt.max() / cnt.sum())
            r["sec_top"] = str(vals[cnt.argmax()])
            r["ticker_div"] = float(len(np.unique(T[idxs])) / len(idxs))
            r["date_span_q"] = int(len(np.unique(Q[idxs])))
            # CONTENT AUDIT: a very low effective dimension is what a pile of near-duplicate
            # templated text looks like. Measure that directly instead of inferring geometry.
            sm = E[rng.choice(idxs, min(len(idxs), 400), replace=False)]
            G = sm @ sm.T
            np.fill_diagonal(G, -1.0)
            r["mean_cos"] = float(G[G > -1].mean())
            r["dup_frac"] = float((G.max(1) > 0.99).mean())
            rows.append(r)

        # NULL: one random pool per cell, matched to THAT cell's size, then the same
        # n_sub subsampling. Isolates "is this cell a distinctive region" from "is this
        # how much equal-size random draws wobble".
        null_rows = []
        for r in rows:
            pool = rng.choice(len(E), r["n"], replace=False)
            p = spectrum(E[pool], n_sub, rng)
            null_rows.append(descriptors(p))

        stat = {"n_cells": len(rows), "n_sub": n_sub}
        for dn in DESCS:
            v = np.array([r[dn] for r in rows], float)
            nv = np.array([r[dn] for r in null_rows], float)
            stat[f"{dn}_mean"] = float(np.nanmean(v))
            stat[f"{dn}_sd"] = float(np.nanstd(v))
            stat[f"{dn}_null_sd"] = float(np.nanstd(nv))
            stat[f"{dn}_var_ratio"] = float(np.nanvar(v) / max(np.nanvar(nv), 1e-30))
            stat[f"{dn}_rho_n"] = spearman(v, np.log([r["n"] for r in rows]))
        out[f"depth{depth}"] = stat
        out[f"depth{depth}_rows"] = rows
        print(
            f"depth {depth}: {len(rows)} cells (n_sub={n_sub})  "
            f"eff_rank {stat['eff_rank_mean']:.1f}+-{stat['eff_rank_sd']:.1f} "
            f"(null sd {stat['eff_rank_null_sd']:.2f}, var ratio {stat['eff_rank_var_ratio']:.1f}x)",
            flush=True,
        )

    # ---------------- sibling coherence: are shapes organised, or scattered? ----------
    print("\n--- sibling coherence (depth 2 cells grouped by depth-1 parent) ---", flush=True)
    d2 = out["depth2_rows"]
    parent = {}
    for nid, lids in cells_at_depth(tree_flat, 1):
        for r in d2:
            if set(r["leaf_ids"]) <= set(lids):
                parent[r["nid"]] = nid
    groups = {}
    for r in d2:
        pid = parent.get(r["nid"])
        if pid is not None:
            groups.setdefault(pid, []).append(r)
    sib_stats = {}
    for dn in DESCS:
        within, between = [], []
        gl = [g for g in groups.values() if len(g) >= 3]
        for g in gl:
            v = np.array([r[dn] for r in g], float)
            within.append(np.nanvar(v))
        allv = np.array([r[dn] for r in d2], float)
        between = np.nanvar(allv)
        icc = 1.0 - (np.nanmean(within) / max(between, 1e-30))
        sib_stats[dn] = float(icc)
        print(f"  {dn:<10} variance explained by parent = {icc:+.3f}")
    out["sibling_icc"] = sib_stats
    out["n_sibling_groups"] = len([g for g in groups.values() if len(g) >= 3])

    # ---------------- does shape predict content / difficulty? ----------------------
    print("\n--- retrieval difficulty per depth-2 cell (probe 32 vs exact) ---", flush=True)
    a_leaf = S.fresh_assign(tree_flat, len(E))
    C, _ = S.leaf_centroids(E, a_leaf, NL)
    leaf2cell = np.full(NL, -1, np.int64)
    for ci, r in enumerate(d2):
        for lf in r["leaf_ids"]:
            leaf2cell[lf] = ci

    qidx, qcell = [], []
    for ci, r in enumerate(d2):
        pool = np.where(leaf2cell[a_leaf] == ci)[0]
        if len(pool) < NQ_PER_CELL:
            continue
        pick = rng.choice(pool, NQ_PER_CELL, replace=False)
        qidx.extend(pick.tolist())
        qcell.extend([ci] * NQ_PER_CELL)
    qidx, qcell = np.array(qidx), np.array(qcell)
    print(f"  {len(qidx)} queries over {len(set(qcell.tolist()))} cells", flush=True)

    truth = S.exact_knn(E[qidx], E, k=K)
    got = S.ivf_search(E[qidx], E, a_leaf, C, PROBE, K)
    rec = np.array([len(set(g.tolist()) & set(t.tolist())) / len(t) for g, t in zip(got, truth)])
    for ci in range(len(d2)):
        m = qcell == ci
        d2[ci]["difficulty"] = float(1.0 - rec[m].mean()) if m.any() else None

    us = [r for r in d2 if r.get("difficulty") is not None]
    y = [r["difficulty"] for r in us]
    print(f"\n  Spearman(descriptor, retrieval difficulty), {len(us)} cells:")
    pred = {}
    for dn in DESCS + ["sec_purity", "ticker_div", "date_span_q"]:
        rho = spearman([r[dn] for r in us], y)
        pred[dn] = rho
        print(f"    {dn:<12} {rho:+.3f}")
    rho_n = spearman([np.log(r["n"]) for r in us], y)
    pred["log_n"] = rho_n
    print(f"    {'log_n':<12} {rho_n:+.3f}   <-- the free baseline to beat")
    nullp = float(
        np.percentile(
            [
                abs(
                    spearman(
                        np.arange(len(us), dtype=float), np.random.default_rng(i).permutation(len(us)).astype(float)
                    )
                )
                for i in range(500)
            ],
            95,
        )
    )
    print(f"    [perm null95 |rho|] {nullp:.3f}")
    out["difficulty_rho"] = pred
    out["difficulty_null95"] = nullp

    print("\n  Spearman(descriptor, content composition):")
    for dn in DESCS:
        rp = spearman([r[dn] for r in d2], [r["sec_purity"] for r in d2])
        rt = spearman([r[dn] for r in d2], [r["ticker_div"] for r in d2])
        print(f"    {dn:<10} vs sec_purity {rp:+.3f}   vs ticker_div {rt:+.3f}")

    # ---- is spectral shape just section type? and is the low tail near-duplicates? ----
    def rank_resid(y, x):
        ry = np.argsort(np.argsort(np.asarray(y, float))).astype(float)
        rx = np.argsort(np.argsort(np.asarray(x, float))).astype(float)
        A = np.vstack([rx, np.ones_like(rx)]).T
        return ry - A @ np.linalg.lstsq(A, ry, rcond=None)[0]

    print("\n  eff_rank by dominant section type (is shape just section type?):")
    for st in sorted({r["sec_top"] for r in d2}):
        g = [r for r in d2 if r["sec_top"] == st]
        v = np.array([r["eff_rank"] for r in g], float)
        dv = np.array([r["dup_frac"] for r in g], float)
        mc = np.array([r["mean_cos"] for r in g], float)
        print(
            f"    {st:<17} cells={len(g):>4}  eff_rank {v.mean():>5.1f}+-{v.std():>4.1f}  "
            f"dup_frac {dv.mean():.3f}  mean_cos {mc.mean():+.3f}"
        )
    print(
        f"    rho(eff_rank, dup_frac)={spearman([r['eff_rank'] for r in d2], [r['dup_frac'] for r in d2]):+.3f}"
        f"   rho(eff_rank, mean_cos)={spearman([r['eff_rank'] for r in d2], [r['mean_cos'] for r in d2]):+.3f}"
    )

    print("\n  ROBUSTNESS of eff_rank -> difficulty:")
    ln = [np.log(r["n"]) for r in us]
    er = [r["eff_rank"] for r in us]
    print(f"    raw                          {spearman(er, y):+.3f}")
    print(f"    partial | log n              {spearman(rank_resid(er, ln), rank_resid(y, ln)):+.3f}")
    print(
        f"    partial | dup_frac           "
        f"{spearman(rank_resid(er, [r['dup_frac'] for r in us]), rank_resid(y, [r['dup_frac'] for r in us])):+.3f}"
    )
    no_other = [r for r in us if r["sec_top"] != "other"]
    print(
        f"    excluding sec_top='other'    "
        f"{spearman([r['eff_rank'] for r in no_other], [r['difficulty'] for r in no_other]):+.3f}"
        f"   ({len(no_other)} of {len(us)} cells)"
    )
    out["robustness_n_no_other"] = len(no_other)

    path = os.path.join(S.CACHE, "cell_spectra_results.json")
    with open(path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nsaved -> {path}")


if __name__ == "__main__":
    if "--leafdemo" in sys.argv:
        E, *_ = S.load()
        leafdemo(E)
    else:
        main()
