"""What is a dyf split actually DOING -- separating modes, or bisecting a blob?

Hypothesis under test (user's framing): a SKEWED spectrum means one dominant feature, so
the split prunes off derivative variation along that single axis; a FLAT spectrum means
either genuine multi-modality or noise.

The framing needs two corrections, both established by measurement:

1. SPECTRA ARE BLIND TO MODALITY. Eigenvalues are second moments. A bimodal cloud and a
   unimodal Gaussian with the same variance along an axis have IDENTICAL spectra, so the
   "multi-modal or noise" branch is not ambiguous-in-practice, it is undecidable from the
   spectrum. Modality has to be tested on the projections themselves, via Ashman's D on a
   2-component GMM fit -- AGAINST THE RIGHT NULL, which took three tries:

     rung 1  1-D standard normal.               D95 ~ 1.7.  Too weak: omits the fact that
             PC1 is CHOSEN as the max-variance direction.
     rung 2  Gaussian cloud with the cell's own covariance, projected on its OWN PC1.
             D95 ~ 1.6-1.8. This is the correct selection-effect null: it shows PCA does
             NOT manufacture modes in a unimodal cloud (agrees with rung 1, so rung 1 was
             coincidentally adequate).
     rung 3  Matched-size RANDOM CORPUS SUBSET on its own PC1. D95 ~ 3.5-3.9, and NO cell
             beats it. But this is THE WRONG QUESTION: it asks "is this cell as multimodal
             as the whole corpus", and the answer is no BY DESIGN because depth 1 already
             separated the section types. Varying the cell contents smuggles in a second
             difference. Recorded here so nobody re-derives it as a negative result.

   The null must vary ONLY the direction, holding the cell fixed: nulls A (random ambient
   direction) and B (rung 2) below. Against those, the split axis wins 11/12 at depth 1
   and 11/12 at depth 2.

2. THERE IS A THIRD FAILURE MODE: THE OFFSET. Routing is `x @ H.T > 0` -- the cut is at
   the ORIGIN, while the hyperplanes are PCA directions fitted on CENTRED data. Nothing
   makes the origin pass through the cell. Measured on a 20k subset: frac>0 per bit =
   0.585 / 0.118 / 0.219 / 0.239, i.e. the mean projection sits ~0.8 sd off the cut on
   PC2-PC4, so those bits are ~80/20 splits. Effective buckets 7.9 of 16. A bit can be
   uninformative because of WHERE it cuts, regardless of its eigenvalue.

So a split axis is characterised by three independent things:
    eigenvalue share   -- is there variance here at all
    offset / balance   -- does the cut land near the centre of mass
    modality           -- does it separate modes or bisect one blob

Modes: `--fits` compares fit_raw_pca / fit / fit_itq on bit balance.
       default runs the per-cell skew-vs-modality analysis.
"""

import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import sec_seqlib as S  # noqa: E402
from sec_cell_volume import cells_at_depth, flatten_ev, spearman  # noqa: E402

NB = 4
SEED = 42
N_FIT = 20000  # subset size for the fit-method comparison
MAX_MODAL = 4000  # cap points per GMM fit, for speed
DEPTHS = (1, 2)


# ------------------------------------------------------------------- balance / offset
def bit_profile(X, H):
    """Per-hyperplane offset and balance. H is (nb, d), rows need not be unit norm."""
    out = []
    for i in range(H.shape[0]):
        hn = H[i] / (np.linalg.norm(H[i]) + 1e-12)
        proj = X @ hn
        sd = float(proj.std()) + 1e-12
        out.append(
            {
                "frac_pos": float((proj > 0).mean()),
                "offset_sd": float(proj.mean() / sd),
                "balance": float(1.0 - abs(2.0 * (proj > 0).mean() - 1.0)),  # 1=even, 0=degenerate
            }
        )
    return out


def eff_buckets(bucket_ids, nb=NB):
    _, cnt = np.unique(bucket_ids, return_counts=True)
    sh = cnt / cnt.sum()
    H = -float((sh * np.log(sh)).sum())
    return float(np.exp(H)), int(len(cnt)), float(sh.max())


def fits():
    """Do the other fit methods centre the cut? Bit balance by fit_method."""
    from dyf_rs import DensityClassifier

    E, *_ = S.load()
    rng = np.random.default_rng(SEED)
    X = E[rng.choice(len(E), N_FIT, replace=False)]
    d = X.shape[1]

    print(f"fit-method comparison on a {N_FIT}-point subset, num_bits={NB}")
    print(f"{'method':<14}{'eff_buckets':>13}{'occupied':>10}{'max_share':>11}   per-bit frac>0")
    for name in ["fit_raw_pca", "fit", "fit_itq"]:
        c = DensityClassifier(embedding_dim=d, num_bits=NB, seed=SEED)
        fn = getattr(c, name, None)
        if fn is None:
            print(f"{name:<14}  (unavailable in this dyf-rs build)")
            continue
        try:
            fn(X)
        except Exception as e:  # noqa: BLE001
            print(f"{name:<14}  FAILED: {type(e).__name__}: {e}")
            continue
        b = np.asarray(c.get_bucket_ids())
        Hm = np.asarray(c.get_hyperplanes(), dtype=np.float64).reshape(NB, d)
        eb, occ, mx = eff_buckets(b)
        prof = bit_profile(X, Hm)
        print(f"{name:<14}{eb:>13.2f}{occ:>10}{mx:>11.3f}   " + " ".join(f"{p['frac_pos']:.3f}" for p in prof))
    print(f"\n  ideal: eff_buckets {1 << NB}, max_share {1 / (1 << NB):.3f}, every frac>0 = 0.500")


# ------------------------------------------------------------------------------ modality
def ashman(x):
    """Ashman's D on a 2-component 1-D GMM. D > ~2 indicates a genuine mode separation."""
    from sklearn.mixture import GaussianMixture

    v = np.asarray(x, float).reshape(-1, 1)
    v = (v - v.mean()) / (v.std() + 1e-12)
    g = GaussianMixture(2, random_state=0, n_init=2).fit(v)
    m = np.asarray(g.means_).ravel()
    s = np.sqrt(np.asarray(g.covariances_).ravel())
    return float(abs(m[0] - m[1]) * np.sqrt(2.0) / np.sqrt(s[0] ** 2 + s[1] ** 2))


def _own_pc1(Z):
    """Project Z onto its own top principal direction (Gram side; n <= d assumed small)."""
    _, V = np.linalg.eigh(Z @ Z.T)
    v = Z.T @ V[:, -1]
    return Z @ (v / (np.linalg.norm(v) + 1e-12))


def split_modality(Z, hn, rng, n_null=8):
    """Does direction `hn` separate modes in centred cell data `Z`?

    Both nulls hold the CELL fixed and vary only the direction -- see the rung-3 warning
    in the module docstring for why a corpus-subset null answers a different question.
      A: random ambient unit direction (is PC1 special, or is the cell split every way?)
      B: Gaussian cloud with this cell's covariance, on its own PC1 (selection effect)
    """
    obs = ashman(Z @ hn)
    n = len(Z)
    ra = []
    for _ in range(n_null):
        u = rng.standard_normal(Z.shape[1])
        ra.append(ashman(Z @ (u / np.linalg.norm(u))))
    rb = []
    for _ in range(n_null):
        A = rng.standard_normal((n, n)) / np.sqrt(n)
        rb.append(ashman(_own_pc1(A @ Z)))
    qa, qb = float(np.percentile(ra, 95)), float(np.percentile(rb, 95))
    return {
        "ashman_d": obs,
        "null_randdir95": qa,
        "null_gausspc1_95": qb,
        "bimodal": bool(obs > qa and obs > qb),
    }


def main():
    E, D, T, SEC, Q = S.load()
    rng = np.random.default_rng(SEED)
    from dyf.dyf_tree import build_dyf_tree

    tree = build_dyf_tree(E, max_depth=S.MAX_DEPTH, num_bits=S.NUM_BITS, min_leaf_size=S.MIN_LEAF, seed=SEED)
    flat = flatten_ev(tree, E)
    print(f"tree: {len(flat)} nodes, {S.n_leaves(flat)} leaves", flush=True)

    out = {}
    for depth in DEPTHS:
        rows = []
        for nid, _ in cells_at_depth(flat, depth):
            node = flat[nid]
            if node["hp"] is None or node["n"] < 500:
                continue
            Xc = E[node["indices"]]
            Hm = np.asarray(node["hp"], dtype=np.float64).reshape(-1, Xc.shape[1])
            # spectral skew from the cell's own top-4 (what dyf stores), n-independent share
            Z = Xc[rng.choice(len(Xc), min(len(Xc), 2000), replace=False)].astype(np.float64)
            Z -= Z.mean(0)
            lam = np.linalg.eigvalsh((Z @ Z.T) / (len(Z) - 1))[::-1]
            lam = np.clip(lam, 1e-14, None)
            share = lam / lam.sum()
            prof = bit_profile(Xc, Hm)
            r = {
                "nid": int(nid),
                "n": int(node["n"]),
                "top1_share": float(share[0]),
                "skew_1_2": float(share[0] / share[1]),
                "eff_rank": float(np.exp(-(share * np.log(share)).sum())),
                "min_balance": float(min(p["balance"] for p in prof)),
                "mean_balance": float(np.mean([p["balance"] for p in prof])),
                "max_abs_offset": float(max(abs(p["offset_sd"]) for p in prof)),
            }
            Zc = Xc.astype(np.float64)
            if len(Zc) > MAX_MODAL:
                Zc = Zc[rng.choice(len(Zc), MAX_MODAL, replace=False)]
            Zc = Zc - Zc.mean(0)
            mods = []
            for i in range(Hm.shape[0]):
                hn = Hm[i] / (np.linalg.norm(Hm[i]) + 1e-12)
                mods.append(split_modality(Zc, hn, rng))
            r["n_bimodal"] = int(sum(m["bimodal"] for m in mods))
            r["axis1_bimodal"] = bool(mods[0]["bimodal"])
            r["axis1_ashman"] = float(mods[0]["ashman_d"])
            r["axis1_null_randdir95"] = float(mods[0]["null_randdir95"])
            r["axis1_null_gausspc1_95"] = float(mods[0]["null_gausspc1_95"])
            r["per_axis_bimodal"] = [bool(m["bimodal"]) for m in mods]
            r["per_axis_ashman"] = [float(m["ashman_d"]) for m in mods]
            r["per_axis_frac_pos"] = [float(p["frac_pos"]) for p in prof]
            rows.append(r)
            print(
                f"  d{depth} nid={nid:<6} n={r['n']:<7} top1={r['top1_share']:.3f} "
                f"eff_rank={r['eff_rank']:>5.1f} bimodal={r['n_bimodal']}/{Hm.shape[0]} "
                f"balance={r['mean_balance']:.2f} maxoff={r['max_abs_offset']:.2f}",
                flush=True,
            )
        out[f"depth{depth}"] = rows

    print("\n" + "=" * 74)
    print("Does spectral skew predict what the split does?")
    print("=" * 74)
    for depth in DEPTHS:
        rows = out[f"depth{depth}"]
        if len(rows) < 5:
            print(f"depth {depth}: only {len(rows)} cells, skipping correlations")
            continue
        nb = np.array([r["n_bimodal"] for r in rows], float)
        print(f"\ndepth {depth}, {len(rows)} cells")
        print(
            f"  bimodal axes per cell: mean {nb.mean():.2f} of 4   "
            f"distribution {np.bincount(nb.astype(int), minlength=5).tolist()}"
        )
        print(f"  cells where axis 1 (PC1) is bimodal: {sum(r['axis1_bimodal'] for r in rows)} of {len(rows)}")
        for k in ["top1_share", "skew_1_2", "eff_rank"]:
            print(
                f"  rho({k:<11}, n_bimodal) = {spearman([r[k] for r in rows], nb):+.3f}   "
                f"rho({k:<11}, mean_balance) = {spearman([r[k] for r in rows], [r['mean_balance'] for r in rows]):+.3f}"
            )
        print(f"  rho(mean_balance, n_bimodal) = {spearman([r['mean_balance'] for r in rows], nb):+.3f}")
        fp = np.array([f for r in rows for f in r["per_axis_frac_pos"]])
        print(
            f"  per-axis frac>0 across all cells: mean {fp.mean():.3f}, "
            f"{100 * (np.abs(fp - 0.5) > 0.3).mean():.0f}% are worse than 80/20"
        )

    path = os.path.join(S.CACHE, "split_anatomy_results.json")
    with open(path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nsaved -> {path}")


if __name__ == "__main__":
    if "--fits" in sys.argv:
        fits()
    else:
        main()
