"""Is "shared density" of a pair real information, or distance in disguise?

The idea under test: dyf measures density PER POINT (bucket occupancy, centroid similarity,
isolation). A pair (x, y) also has a *shared* density -- how populated is the region between
them. And the partner can be the TEMPORAL predecessor rather than the geometric neighbour,
which turns per-point density into transition density without asking the index to be
order-aware. That last part matters: `SEQUENCE_NOTES.md` closed the sequence arc because dyf
is verified permutation-invariant, but this computes over the index rather than inside it.

THE CONFOUND THAT WOULD MAKE THIS TRIVIAL. Two points that are close together sit in a
similar region and share neighbours; two far-apart points do not. So "shared density predicts
relatedness" is nearly guaranteed by distance alone, and a trial-boundary detection task on
CMU MoCap is worse still -- boundaries there are jumps between unrelated recordings that raw
step size detects for free.

So the question is asked at MATCHED DISTANCE: within a narrow band of pair distance, does
shared density separate same-motion pairs from different-motion pairs? Distance is constant
inside a band by construction, so any separation is information distance does not carry.

ABLATIONS, because the competitor is never "nothing":
  distance        AUC ~0.5 inside a band by construction -- the sanity check
  endpoint_mean   mean of the two points' OWN densities: what dyf gives today. Shared
                  density must beat this or it is a costlier route to an existing number.
  midpoint        count of corpus points within r of (x+y)/2
  segment_min     min density sampled along the segment -- catches a thin isthmus that the
                  midpoint alone would miss
  lca_depth       tree depth at which the two points' routing diverges (nearly free)

Also reported: consecutive-frame pairs (the "neighbour that came previously") within a trial
versus across a trial boundary, to see whether the temporal partner behaves differently from
a geometric one.

Exact counts on a subsample are used rather than an index estimate, so index approximation is
not a confound. In production the same quantity is a dyf density query at the midpoint.
"""

import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import sec_seqlib as S  # noqa: E402

PAPER = os.path.expanduser("~/Projects/semantic-proprioception-paper/experiments/data")
N_SUB = 40000
N_PAIRS = 6000
K_REF = 10  # radius = median distance to the K_REF-th neighbour
SEG_STEPS = 5
DIST_BANDS = 6
SEED = 42


def auc(scores, labels):
    """Rank AUC of `scores` predicting label==1. 0.5 = no information."""
    s = np.asarray(scores, float)
    y = np.asarray(labels).astype(bool)
    if y.all() or not y.any():
        return float("nan")
    r = np.argsort(np.argsort(s)).astype(float)
    n1, n0 = int(y.sum()), int((~y).sum())
    return float((r[y].sum() - n1 * (n1 - 1) / 2) / (n1 * n0))


def density_at(Q, X, radius, chunk=8192):
    """Exact count of X within `radius` of each row of Q."""
    out = np.zeros(len(Q), dtype=np.int64)
    r2 = radius * radius
    for c in range(0, len(X), chunk):
        Xc = X[c : c + chunk]
        d2 = (Q * Q).sum(1)[:, None] + (Xc * Xc).sum(1)[None, :] - 2.0 * (Q @ Xc.T)
        out += (d2 <= r2).sum(1)
    return out


def main():
    import polars as pl

    rng = np.random.default_rng(SEED)
    X = np.load(os.path.join(PAPER, "cmu_mocap_features.npy")).astype(np.float32)
    meta = pl.read_parquet(os.path.join(PAPER, "cmu_mocap_metadata.parquet"))
    assert len(X) == len(meta)

    keep = np.sort(rng.choice(len(X), min(N_SUB, len(X)), replace=False))
    Xs = np.ascontiguousarray(X[keep])
    motion = meta["motion_type"].to_numpy()[keep]
    trial = meta["trial_id"].to_numpy()[keep]
    print(f"subsample {Xs.shape}, {len(set(motion.tolist()))} motion types, {len(set(trial.tolist()))} trials")

    # radius from the local scale of the data
    probe = Xs[rng.choice(len(Xs), 1500, replace=False)]
    d2 = (probe * probe).sum(1)[:, None] + (Xs * Xs).sum(1)[None, :] - 2.0 * (probe @ Xs.T)
    np.maximum(d2, 0, out=d2)
    kth = np.sqrt(np.partition(d2, K_REF, axis=1)[:, K_REF])
    radius = float(np.median(kth))
    print(f"radius = median {K_REF}-NN distance = {radius:.3f}")

    # ---- pairs, stratified later by their distance ---------------------------------
    ia = rng.choice(len(Xs), N_PAIRS, replace=False)
    ib = rng.choice(len(Xs), N_PAIRS, replace=False)
    ok = ia != ib
    ia, ib = ia[ok], ib[ok]
    A, B = Xs[ia], Xs[ib]
    dist = np.linalg.norm(A - B, axis=1)
    same_motion = motion[ia] == motion[ib]
    print(f"{len(ia)} pairs, {100 * same_motion.mean():.1f}% same motion_type")

    mid = (A + B) / 2.0
    dens_mid = density_at(mid, Xs, radius)
    dens_a = density_at(A, Xs, radius)
    dens_b = density_at(B, Xs, radius)
    seg = np.stack([density_at(A + (B - A) * t, Xs, radius) for t in np.linspace(0.2, 0.8, SEG_STEPS)])
    seg_min = seg.min(0)
    endpoint_mean = (dens_a + dens_b) / 2.0

    feats = {
        "distance(neg)": -dist,
        "endpoint_mean": endpoint_mean,
        "midpoint": dens_mid.astype(float),
        "segment_min": seg_min.astype(float),
        "midpoint/endpoint": dens_mid / np.maximum(endpoint_mean, 1e-9),
    }

    print("\n" + "=" * 78)
    print("AUC for 'same motion_type', WITHIN matched-distance bands")
    print("=" * 78)
    edges = np.quantile(dist, np.linspace(0, 1, DIST_BANDS + 1))
    names = list(feats)
    print(f"{'dist band':<18}{'pairs':>7}{'%same':>7}" + "".join(f"{n:>19}" for n in names))
    rows = []
    for i in range(DIST_BANDS):
        m = (dist >= edges[i]) & (dist < edges[i + 1] if i < DIST_BANDS - 1 else dist <= edges[i + 1])
        if m.sum() < 100 or same_motion[m].all() or not same_motion[m].any():
            continue
        line = f"{f'{edges[i]:.1f}-{edges[i + 1]:.1f}':<18}{m.sum():>7}{100 * same_motion[m].mean():>6.0f}%"
        rec = {"band": f"{edges[i]:.1f}-{edges[i + 1]:.1f}", "n": int(m.sum())}
        for n in names:
            a = auc(feats[n][m], same_motion[m])
            rec[n] = a
            line += f"{a:>19.3f}"
        rows.append(rec)
        print(line)

    print("\npooled (distance NOT controlled -- shows how much of it is just distance):")
    line = f"{'all pairs':<18}{len(dist):>7}{100 * same_motion.mean():>6.0f}%"
    for n in names:
        line += f"{auc(feats[n], same_motion):>19.3f}"
    print(line)

    # ---- the temporal partner: lag sweep, with boundaries DELIBERATELY included -----
    # Two failures in the first version of this section, both worth recording:
    #  1. Random sampling of adjacent pairs found 3 boundaries in 4000 (there are only 79
    #     in 140,837 frames), so the AUC was computed on n=3. Boundaries are now enumerated.
    #  2. At lag 1 the step (3.8) is far below the density radius (34.4), so the midpoint
    #     lies inside both endpoints' neighbourhoods BY CONSTRUCTION and mid/endpoint is
    #     exactly 1.000. Shared density is degenerate unless the step is comparable to the
    #     density scale -- hence the lag sweep.
    print("\n" + "=" * 78)
    print("The 'neighbour that came previously': lag sweep, all boundaries enumerated")
    print("=" * 78)
    tid = meta["trial_id"].to_numpy()
    print(f"density radius = {radius:.1f}; lag-1 median step = 3.8, so lag must grow to compete")
    print(
        f"{'lag':>5}{'bnd':>6}{'within':>8}{'step(w)':>9}{'step(b)':>9}{'AUC step':>10}{'AUC -mid':>10}{'AUC -m/e':>10}"
    )
    temporal = []
    for lag in (1, 5, 20, 60, 150):
        b = np.flatnonzero(tid[:-lag] != tid[lag:])  # every boundary at this lag
        w_all = np.flatnonzero(tid[:-lag] == tid[lag:])
        if len(b) < 10:
            continue
        w = rng.choice(w_all, min(len(w_all), 20 * len(b)), replace=False)
        prev = np.concatenate([b, w])
        lab_bnd = np.concatenate([np.ones(len(b), bool), np.zeros(len(w), bool)])
        Ap = X[prev].astype(np.float32)
        Bp = X[prev + lag].astype(np.float32)
        dstep = np.linalg.norm(Ap - Bp, axis=1)
        dmid = density_at((Ap + Bp) / 2.0, Xs, radius).astype(float)
        dend = (density_at(Ap, Xs, radius) + density_at(Bp, Xs, radius)) / 2.0
        ratio = dmid / np.maximum(dend, 1e-9)
        a_step, a_mid, a_ratio = auc(dstep, lab_bnd), auc(-dmid, lab_bnd), auc(-ratio, lab_bnd)
        temporal.append({"lag": lag, "n_bnd": int(len(b)), "auc_step": a_step, "auc_mid": a_mid, "auc_ratio": a_ratio})
        print(
            f"{lag:>5}{len(b):>6}{len(w):>8}{np.median(dstep[~lab_bnd]):>9.1f}"
            f"{np.median(dstep[lab_bnd]):>9.1f}{a_step:>10.3f}{a_mid:>10.3f}{a_ratio:>10.3f}"
        )
    print("  step distance is the free baseline; shared density must beat it to matter.")

    path = os.path.join(S.CACHE, "shared_density_results.json")
    with open(path, "w") as f:
        json.dump({"radius": radius, "bands": rows, "temporal": temporal}, f, indent=2)
    print(f"\nsaved -> {path}")


if __name__ == "__main__":
    main()
