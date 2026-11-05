"""Is a point's shape profile noise (tree-dependent) or important (data-dependent)?  And do
dyf's shipped bridge notions coincide with the unstable points?

Per dataset: three deep trees (8-way, depth 8; seeds 42/1/2) -> per point, the profile of
its leaf under each seed and its nearest-family label.  Per point:
  stab        mean over seed pairs of the RMS distance in log(1+d) between its profiles over
              shared rungs (>= 2 rungs; NaN otherwise)
  null        the same distance between a point and a RANDOM other point (seed 42 profiles)
  band        reproducible = stab below the null's 10th percentile; noise = stab above the
              null median; ambiguous between
  fam_agree   all three seeds give the same family; chance = sum_f p_f^3 (seed-42 frequencies)
Shipped flags, computed on the flat classifier (num_bits derived from n as find_super_connectors
does; seeds 42 and 1 for flag reproducibility):
  bridge      multi_resolution_analysis(dense_threshold=10): recovery_depth in 1..num_bits
              (sparse at full resolution, recovered coarser); orphan = never recovered
  connector   bridge_persistence(relative_threshold=0.8): persistence >= 1 depth
  boundary    boundary_persistence_scores(seed-42 deep tree, margin_pct=0.10) > 0
Outcomes (seed-42 shipped depth-4 pipeline): leaf purity, re-cut moved.
Report: band shares; purity / moved / bridge / connector / boundary rate by band; AUC of
stab for each flag; flag reproducibility across seeds vs chance; family agreement vs chance;
reproducible share by family.
Env: /Users/jdonaldson/Projects/dyf/.venv/bin/python profile_stability_test.py [names]
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
from dyf_rs import DensityClassifier
from sklearn.metrics import roc_auc_score

from dyf import boundary_persistence_scores, build_dyf_tree
from dyf.agglomerate import _reassign_points, _run_louvain_on_centroids
from dyf.rag import _derive_num_bits

sys.path.insert(0, "/Volumes/Models/dyf_bench_2026-11-04")
from community_conductance_bench import leaf_map, load  # noqa: E402
from dimension_profile_test import unit  # noqa: E402
from leaf_shape_families import FAMS, leaf_profiles, nearest_family  # noqa: E402

OUT = Path("/Volumes/Models/dyf_bench_2026-11-04")
TREE = dict(num_bits=3, max_depth=8, min_leaf_size=20)
SHIPPED = dict(num_bits=3, max_depth=4, min_leaf_size=20)
SEEDS = (42, 1, 2)
ORDER = ["mnist", "fashion_mnist", "cmu_mocap", "ag_news", "tms_facs", "tms_droplet", "brain_1m"]
NR = 6
import os  # noqa: E402

SKIP_PERSISTENCE = os.environ.get("SKIP_PERSISTENCE") == "1"  # bridge_persistence took ~30 min on MoCap


def point_profiles(tree, Xn, n):
    P = np.full((n, NR), np.nan)
    fam = np.full(n, -1, dtype=np.int64)
    for lf in leaf_profiles(tree, Xn):
        prof = (lf["profile"] + [np.nan] * NR)[:NR]
        P[lf["indices"]] = prof
        f, _ = nearest_family(lf["profile"])
        fam[lf["indices"]] = FAMS.index(f) if f is not None else -1
    return P, fam


def pair_dist(A, B):
    """RMS over shared defined rungs of log(1+d); NaN if < 2 shared rungs."""
    D = (np.log1p(A) - np.log1p(B)) ** 2
    ok = ~np.isnan(D)
    cnt = ok.sum(1)
    with np.errstate(invalid="ignore"):
        out = np.sqrt(np.where(ok, D, 0).sum(1) / np.maximum(cnt, 1))
    out[cnt < 2] = np.nan
    return out


def flat_flags(X, seed):
    nb = _derive_num_bits(len(X), 20)
    clf = DensityClassifier(embedding_dim=X.shape[1], num_bits=nb, seed=seed, skip_isolation=True)
    if hasattr(clf, "fit_raw_pca"):
        clf.fit_raw_pca(X)
    else:
        clf.fit(X)
    mra = clf.multi_resolution_analysis(10)
    rd = np.asarray(mra.recovery_depth)
    bridge = (rd >= 1) & (rd <= nb)
    orphan = rd > nb
    if SKIP_PERSISTENCE:
        return bridge, orphan, np.zeros(len(X), dtype=bool), nb
    bp = clf.bridge_persistence(X, 0.8)
    pers = np.asarray(bp.bridge_persistence).astype(float)
    # the 0.8 relative threshold flags ~100% of points at the derived bit count (mnist: 100%),
    # so the flag is the top decile of persistence (depths where the item is a connector)
    connector = pers >= np.percentile(pers, 90)
    return bridge, orphan, connector, nb


def auc(y, s):
    m = ~np.isnan(s)
    if y[m].sum() == 0 or y[m].sum() == m.sum():
        return None
    return float(roc_auc_score(y[m], s[m]))


def run(name):
    t0 = time.perf_counter()
    X, y = load(name)
    n = len(X)
    Xn = unit(X)
    profs, fams, tree42 = {}, {}, None
    for s in SEEDS:
        tree = build_dyf_tree(X, seed=s, **TREE)
        profs[s], fams[s] = point_profiles(tree, Xn, n)
        if s == 42:
            tree42 = tree
    rng = np.random.default_rng(0)
    pairs = [(42, 1), (42, 2), (1, 2)]
    D = np.stack([pair_dist(profs[a], profs[b]) for a, b in pairs])
    stab = np.nanmean(D, axis=0)
    null = pair_dist(profs[42], profs[42][rng.permutation(n)])
    q10, q50 = np.nanpercentile(null, 10), np.nanpercentile(null, 50)
    band = np.full(n, 1, dtype=np.int64)  # 0 reproducible, 1 ambiguous, 2 noise, -1 undefined
    band[stab < q10] = 0
    band[stab > q50] = 2
    band[np.isnan(stab)] = -1
    f42 = fams[42]
    agree = (fams[42] == fams[1]) & (fams[1] == fams[2]) & (f42 >= 0)
    pf = np.bincount(f42[f42 >= 0], minlength=len(FAMS)) / max((f42 >= 0).sum(), 1)
    chance = float((pf**3).sum())
    # shipped pipeline outcomes
    item_leaf = leaf_map(build_dyf_tree(X, seed=42, **SHIPPED), n)
    L = item_leaf.max() + 1
    cent = np.zeros((L, X.shape[1]), dtype=np.float32)
    np.add.at(cent, item_leaf, Xn)
    cent = unit(cent)
    ll, _, _ = _run_louvain_on_centroids(cent, k=10, resolution=1.0, similarity_threshold=0.5)
    block = np.asarray(ll)[item_leaf].astype(np.int32)
    moved = _reassign_points(block.copy(), X) != block
    leaf_pur = np.array([np.bincount(y[item_leaf == l]).max() / (item_leaf == l).sum() for l in range(L)])[item_leaf]
    # flags
    bridge42, orphan42, conn42, nb = flat_flags(X, 42)
    bridge1, orphan1, conn1, _ = flat_flags(X, 1)
    boundary = boundary_persistence_scores(tree42, margin_pct=0.10) > 0
    out = {
        "n": n,
        "num_bits_flat": int(nb),
        "null_q10": float(q10),
        "null_q50": float(q50),
        "stab_median": float(np.nanmedian(stab)),
        "family_agree": float(agree.mean()),
        "family_agree_chance": chance,
        "bands": {},
        "flag_repro": {},
        "auc_stab": {},
        "by_family": {},
    }
    print(
        f"\n== {name}: n={n} flat bits={nb}; cross-seed profile distance median {np.nanmedian(stab):.3f} vs random-pair "
        f"q10 {q10:.3f} / median {q50:.3f}; family agreement (3 seeds) {agree.mean():.1%} vs chance {chance:.1%}",
        flush=True,
    )
    print("   band          share  purity  moved  bridge  orphan  connector  boundary", flush=True)
    for b, lab in ((0, "reproducible"), (1, "ambiguous"), (2, "noise"), (-1, "undefined")):
        m = band == b
        if not m.any():
            continue
        row = dict(
            share=float(m.mean()),
            purity=float(leaf_pur[m].mean()),
            moved=float(moved[m].mean()),
            bridge=float(bridge42[m].mean()),
            orphan=float(orphan42[m].mean()),
            connector=float(conn42[m].mean()),
            boundary=float(boundary[m].mean()),
        )
        out["bands"][lab] = row
        print(
            f"   {lab:12s} {row['share']:6.1%}   {row['purity']:.2f}  {row['moved']:5.1%}  {row['bridge']:5.1%}  {row['orphan']:5.1%}   {row['connector']:5.1%}    {row['boundary']:5.1%}",
            flush=True,
        )
    for fl, a, b_ in (("bridge", bridge42, bridge1), ("orphan", orphan42, orphan1), ("connector", conn42, conn1)):
        p = a.mean()
        ch = p * b_.mean() + (1 - p) * (1 - b_.mean())
        out["flag_repro"][fl] = dict(
            rate=float(p), agree=float((a == b_).mean()), chance=float(ch), auc_stab=auc(a, stab)
        )
        print(
            f"   {fl:9s}: rate {p:5.1%}; seed-42 vs seed-1 agreement {(a == b_).mean():.1%} (chance {ch:.1%}); AUC(flag | stab) {out['flag_repro'][fl]['auc_stab']}",
            flush=True,
        )
    out["auc_stab"]["moved"] = auc(moved, stab)
    print(f"   AUC(moved | stab) {out['auc_stab']['moved']}; AUC(boundary | stab) {auc(boundary, stab)}", flush=True)
    print("   by family (seed 42): share | reproducible share | noise share | purity | moved", flush=True)
    for i, f in enumerate(FAMS):
        m = f42 == i
        if m.sum() < 50:
            continue
        row = dict(
            share=float(m.mean()),
            reproducible=float((band[m] == 0).mean()),
            noise=float((band[m] == 2).mean()),
            purity=float(leaf_pur[m].mean()),
            moved=float(moved[m].mean()),
        )
        out["by_family"][f] = row
        print(
            f"     {f:8s} {row['share']:6.1%} | {row['reproducible']:5.1%} | {row['noise']:5.1%} | {row['purity']:.2f} | {row['moved']:5.1%}",
            flush=True,
        )
    out["seconds"] = time.perf_counter() - t0
    print(f"   ({out['seconds']:.0f}s)", flush=True)
    return out


if __name__ == "__main__":
    names = sys.argv[1:] or ORDER
    out_path = OUT / (
        "profile_stability_light_results.json" if SKIP_PERSISTENCE else "profile_stability_test_results.json"
    )
    results = json.loads(out_path.read_text()) if out_path.exists() else {}
    for name in names:
        results[name] = run(name)
        out_path.write_text(json.dumps(results, indent=1))
    print("\nwrote", out_path)
