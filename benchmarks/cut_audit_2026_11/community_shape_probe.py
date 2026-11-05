"""Shape per COMMUNITY instead of per leaf.

Leaf ladders start at the root and carry the root's cuts (brain's hub = three top-level pieces
with one shape).  The community stage is where the pipeline re-merges pieces, and communities
are what dyfviz shows.  So: treat each shipped community as its own dataset, fit a deep tree on
its points alone, and read the dimension profile / family / pocket share from that ladder.

Per dataset (seed 42):
  shipped     depth-4 8-way tree -> leaf-centroid Louvain -> _reassign_points -> final labels
  within      per community with >= MIN_COMM points: binary tree (num_bits=1, max_depth=14)
              on the community's points; per-leaf profiles (MIN_NODE mask, clip 60) as in
              leaf_shape_families.leaf_profiles; a binary synthetic dictionary (regenerated
              with the same ladder config, cached) gives each leaf a nearest family.
  global      incumbent read-out: the same binary ladder fit ONCE on the whole dataset
              (max_depth=16), leaves restricted to the community's points.
  free        incumbent numbers that need no tree: log size, radius (1 - mean cosine to the
              unit centroid), kNN-15 conductance phi.
Per community, both shape arms give: mean d, min d, slope, dip share (points in leaves with a
rung < 0.5), spike share (a rung >= 20), family shares, pocket (= blobs) share, mean reliable
rungs.  Targets fixed beforehand: purity (majority-class share), moved-in share (final != block
among members), phi.
Scores:
  1. pooled Spearman (ranks taken within dataset, communities >= MIN_COMM) of each feature vs
     each target, with a within-dataset permutation null (500 draws);
  2. leave-one-dataset-out logistic AUC for "bottom-quartile purity within its dataset" from
     FREE / GLOBAL / WITHIN / WITHIN+FREE feature sets.
Also printed: the largest communities per dataset with both profiles side by side, and (brain,
mnist) how many communities each large class occupies -- does the community stage reassemble
what the root split?
Env: /Users/jdonaldson/Projects/dyf/.venv/bin/python community_shape_probe.py [names]
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
from scipy.stats import rankdata, spearmanr
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from dyf import build_dyf_tree
from dyf.agglomerate import _reassign_points, _run_louvain_on_centroids

sys.path.insert(0, "/Volumes/Models/dyf_bench_2026-11-04")
import dimension_profile_test as dpt  # noqa: E402
from community_conductance_bench import leaf_map, load  # noqa: E402
from dimension_profile_test import unit  # noqa: E402
from leaf_shape_families import FAMS, leaf_profiles  # noqa: E402

OUT = Path("/Volumes/Models/dyf_bench_2026-11-04")
SHIPPED = dict(num_bits=3, max_depth=4, min_leaf_size=20)
WITHIN = dict(num_bits=1, max_depth=14, min_leaf_size=20)
GLOBAL = dict(num_bits=1, max_depth=16, min_leaf_size=20)
ORDER = ["mnist", "fashion_mnist", "cmu_mocap", "ag_news", "tms_facs", "tms_droplet", "brain_1m"]
MIN_COMM = 1000
NRUNG = 8  # rungs compared against the binary dictionary
SEED = 42
SHAPE_FEATS = [
    "mean_d",
    "min_d",
    "slope",
    "dip_share",
    "spike_share",
    "pocket_share",
    "curve_sheet_share",
    "ball_share",
    "rungs",
]
FREE_FEATS = ["log_size", "radius", "phi"]
TARGETS = ["purity", "moved_in", "phi"]


def binary_dictionary():
    cache = OUT / "community_shape_binary_dictionary.json"
    if cache.exists():
        return json.loads(cache.read_text())
    old = dpt.TREE
    dpt.TREE = WITHIN
    try:
        syn = dpt.synthetic()
    finally:
        dpt.TREE = old
    d = {k: v["mean_profile"] for k, v in syn.items()}
    cache.write_text(json.dumps(d, indent=1))
    return d


DICT = binary_dictionary()


def nearest_family(prof):
    best, bd = None, np.inf
    for f in FAMS:
        fp = DICT[f]
        pairs = [
            (a, b)
            for a, b in zip(prof[:NRUNG], fp[:NRUNG])
            if a is not None and np.isfinite(a) and b is not None and np.isfinite(b)
        ]
        if len(pairs) < 3:
            continue
        d = np.sqrt(np.mean([(np.log1p(a) - np.log1p(b)) ** 2 for a, b in pairs]))
        if d < bd:
            best, bd = f, d
    return best


def leaf_table(tree, Xn_sub):
    """per leaf of a tree over Xn_sub: local indices, profile, family, summary stats."""
    rows = []
    for lf in leaf_profiles(tree, Xn_sub):
        p = np.array([v if v is not None else np.nan for v in lf["profile"]], dtype=float)
        ok = np.isfinite(p)
        rungs = int(ok.sum())
        slope = float(np.polyfit(np.flatnonzero(ok), p[ok], 1)[0]) if rungs >= 3 else np.nan
        rows.append(
            dict(
                idx=lf["indices"],
                profile=p,
                rungs=rungs,
                fam=nearest_family(p),
                mean_d=float(np.nanmean(p)) if rungs else np.nan,
                min_d=float(np.nanmin(p)) if rungs else np.nan,
                slope=slope,
                dip=bool(rungs and np.nanmin(p) < 0.5),
                spike=bool(rungs and np.nanmax(p) >= 20),
            )
        )
    return rows


def aggregate(rows, weights):
    """points-weighted shape features from leaf rows; weights = points of each leaf inside the community."""
    w = np.asarray(weights, dtype=float)
    tot = w.sum()
    if tot == 0:
        return {f: np.nan for f in SHAPE_FEATS}

    def wmean(key):
        v = np.array([r[key] for r in rows], dtype=float)
        m = np.isfinite(v) & (w > 0)
        return float(np.average(v[m], weights=w[m])) if m.any() else np.nan

    fam = np.array([r["fam"] or "none" for r in rows])
    share = lambda names: float(w[np.isin(fam, names)].sum() / tot)  # noqa: E731
    return dict(
        mean_d=wmean("mean_d"),
        min_d=wmean("min_d"),
        slope=wmean("slope"),
        dip_share=float(w[[r["dip"] for r in rows]].sum() / tot),
        spike_share=float(w[[r["spike"] for r in rows]].sum() / tot),
        pocket_share=share(["blobs"]),
        curve_sheet_share=share(["curve", "sheet"]),
        ball_share=share(["ball"]),
        rungs=wmean("rungs"),
        unread=share(["none"]),
    )


def mean_profile(rows, weights, J=NRUNG):
    P = np.full((len(rows), J), np.nan)
    for i, r in enumerate(rows):
        p = r["profile"][:J]
        P[i, : len(p)] = p
    w = np.asarray(weights, dtype=float)[:, None] * np.isfinite(P)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.nansum(np.nan_to_num(P) * w, 0) / w.sum(0)


def fmt(p):
    return " ".join(f"{v:4.1f}" if np.isfinite(v) else "   ." for v in p)


def per_community_phi(lab, J):
    n, k = J.shape
    src = np.repeat(lab, k)
    dst = lab[J.ravel()]
    cross = src != dst
    out_edges = np.bincount(src[cross], minlength=lab.max() + 1)
    size = np.bincount(lab, minlength=lab.max() + 1)
    vol = k * size
    return out_edges / np.maximum(np.minimum(vol, k * n - vol), 1)


def run(name):
    t0 = time.perf_counter()
    X, y = load(name)
    n = len(X)
    Xn = unit(X)
    J = np.load(OUT / f"knn15_{name}.npy")
    # shipped pipeline
    item_leaf4 = leaf_map(build_dyf_tree(X, seed=SEED, **SHIPPED), n)
    L4 = item_leaf4.max() + 1
    cent4 = np.zeros((L4, X.shape[1]), dtype=np.float32)
    np.add.at(cent4, item_leaf4, Xn)
    cent4 = unit(cent4)
    ll, _, _ = _run_louvain_on_centroids(cent4, k=10, resolution=1.0, similarity_threshold=0.5)
    block = np.asarray(ll)[item_leaf4].astype(np.int32)
    final = _reassign_points(block.copy(), X)
    _, final = np.unique(final, return_inverse=True)
    moved = final != block
    phi = per_community_phi(final, J)
    K = final.max() + 1
    t1 = time.perf_counter()
    # global binary ladder, once
    grows = leaf_table(build_dyf_tree(X, seed=SEED, **GLOBAL), Xn)
    gleaf = np.full(n, -1)
    for i, r in enumerate(grows):
        gleaf[r["idx"]] = i
    t2 = time.perf_counter()
    print(
        f"\n== {name}: n={n} communities={K} (>= {MIN_COMM}: {(np.bincount(final) >= MIN_COMM).sum()}); shipped {t1 - t0:.0f}s, global binary tree {len(grows)} leaves {t2 - t1:.0f}s",
        flush=True,
    )
    comms = []
    tw = 0.0
    for c in range(K):
        idx = np.flatnonzero(final == c)
        if len(idx) < MIN_COMM:
            continue
        cls = int(np.bincount(y[idx]).argmax())
        cen = Xn[idx].mean(0)
        cen /= np.linalg.norm(cen) + 1e-12
        row = dict(
            comm=int(c),
            size=int(len(idx)),
            log_size=float(np.log(len(idx))),
            radius=float(1 - (Xn[idx] @ cen).mean()),
            phi=float(phi[c]),
            cls=cls,
            purity=float(np.mean(y[idx] == cls)),
            moved_in=float(moved[idx].mean()),
        )
        # within-community tree
        tc = time.perf_counter()
        wrows = leaf_table(build_dyf_tree(np.ascontiguousarray(X[idx]), seed=SEED, **WITHIN), Xn[idx])
        tw += time.perf_counter() - tc
        row["within"] = aggregate(wrows, [len(r["idx"]) for r in wrows])
        row["within_profile"] = [float(v) for v in mean_profile(wrows, [len(r["idx"]) for r in wrows])]
        # global leaves restricted to the community
        gl = np.bincount(gleaf[idx], minlength=len(grows))
        sub = np.flatnonzero(gl)
        row["global"] = aggregate([grows[i] for i in sub], gl[sub])
        row["global_profile"] = [float(v) for v in mean_profile([grows[i] for i in sub], gl[sub])]
        comms.append(row)
    print(f"   within-community trees: {len(comms)} communities, {tw:.0f}s total", flush=True)
    # eyeball: largest 6 communities
    print(
        "   largest communities: size | purity cls | moved-in | phi || within: rungs fam-mix(curve+sheet/ball/pocket) dip spike | profile || global-restricted profile"
    )
    for r in sorted(comms, key=lambda r: -r["size"])[:6]:
        w, _g = r["within"], r["global"]
        print(
            f"   {r['size']:7d} | {r['purity']:.2f} {r['cls']:3d} | {r['moved_in']:.2f} | {r['phi']:.3f} || {w['rungs']:.1f} {w['curve_sheet_share']:.2f}/{w['ball_share']:.2f}/{w['pocket_share']:.2f} {w['dip_share']:.2f} {w['spike_share']:.2f} | {fmt(r['within_profile'])} || {fmt(r['global_profile'])}",
            flush=True,
        )
    # does the community stage reassemble classes the root split?  communities holding >= 5% of each large class
    reassembly = {}
    for t in np.argsort(-np.bincount(y))[:8]:
        dist = np.bincount(final[y == t], minlength=K) / (y == t).sum()
        reassembly[int(t)] = dict(communities=int((dist >= 0.05).sum()), largest=float(dist.max()))
    print(
        "   largest classes -> communities holding >= 5%: "
        + ", ".join(f"{t}: {v['communities']} (max {v['largest']:.2f})" for t, v in reassembly.items()),
        flush=True,
    )
    # within-dataset Spearman per feature vs target
    sp = {}
    for arm in ("within", "global"):
        for f in SHAPE_FEATS:
            v = np.array([r[arm][f] for r in comms], dtype=float)
            for tgt in TARGETS:
                u = np.array([r[tgt] for r in comms], dtype=float)
                m = np.isfinite(v) & np.isfinite(u)
                sp[f"{arm}.{f}|{tgt}"] = (
                    float(spearmanr(v[m], u[m])[0]) if m.sum() >= 5 and np.std(v[m]) > 0 else float("nan")
                )
    for f in FREE_FEATS:
        v = np.array([r[f] for r in comms], dtype=float)
        for tgt in TARGETS:
            u = np.array([r[tgt] for r in comms], dtype=float)
            sp[f"free.{f}|{tgt}"] = float(spearmanr(v, u)[0]) if len(v) >= 5 and np.std(v) > 0 else float("nan")
    print(
        "   Spearman vs purity (within / global / free): "
        + "  ".join(
            f"{f} {sp[f'within.{f}|purity']:+.2f}/{sp[f'global.{f}|purity']:+.2f}"
            for f in ["mean_d", "min_d", "dip_share", "pocket_share", "spike_share"]
        )
        + "  "
        + "  ".join(f"{f} {sp[f'free.{f}|purity']:+.2f}" for f in FREE_FEATS),
        flush=True,
    )
    return dict(
        n=n,
        communities=K,
        scored=len(comms),
        seconds_shipped=t1 - t0,
        seconds_global=t2 - t1,
        seconds_within=tw,
        reassembly=reassembly,
        spearman=sp,
        comms=comms,
        seconds=time.perf_counter() - t0,
    )


def pooled(results):
    """ranks within dataset, pooled Spearman with a within-dataset permutation null; LODO AUC."""
    names = [k for k in ORDER if k in results]
    rng = np.random.default_rng(0)
    feat_cols, groups, T = {}, [], {t: [] for t in TARGETS}
    for arm in ("within", "global"):
        for f in SHAPE_FEATS:
            feat_cols[f"{arm}.{f}"] = []
    for f in FREE_FEATS:
        feat_cols[f"free.{f}"] = []
    for gi, name in enumerate(names):
        comms = results[name]["comms"]
        if len(comms) < 4:
            continue
        for r in comms:
            groups.append(gi)
            for t in TARGETS:
                T[t].append(r[t])
            for arm in ("within", "global"):
                for f in SHAPE_FEATS:
                    feat_cols[f"{arm}.{f}"].append(r[arm][f])
            for f in FREE_FEATS:
                feat_cols[f"free.{f}"].append(r[f])
    groups = np.array(groups)

    def within_rank(v):
        v = np.array(v, dtype=float)
        out = np.full_like(v, np.nan)
        for g in np.unique(groups):
            m = (groups == g) & np.isfinite(v)
            if m.sum() > 1:
                out[m] = (rankdata(v[m]) - 1) / (m.sum() - 1)
        return out

    R = {k: within_rank(v) for k, v in feat_cols.items()}
    RT = {t: within_rank(v) for t, v in T.items()}
    print(f"\n=== pooled over {len(groups)} communities from {len(np.unique(groups))} datasets (ranks within dataset)")
    print("   feature                 | rho vs purity  (null 95%)  | rho vs moved-in | rho vs phi")
    pooled_rows = {}
    for k, v in R.items():
        line = f"   {k:24s}|"
        pooled_rows[k] = {}
        for t in TARGETS:
            u = RT[t]
            m = np.isfinite(v) & np.isfinite(u)
            if m.sum() < 10:
                line += "      .         |"
                continue
            rho = float(spearmanr(v[m], u[m])[0])
            null = []
            for _ in range(500):
                p = u[m].copy()
                gm = groups[m]
                for g in np.unique(gm):
                    mm = gm == g
                    p[mm] = rng.permutation(p[mm])
                null.append(spearmanr(v[m], p)[0])
            q = float(np.percentile(np.abs(null), 95))
            pooled_rows[k][t] = dict(rho=rho, null95=q, n=int(m.sum()))
            line += (
                f" {rho:+.3f} ({q:.3f}){'*' if abs(rho) > q else ' '} |"
                if t == "purity"
                else f" {rho:+.3f}{'*' if abs(rho) > q else ' '} |"
            )
        print(line, flush=True)
    # LODO AUC: bottom-quartile purity within dataset
    pur = np.array(T["purity"], dtype=float)
    ybin = np.zeros(len(pur), dtype=int)
    for g in np.unique(groups):
        m = groups == g
        ybin[m] = pur[m] <= np.percentile(pur[m], 25)
    sets = {
        "FREE": [f"free.{f}" for f in FREE_FEATS],
        "GLOBAL": [f"global.{f}" for f in SHAPE_FEATS],
        "WITHIN": [f"within.{f}" for f in SHAPE_FEATS],
        "WITHIN+FREE": [f"within.{f}" for f in SHAPE_FEATS] + [f"free.{f}" for f in FREE_FEATS],
        "GLOBAL+FREE": [f"global.{f}" for f in SHAPE_FEATS] + [f"free.{f}" for f in FREE_FEATS],
    }
    print(
        "\n   leave-one-dataset-out logistic AUC, target = bottom-quartile purity within its dataset (features rank-normalised within dataset)"
    )
    print("   set          " + " ".join(f"{names[g][:8]:>9s}" for g in np.unique(groups)) + "    pooled")
    lodo = {}
    for sname, cols in sets.items():
        F = np.column_stack([R[c] for c in cols])
        aucs, pooled_scores = [], np.full(len(ybin), np.nan)
        for g in np.unique(groups):
            te, tr = groups == g, groups != g
            if ybin[te].min() == ybin[te].max():
                aucs.append(np.nan)
                continue
            clf = make_pipeline(SimpleImputer(), StandardScaler(), LogisticRegression(C=1.0, max_iter=2000))
            clf.fit(F[tr], ybin[tr])
            s = clf.predict_proba(F[te])[:, 1]
            pooled_scores[te] = s
            aucs.append(float(roc_auc_score(ybin[te], s)))
        m = np.isfinite(pooled_scores)
        pa = float(roc_auc_score(ybin[m], pooled_scores[m]))
        lodo[sname] = dict(per_dataset=aucs, pooled=pa)
        print(
            f"   {sname:12s} "
            + " ".join(f"{a:9.2f}" if np.isfinite(a) else "        ." for a in aucs)
            + f"    {pa:.3f}",
            flush=True,
        )
    return dict(pooled_spearman=pooled_rows, lodo_auc=lodo, n_communities=int(len(groups)))


def jsonable(o):
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.floating):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(type(o).__name__)


if __name__ == "__main__":
    names = sys.argv[1:] or ORDER
    out_path = OUT / "community_shape_probe_results.json"
    results = json.loads(out_path.read_text()) if out_path.exists() else {}
    for name in names:
        results[name] = run(name)
        out_path.write_text(
            json.dumps({k: v for k, v in results.items() if k != "_pooled"}, indent=1, default=jsonable)
        )
    results["_pooled"] = pooled(results)
    out_path.write_text(json.dumps(results, indent=1, default=jsonable))
    print("\nwrote", out_path)
