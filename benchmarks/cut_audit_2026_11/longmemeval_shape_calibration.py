"""Does the shape of the branch a query lands in add calibration beyond dyf's incumbent
routing signals?  LongMemEval pooled corpus, MiniLM-L6 384-d, 500 questions.

Incumbent (benchmarks/longmemeval_diagnostics.py, 2026-04-07): logistic regression on
top1_flat_cos (baseline) + dyf routing features {max_centroid_cos, leaf_size_np1, min_margin,
candidates_scored} predicting hit = gold in flat top-5; routing added +0.076 CV AUC then.
New per-query features from a DEEP tree (8-way, depth 8, seeds 42/1/2) on the pool:
  q_dim        mean local dimension along the branch the query routes into (seed 42)
  q_profile    its per-rung dimensions (first 4 rungs, forward-filled)
  q_family     nearest dictionary family (one-hot)
  q_stab       cross-seed profile distance of that leaf's points (mean), lower = pinned
  q_leaf_r     angular radius of the leaf; q_leaf_n its size
  q_margin     min over rungs of (own-child cos - best-other cos) / node radius for the query
Models (5-fold CV AUC, standardised features): top1; top1+top5; incumbent full; full+shape;
shape alone; also 25%-budget abstention recall for full vs full+shape.
Pre-registered bar: shape must add >= +0.02 CV AUC over the incumbent full model.
Env: /Users/jdonaldson/Projects/dyf/.venv/bin/python longmemeval_shape_calibration.py
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

from dyf import LazyIndex, build_dyf_tree, write_lazy_index
from dyf.configs import EmbedderConfig

sys.path.insert(0, "/Volumes/Models/dyf_bench_2026-11-04")
from dimension_profile_test import MIN_NODE  # noqa: E402
from hash_leaf_graph_test import index_tree, route  # noqa: E402
from leaf_shape_families import FAMS, nearest_family  # noqa: E402

DATA = Path("/Volumes/Models/longmemeval-data/longmemeval_s_cleaned.json")
CACHE = Path("/Volumes/Models/longmemeval-data")
OUT = Path("/Volumes/Models/dyf_bench_2026-11-04")
K = 5
DEEP = dict(num_bits=3, max_depth=8, min_leaf_size=20)
SEEDS = (42, 1, 2)


def build_pool(data):
    texts, ids, id_to_idx = [], [], {}
    for e in data:
        for sid, sess in zip(e["haystack_session_ids"], e["haystack_sessions"]):
            if sid in id_to_idx:
                continue
            id_to_idx[sid] = len(texts)
            ids.append(sid)
            texts.append("\n".join(t.get("content", "") for t in sess if isinstance(t, dict)))
    return texts, ids, id_to_idx


def unit(a):
    return a / np.maximum(np.linalg.norm(a, axis=1, keepdims=True), 1e-12)


def embed_cached(texts, tag):
    p = CACHE / f"minilm_{tag}.npy"
    if p.exists():
        return np.load(p)
    E = EmbedderConfig.LOW.embed(texts, batch_size=64, verbose=True).astype(np.float32)
    np.save(p, E)
    return E


def deep_tree_features(pool, queries, seed):
    """Per query: profile of the routed leaf (first 4 rungs), leaf radius/size, margin ratio."""
    tree = build_dyf_tree(pool, seed=seed, **DEEP)
    n = len(pool)
    item_leaf, _ = index_tree(tree, n)
    Xn = unit(pool)
    Qn = unit(queries)
    # per-node stats along paths: build leaf -> (profile, radius, size); and route queries
    leaf_prof, leaf_r, leaf_n = {}, {}, {}

    def stats(idx):
        c = Xn[idx].mean(0)
        c /= np.linalg.norm(c) + 1e-12
        return c, max(1.0 - float((Xn[idx] @ c).mean()), 1e-6)

    def walk(node, lad):
        idx = np.asarray(node["indices"])
        c, r = stats(idx)
        lad = lad + [(len(idx), r)]
        ch = node.get("children") or []
        if not ch:
            prof = []
            for j in range(1, len(lad)):
                n0, r0 = lad[j - 1]
                n1, r1 = lad[j]
                dr = np.log(1 / r1) - np.log(1 / r0)
                d = (np.log(n0) - np.log(n1)) / dr if dr > 1e-6 else np.nan
                prof.append(np.nan if (n1 < MIN_NODE or not np.isfinite(d)) else float(np.clip(d, 0, 60)))
            leaf_prof[node["leaf_id"]] = prof
            leaf_r[node["leaf_id"]] = r
            leaf_n[node["leaf_id"]] = len(idx)
            return
        for k in ch:
            walk(k, lad)

    walk(tree, [])
    q_leaf = route(tree, Qn, np.arange(len(Qn)))
    # query margin ratio along its path: descend again recording margins
    q_margin = np.full(len(Qn), np.inf)

    def descend(node, rows):
        ch = node.get("children") or []
        if not ch or not len(rows):
            return
        hp = np.asarray(node["hyperplanes"], dtype=np.float32)
        idx = np.asarray(node["indices"])
        _, r = stats(idx)
        kid_c = np.stack([stats(np.asarray(k["indices"]))[0] for k in ch])
        S = Qn[rows] @ kid_c.T
        best = S.argmax(1)
        own = S[np.arange(len(rows)), best]
        S[np.arange(len(rows)), best] = -np.inf
        other = S.max(1) if S.shape[1] > 1 else np.full(len(rows), -1.0)
        q_margin[rows] = np.minimum(q_margin[rows], (own - other) / r)
        P = Qn[rows] @ hp.T
        bits = (P > 0).astype(np.int64)
        ids = (bits * (1 << np.arange(hp.shape[0]))).sum(1)
        b2c = {int(k): v for k, v in node["bucket_id_to_child"].items()}
        for bid in np.unique(ids):
            m = ids == bid
            if int(bid) in b2c:
                descend(ch[b2c[int(bid)]], rows[m])

    descend(tree, np.arange(len(Qn)))
    q_margin[~np.isfinite(q_margin)] = np.nan
    P = np.full((len(Qn), 4), np.nan)
    fam = np.full(len(Qn), -1)
    r = np.zeros(len(Qn))
    sz = np.zeros(len(Qn))
    for i, lf in enumerate(q_leaf):
        pr = leaf_prof[lf]
        P[i] = (pr + [np.nan] * 4)[:4]
        f, _ = nearest_family(pr)
        fam[i] = FAMS.index(f) if f is not None else -1
        r[i] = leaf_r[lf]
        sz[i] = leaf_n[lf]
    return dict(
        profile=P,
        dim=np.nanmean(P, axis=1),
        fam=fam,
        radius=r,
        size=sz,
        margin=q_margin,
        item_leaf=item_leaf,
        q_leaf=q_leaf,
        leaf_prof=leaf_prof,
    )


def cv_auc(X, y, seed=0):
    X = np.nan_to_num(X, nan=np.nanmean(X, axis=0) if X.ndim == 2 else 0.0)
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    scores = np.zeros(len(y))
    for tr, te in skf.split(X, y):
        sc = StandardScaler().fit(X[tr])
        m = LogisticRegression(max_iter=2000, C=1.0).fit(sc.transform(X[tr]), y[tr])
        scores[te] = m.predict_proba(sc.transform(X[te]))[:, 1]
    return float(roc_auc_score(y, scores)), scores


def main():
    data = json.loads(DATA.read_text())
    pool_texts, pool_ids, id_to_idx = build_pool(data)
    entries = [(e, [g for g in e["answer_session_ids"] if g in id_to_idx]) for e in data]
    entries = [(e, g) for e, g in entries if g]
    questions = [e["question"] for e, _ in entries]
    gold = [set(id_to_idx[s] for s in g) for _, g in entries]
    print(f"pool {len(pool_texts)} sessions, {len(questions)} questions", flush=True)
    t0 = time.perf_counter()
    pool = unit(embed_cached(pool_texts, "pool"))
    queries = unit(embed_cached(questions, "queries"))
    print(f"embeddings ready ({time.perf_counter() - t0:.0f}s)", flush=True)
    # flat retrieval
    sims = queries @ pool.T
    top = np.argsort(-sims, axis=1)[:, :K]
    hits = np.array([int(bool(set(top[i].tolist()) & gold[i])) for i in range(len(queries))])
    top1 = sims[np.arange(len(queries)), top[:, 0]]
    top5 = np.take_along_axis(sims, top, axis=1).mean(1)
    print(f"flat recall_any@5 = {hits.mean():.4f}", flush=True)
    # incumbent routing features (depth-4 tree as in the diagnostics script)
    tree4 = build_dyf_tree(pool, max_depth=4, num_bits=3, min_leaf_size=8, seed=42, fit_method="raw_pca")
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "diag.dyf")
        write_lazy_index(
            tree4,
            pool,
            path,
            compression="none",
            quantization="float32",
            build_params={"max_depth": 4, "num_bits": 3, "min_leaf_size": 8, "seed": 42},
        )
        idx = LazyIndex(path)
        _ = idx.search_ivf(queries[0], k=1, nprobe=1)
        cents = idx._centroid_matrix
        max_cc = (queries @ cents.T).max(axis=1)
        leaf_np1, min_margin, cand = np.zeros(len(queries)), np.zeros(len(queries)), np.zeros(len(queries))
        for i, q in enumerate(queries):
            r1 = idx.search(q, k=K, nprobe=1, return_routing=True).routing or {}
            leaf_np1[i] = r1.get("candidates_scored", 0)
            r3 = idx.search(q, k=K, nprobe=3, return_routing=True).routing or {}
            mm = r3.get("min_margin")
            min_margin[i] = mm if mm is not None else 0.0
            cand[i] = r3.get("candidates_scored", 0)
    # shape features from deep trees
    feats = {s: deep_tree_features(pool, queries, s) for s in SEEDS}
    f42 = feats[42]

    # cross-seed stability of the query's leaf points (seed-42 leaf), from per-point profiles
    def point_profiles(f):
        P = np.full((len(pool), 4), np.nan)
        for lf, pr in f["leaf_prof"].items():
            P[f["item_leaf"] == lf] = (pr + [np.nan] * 4)[:4]
        return P

    PP = {s: point_profiles(feats[s]) for s in SEEDS}

    def pair_dist(A, B):
        D = (np.log1p(A) - np.log1p(B)) ** 2
        ok = ~np.isnan(D)
        cnt = ok.sum(1)
        out = np.sqrt(np.where(ok, D, 0).sum(1) / np.maximum(cnt, 1))
        out[cnt < 2] = np.nan
        return out

    stab_pt = np.nanmean(
        np.stack([pair_dist(PP[42], PP[1]), pair_dist(PP[42], PP[2]), pair_dist(PP[1], PP[2])]), axis=0
    )
    q_stab = np.array([np.nanmean(stab_pt[f42["item_leaf"] == lf]) for lf in f42["q_leaf"]])
    fam_oh = np.stack([(f42["fam"] == i).astype(float) for i in range(len(FAMS))], axis=1)
    prof_ff = f42["profile"].copy()
    for j in range(1, 4):
        bad = np.isnan(prof_ff[:, j])
        prof_ff[bad, j] = prof_ff[bad, j - 1]
    y = hits
    sets = {
        "top1": np.column_stack([top1]),
        "top1+top5": np.column_stack([top1, top5]),
        "incumbent full": np.column_stack([top1, max_cc, leaf_np1, min_margin, cand]),
        "shape only": np.column_stack(
            [f42["dim"], prof_ff, fam_oh, q_stab, f42["radius"], np.log1p(f42["size"]), f42["margin"]]
        ),
        "full + shape": np.column_stack(
            [
                top1,
                max_cc,
                leaf_np1,
                min_margin,
                cand,
                f42["dim"],
                prof_ff,
                fam_oh,
                q_stab,
                f42["radius"],
                np.log1p(f42["size"]),
                f42["margin"],
            ]
        ),
        "full + top5 + shape": np.column_stack(
            [
                top1,
                top5,
                max_cc,
                leaf_np1,
                min_margin,
                cand,
                f42["dim"],
                prof_ff,
                fam_oh,
                q_stab,
                f42["radius"],
                np.log1p(f42["size"]),
                f42["margin"],
            ]
        ),
    }
    res = {"n_queries": int(len(y)), "hit_rate": float(y.mean()), "cv_auc": {}, "solo_auc": {}, "abstain25": {}}
    print("\nCV AUC (5-fold, 3 repeats):")
    scores = {}
    for name, X in sets.items():
        aucs, sc = [], None
        for rep in range(3):
            a, sc = cv_auc(X, y, seed=rep)
            aucs.append(a)
        res["cv_auc"][name] = [float(np.mean(aucs)), float(np.std(aucs))]
        scores[name] = sc
        print(f"   {name:22s} {np.mean(aucs):.4f} ± {np.std(aucs):.4f}", flush=True)
    solo = {
        "q_dim": f42["dim"],
        "q_stab": q_stab,
        "q_radius": f42["radius"],
        "q_size": f42["size"],
        "q_margin": f42["margin"],
        "top1": top1,
        "top5": top5,
        "max_centroid_cos": max_cc,
        "min_margin": min_margin,
    }
    print("\nsolo AUC (rank only; sign-corrected):")
    for k_, v in solo.items():
        m = ~np.isnan(v)
        a = roc_auc_score(y[m], v[m])
        res["solo_auc"][k_] = float(a)
        print(f"   {k_:18s} {max(a, 1 - a):.3f} ({'+' if a >= 0.5 else '-'})", flush=True)
    print("\nby query family (seed 42): n, hit rate, mean top1")
    for i, f in enumerate(FAMS):
        m = f42["fam"] == i
        if m.sum():
            print(f"   {f:8s} n={m.sum():3d} hit={y[m].mean():.3f} top1={top1[m].mean():.3f}", flush=True)
    print("\n25%-budget abstention (keep the 75% most confident): recall among kept")
    for name in ("top1", "incumbent full", "full + shape", "full + top5 + shape"):
        sc = scores[name]
        keep = sc >= np.percentile(sc, 25)
        res["abstain25"][name] = float(y[keep].mean())
        print(f"   {name:22s} {y[keep].mean():.4f}", flush=True)
    (OUT / "longmemeval_shape_calibration_results.json").write_text(json.dumps(res, indent=1))
    print("\nwrote", OUT / "longmemeval_shape_calibration_results.json")


if __name__ == "__main__":
    main()
