"""LongMemEval query diagnostics: does dyf predict retrieval quality?

The hypothesis (from the project CLAUDE.md): dyf tree is not a competitive
retriever at this scale, but its structure may serve as a query-time
*diagnostic* — flagging queries that land in sparse regions, far from
centroids, or near routing boundaries as likely to retrieve poorly.

If dyf's signals add predictive power beyond what the top-1 flat cosine
similarity already reveals, dyf is useful as a confidence/query-expansion
layer on top of flat retrieval. If not, it's redundant.

Features tested per query:
  - top1_flat_cos:     top-1 cosine sim from flat brute-force (baseline)
  - top5_mean_cos:     mean of top-5 flat cosine sims (sharpness)
  - max_centroid_cos:  max cosine sim to any leaf centroid
  - leaf_size:         size of the leaf the query routes into (single-probe)
  - min_margin:        min decision margin along the routing path
  - candidates_scored: total docs scored under nprobe=3

Label: did flat top-5 contain any gold answer_session_id? (hit=1, miss=0)

Analysis:
  1. Quartile tables — recall_any@5 split by each signal
  2. Logistic regression — all features, check marginal coefficients
  3. Correlation matrix — rule out pure redundancy

Usage:
    python benchmarks/longmemeval_diagnostics.py
    python benchmarks/longmemeval_diagnostics.py --out benchmarks/results/diag.json
"""

from __future__ import annotations

import argparse
import json
import os
import tempfile
import time
from pathlib import Path

import numpy as np

from dyf import EmbedderConfig, LazyIndex, build_dyf_tree, write_lazy_index

DEFAULT_DATA = "/tmp/longmemeval-data/longmemeval_s_cleaned.json"
K = 5


def build_pool(data: list[dict]) -> tuple[list[str], list[str], dict[str, int]]:
    id_to_idx: dict[str, int] = {}
    pool_texts: list[str] = []
    pool_ids: list[str] = []
    for entry in data:
        for session, sess_id in zip(
            entry["haystack_sessions"], entry["haystack_session_ids"]
        ):
            if sess_id in id_to_idx:
                continue
            user_turns = [t["content"] for t in session if t["role"] == "user"]
            doc = "\n".join(user_turns)
            if not doc.strip():
                continue
            id_to_idx[sess_id] = len(pool_texts)
            pool_texts.append(doc)
            pool_ids.append(sess_id)
    return pool_texts, pool_ids, id_to_idx


def l2_normalize(x: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return x / norms


def quartile_table(values: np.ndarray, labels: np.ndarray, name: str) -> str:
    """Bucket queries into quartiles by `values` and report hit rate."""
    q0 = float(np.quantile(values, 0.25))
    q1 = float(np.quantile(values, 0.50))
    q2 = float(np.quantile(values, 0.75))
    lo = values <= q0
    m1 = (values > q0) & (values <= q1)
    m2 = (values > q1) & (values <= q2)
    hi = values > q2
    rows = []
    rows.append(f"  {name}")
    rows.append(f"    Q1 (<= {q0:.4f}): n={lo.sum():3d}  hit_rate={labels[lo].mean():.3f}")
    rows.append(f"    Q2 (<= {q1:.4f}): n={m1.sum():3d}  hit_rate={labels[m1].mean():.3f}")
    rows.append(f"    Q3 (<= {q2:.4f}): n={m2.sum():3d}  hit_rate={labels[m2].mean():.3f}")
    rows.append(f"    Q4 (>  {q2:.4f}): n={hi.sum():3d}  hit_rate={labels[hi].mean():.3f}")
    return "\n".join(rows)


def logistic_regression(X: np.ndarray, y: np.ndarray, feature_names: list[str]) -> dict:
    """Standardize features, fit sklearn LogisticRegression, report coefs + AUC."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.preprocessing import StandardScaler

    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)
    model = LogisticRegression(max_iter=2000, C=1.0)
    model.fit(Xs, y)
    probs = model.predict_proba(Xs)[:, 1]
    auc = roc_auc_score(y, probs)
    coefs = {name: float(c) for name, c in zip(feature_names, model.coef_[0])}
    intercept_arr = np.asarray(model.intercept_).flatten()
    return {"auc": float(auc), "coefs": coefs, "intercept": float(intercept_arr[0])}


def solo_auc(values: np.ndarray, labels: np.ndarray) -> float:
    """AUC of a single feature vs binary label (no model — just rank)."""
    from sklearn.metrics import roc_auc_score
    return float(roc_auc_score(labels, values))


def run(args: argparse.Namespace) -> None:
    data_path = Path(args.data)
    if not data_path.exists():
        raise SystemExit(f"Dataset not found at {data_path}")

    print(f"Loading {data_path}...")
    with data_path.open() as f:
        data = json.load(f)
    if args.limit > 0:
        data = data[: args.limit]

    print("Pooling...")
    pool_texts, pool_ids, id_to_idx = build_pool(data)
    n_pool = len(pool_texts)
    print(f"  pool size: {n_pool}")

    entries = []
    for e in data:
        gold = [g for g in e["answer_session_ids"] if g in id_to_idx]
        if gold:
            entries.append((e, gold))
    questions = [e["question"] for e, _ in entries]
    gold_sets = [set(g) for _, g in entries]
    print(f"  {len(entries)} questions")

    embed_cfg = getattr(EmbedderConfig, args.model)
    print(f"Embedder: {embed_cfg.name}")
    print("Embedding...")
    t0 = time.time()
    all_texts = pool_texts + questions
    all_embs = embed_cfg.embed(all_texts, batch_size=64, verbose=True)
    all_embs = l2_normalize(all_embs.astype(np.float32))
    print(f"  done in {time.time() - t0:.1f}s")

    pool_embs = all_embs[:n_pool]
    query_embs = all_embs[n_pool:]

    # -------------------------------------------------------------------------
    # Flat ground truth + per-query top1/top5 cosine
    # -------------------------------------------------------------------------
    print("\nComputing flat retrieval signals...")
    top1_cos = np.empty(len(query_embs), dtype=np.float32)
    top5_mean_cos = np.empty(len(query_embs), dtype=np.float32)
    hits = np.empty(len(query_embs), dtype=np.int32)

    # Batched matmul for efficiency
    sims_all = query_embs @ pool_embs.T  # (Q, N)
    for i in range(len(query_embs)):
        sims = sims_all[i]
        top_k_idx = np.argpartition(-sims, K)[:K]
        top_k_idx = top_k_idx[np.argsort(-sims[top_k_idx])]
        top1_cos[i] = sims[top_k_idx[0]]
        top5_mean_cos[i] = sims[top_k_idx].mean()
        top_ids = {pool_ids[j] for j in top_k_idx}
        hits[i] = int(bool(top_ids & gold_sets[i]))
    print(f"  hit rate (recall_any@5): {hits.mean():.4f}")

    # -------------------------------------------------------------------------
    # Dyf tree diagnostics
    # -------------------------------------------------------------------------
    print("\nBuilding dyf tree...")
    t0 = time.time()
    tree = build_dyf_tree(
        pool_embs,
        max_depth=args.max_depth,
        num_bits=args.num_bits,
        min_leaf_size=8,
        seed=42,
        fit_method=args.fit_method,
    )
    print(f"  tree: {time.time() - t0:.1f}s")

    with tempfile.TemporaryDirectory() as tmp:
        index_path = os.path.join(tmp, "diag.dyf")
        write_lazy_index(
            tree,
            pool_embs,
            index_path,
            compression="none",
            quantization="float32",
            build_params={
                "max_depth": args.max_depth,
                "num_bits": args.num_bits,
                "min_leaf_size": 8,
                "seed": 42,
            },
        )
        idx = LazyIndex(index_path)

        # Centroid matrix (built lazily inside search_ivf — trigger it)
        _ = idx.search_ivf(query_embs[0], k=1, nprobe=1)
        centroids = idx._centroid_matrix  # (n_leaves, dim), unit-norm
        print(f"  tree has {len(centroids)} leaves")

        # Per-query dyf signals
        max_centroid_cos = np.empty(len(query_embs), dtype=np.float32)
        leaf_size_np1 = np.empty(len(query_embs), dtype=np.int32)
        min_margin = np.empty(len(query_embs), dtype=np.float32)
        candidates_scored = np.empty(len(query_embs), dtype=np.int32)

        # Max centroid cosine: global diagnostic, not tied to routing
        max_centroid_cos[:] = (query_embs @ centroids.T).max(axis=1)

        print("Collecting routing diagnostics...")
        for i, q in enumerate(query_embs):
            # nprobe=1 gives "the one leaf the query routes into"
            r1 = idx.search(q, k=K, nprobe=1, return_routing=True)
            routing1 = r1.routing or {}
            leaf_size_np1[i] = routing1.get("candidates_scored", 0)
            # nprobe=3 for margin + candidate count (matches tree defaults)
            r3 = idx.search(q, k=K, nprobe=3, return_routing=True)
            routing3 = r3.routing or {}
            mm = routing3.get("min_margin")
            min_margin[i] = mm if mm is not None else 0.0
            candidates_scored[i] = routing3.get("candidates_scored", 0)
        print("  done")

    # -------------------------------------------------------------------------
    # Analysis
    # -------------------------------------------------------------------------
    print("\n" + "=" * 72)
    print("  QUARTILE TABLES — recall_any@5 by signal decile")
    print("=" * 72)
    print(quartile_table(top1_cos, hits, "top1_flat_cos (BASELINE)"))
    print(quartile_table(top5_mean_cos, hits, "top5_mean_cos"))
    print(quartile_table(max_centroid_cos, hits, "max_centroid_cos"))
    print(quartile_table(-leaf_size_np1.astype(np.float32), hits, "leaf_size (inverted: small=bad)"))
    print(quartile_table(min_margin, hits, "min_margin"))
    print(quartile_table(-candidates_scored.astype(np.float32), hits, "candidates_scored (inverted)"))

    # Solo AUCs
    print("\n" + "=" * 72)
    print("  SOLO AUC (each feature alone as predictor of hit)")
    print("=" * 72)
    for name, vals in [
        ("top1_flat_cos (BASELINE)", top1_cos),
        ("top5_mean_cos", top5_mean_cos),
        ("max_centroid_cos", max_centroid_cos),
        ("leaf_size_np1", leaf_size_np1.astype(np.float32)),
        ("min_margin", min_margin),
        ("candidates_scored", candidates_scored.astype(np.float32)),
    ]:
        auc = solo_auc(vals, hits)
        print(f"  {name:35} AUC = {auc:.4f}")

    # Correlation with top1_flat_cos
    print("\n" + "=" * 72)
    print("  CORRELATION WITH top1_flat_cos (rule out pure redundancy)")
    print("=" * 72)
    for name, vals in [
        ("top5_mean_cos", top5_mean_cos),
        ("max_centroid_cos", max_centroid_cos),
        ("leaf_size_np1", leaf_size_np1.astype(np.float32)),
        ("min_margin", min_margin),
        ("candidates_scored", candidates_scored.astype(np.float32)),
    ]:
        r = float(np.corrcoef(top1_cos, vals)[0, 1])
        print(f"  corr(top1_flat_cos, {name:25}) = {r:+.4f}")

    # Logistic regression: does dyf add anything beyond top1_cos?
    print("\n" + "=" * 72)
    print("  LOGISTIC REGRESSION — does dyf add predictive value?")
    print("=" * 72)

    feat_names_baseline = ["top1_flat_cos"]
    X_baseline = np.column_stack([top1_cos])
    res_baseline = logistic_regression(X_baseline, hits, feat_names_baseline)
    print(f"\n  BASELINE (top1_flat_cos only): AUC = {res_baseline['auc']:.4f}")
    for n, c in res_baseline["coefs"].items():
        print(f"    {n:25} coef = {c:+.4f}")

    feat_names_dyf_only = ["max_centroid_cos", "leaf_size_np1", "min_margin", "candidates_scored"]
    X_dyf_only = np.column_stack(
        [max_centroid_cos, leaf_size_np1.astype(np.float32), min_margin, candidates_scored.astype(np.float32)]
    )
    res_dyf = logistic_regression(X_dyf_only, hits, feat_names_dyf_only)
    print(f"\n  DYF ONLY: AUC = {res_dyf['auc']:.4f}")
    for n, c in res_dyf["coefs"].items():
        print(f"    {n:25} coef = {c:+.4f}")

    feat_names_full = ["top1_flat_cos", "max_centroid_cos", "leaf_size_np1", "min_margin", "candidates_scored"]
    X_full = np.column_stack(
        [top1_cos, max_centroid_cos, leaf_size_np1.astype(np.float32), min_margin, candidates_scored.astype(np.float32)]
    )
    res_full = logistic_regression(X_full, hits, feat_names_full)
    print(f"\n  FULL (top1 + dyf): AUC = {res_full['auc']:.4f}")
    for n, c in res_full["coefs"].items():
        print(f"    {n:25} coef = {c:+.4f}")

    delta = res_full["auc"] - res_baseline["auc"]
    print(f"\n  Δ AUC (full - baseline) = {delta:+.4f}")
    if delta > 0.01:
        print("  -> Dyf signals add measurable predictive value beyond top1_cos.")
    else:
        print("  -> Dyf signals are redundant with top1_cos at this corpus scale.")

    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        summary = {
            "pool_size": n_pool,
            "n_questions": len(query_embs),
            "hit_rate": float(hits.mean()),
            "solo_aucs": {
                "top1_flat_cos": solo_auc(top1_cos, hits),
                "top5_mean_cos": solo_auc(top5_mean_cos, hits),
                "max_centroid_cos": solo_auc(max_centroid_cos, hits),
                "leaf_size_np1": solo_auc(leaf_size_np1.astype(np.float32), hits),
                "min_margin": solo_auc(min_margin, hits),
                "candidates_scored": solo_auc(candidates_scored.astype(np.float32), hits),
            },
            "correlations_with_top1": {
                "top5_mean_cos": float(np.corrcoef(top1_cos, top5_mean_cos)[0, 1]),
                "max_centroid_cos": float(np.corrcoef(top1_cos, max_centroid_cos)[0, 1]),
                "leaf_size_np1": float(np.corrcoef(top1_cos, leaf_size_np1)[0, 1]),
                "min_margin": float(np.corrcoef(top1_cos, min_margin)[0, 1]),
                "candidates_scored": float(np.corrcoef(top1_cos, candidates_scored)[0, 1]),
            },
            "logistic_regression": {
                "baseline": res_baseline,
                "dyf_only": res_dyf,
                "full": res_full,
                "delta_auc": delta,
            },
        }
        with out_path.open("w") as f:
            json.dump(summary, f, indent=2)
        print(f"\n  wrote {out_path}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", default=DEFAULT_DATA)
    p.add_argument("--model", default="LOW", choices=["LOW", "MEDIUM", "MEDIUM_BGE", "HIGH"])
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--max-depth", type=int, default=4)
    p.add_argument("--num-bits", type=int, default=3)
    p.add_argument("--fit-method", default="raw_pca", choices=["raw_pca", "pca", "itq"])
    p.add_argument("--out", default=None)
    args = p.parse_args()
    run(args)


if __name__ == "__main__":
    main()
