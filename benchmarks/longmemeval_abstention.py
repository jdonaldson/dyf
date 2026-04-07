"""LongMemEval abstention layer: can dyf + top1_cos gate low-confidence queries?

Follow-up to `longmemeval_diagnostics.py`. That experiment showed the full
classifier (top1_flat_cos + dyf routing signals) hits AUC 0.727 vs a 0.639
baseline for predicting whether flat retrieval lands a gold session in top-5.

The natural UX question: if we trust the classifier to gate which queries are
worth answering vs. which to abstain on, does recall on the *kept* queries
actually go up? That would convert AUC into a concrete guarantee — "on the
queries we answer, we hit X% recall" — which is what a real memory system needs.

Method:
  1. Build the same features as diagnostics (top1_cos + 4 dyf signals).
  2. Get honest, out-of-sample predicted probabilities via 5-fold CV.
  3. Sort queries by predicted probability.
  4. At each abstention budget (0, 10, 20, 25, 33, 50%), compute:
       - recall on kept queries (the promise to the user)
       - recall on abstained queries (did we actually catch the misses?)
       - per-type breakdown on kept
  5. Compare to two baselines:
       - uniform (answer everything, recall = hit_rate)
       - top1-cos-only gate (what flat retrieval alone can decide)

Why CV: fitting + evaluating logistic regression on the same 500 points
overstates by ~1-2pp. Honest abstention needs honest gate probabilities.

Usage:
    python benchmarks/longmemeval_abstention.py \
        --out benchmarks/results/longmemeval_abstention.json
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
BUDGETS = [0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.33, 0.50]
# Precision@coverage: what if we keep ONLY the top X% most confident queries?
COVERAGES = [0.05, 0.10, 0.20, 0.25, 0.33, 0.50, 0.75, 1.00]


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


def cv_probs(X: np.ndarray, y: np.ndarray, n_splits: int = 5, seed: int = 42) -> np.ndarray:
    """Out-of-fold predicted probabilities for honest abstention scoring."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedKFold
    from sklearn.preprocessing import StandardScaler

    probs = np.empty(len(y), dtype=np.float64)
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    for train_idx, test_idx in skf.split(X, y):
        scaler = StandardScaler()
        X_tr = scaler.fit_transform(X[train_idx])
        X_te = scaler.transform(X[test_idx])
        model = LogisticRegression(max_iter=2000, C=1.0)
        model.fit(X_tr, y[train_idx])
        probs[test_idx] = model.predict_proba(X_te)[:, 1]
    return probs


def evaluate_abstention(
    probs: np.ndarray,
    hits: np.ndarray,
    qtypes: list[str],
    budgets: list[float],
) -> list[dict]:
    """For each budget, abstain on the lowest-probability fraction. Report kept/abstained recall."""
    n = len(probs)
    order = np.argsort(probs)  # ascending: lowest conf first
    rows = []
    unique_types = sorted(set(qtypes))
    types_arr = np.array(qtypes)
    for b in budgets:
        k_abstain = int(round(b * n))
        abstain_idx = order[:k_abstain]
        kept_idx = order[k_abstain:]
        kept_hits = hits[kept_idx]
        abstain_hits = hits[abstain_idx]
        row = {
            "budget": b,
            "n_kept": int(len(kept_idx)),
            "n_abstain": int(len(abstain_idx)),
            "recall_kept": float(kept_hits.mean()) if len(kept_hits) else 0.0,
            "recall_abstain": float(abstain_hits.mean()) if len(abstain_hits) else 0.0,
            "misses_caught_frac": (
                float((1 - abstain_hits).sum() / (1 - hits).sum())
                if (1 - hits).sum() > 0
                else 0.0
            ),
            "per_type_kept": {},
        }
        for t in unique_types:
            tmask = types_arr[kept_idx] == t
            if tmask.sum() > 0:
                row["per_type_kept"][t] = {
                    "n": int(tmask.sum()),
                    "recall": float(kept_hits[tmask].mean()),
                }
            else:
                row["per_type_kept"][t] = {"n": 0, "recall": 0.0}
        rows.append(row)
    return rows


def print_abstention_table(rows: list[dict], label: str) -> None:
    print(f"\n  {label}")
    print(
        f"  {'budget':>7} {'n_kept':>7} {'n_abs':>6} "
        f"{'R@5 kept':>10} {'R@5 abs':>10} {'misses caught':>14}"
    )
    for r in rows:
        print(
            f"  {r['budget']:>7.0%} {r['n_kept']:>7d} {r['n_abstain']:>6d} "
            f"{r['recall_kept']:>10.4f} {r['recall_abstain']:>10.4f} "
            f"{r['misses_caught_frac']:>13.1%}"
        )


def evaluate_coverage(
    probs: np.ndarray, hits: np.ndarray, coverages: list[float]
) -> list[dict]:
    """At each coverage, keep only the top X% most confident. Report recall on kept."""
    n = len(probs)
    order = np.argsort(-probs)  # descending: highest conf first
    rows = []
    for c in coverages:
        k_keep = int(round(c * n))
        if k_keep == 0:
            rows.append({"coverage": c, "n_kept": 0, "recall": 0.0})
            continue
        kept = order[:k_keep]
        rows.append(
            {
                "coverage": c,
                "n_kept": int(k_keep),
                "recall": float(hits[kept].mean()),
            }
        )
    return rows


def print_coverage_table(rows: list[dict], label: str) -> None:
    print(f"\n  {label}")
    print(f"  {'coverage':>9} {'n_kept':>7} {'R@5':>8}")
    for r in rows:
        print(f"  {r['coverage']:>9.0%} {r['n_kept']:>7d} {r['recall']:>8.4f}")


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
    qtypes = [e.get("question_type", "unknown") for e, _ in entries]
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

    if args.pca > 0:
        from sklearn.decomposition import PCA
        print(f"\nPCA distillation: {pool_embs.shape[1]}d → {args.pca}d")
        t0 = time.time()
        pca = PCA(n_components=args.pca, random_state=42)
        pool_embs = pca.fit_transform(pool_embs).astype(np.float32)
        query_embs = pca.transform(query_embs).astype(np.float32)
        # Re-normalize: PCA doesn't preserve unit norm
        pool_embs = l2_normalize(pool_embs)
        query_embs = l2_normalize(query_embs)
        var_explained = float(pca.explained_variance_ratio_.sum())
        print(f"  fit+transform: {time.time() - t0:.1f}s")
        print(f"  variance retained: {var_explained:.4f}")

    # -------------------------------------------------------------------------
    # Flat features + gold labels
    # -------------------------------------------------------------------------
    print("\nComputing flat retrieval signals...")
    nq = len(query_embs)
    top1_cos = np.empty(nq, dtype=np.float32)
    top5_mean_cos = np.empty(nq, dtype=np.float32)
    hits = np.empty(nq, dtype=np.int32)

    sims_all = query_embs @ pool_embs.T  # (Q, N)
    for i in range(nq):
        sims = sims_all[i]
        top_k_idx = np.argpartition(-sims, K)[:K]
        top_k_idx = top_k_idx[np.argsort(-sims[top_k_idx])]
        top1_cos[i] = sims[top_k_idx[0]]
        top5_mean_cos[i] = sims[top_k_idx].mean()
        top_ids = {pool_ids[j] for j in top_k_idx}
        hits[i] = int(bool(top_ids & gold_sets[i]))
    hit_rate = float(hits.mean())
    print(f"  hit rate (recall_any@5): {hit_rate:.4f}")

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
        index_path = os.path.join(tmp, "abstain.dyf")
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

        _ = idx.search_ivf(query_embs[0], k=1, nprobe=1)  # force centroid build
        centroids = idx._centroid_matrix
        print(f"  tree has {len(centroids)} leaves")

        max_centroid_cos = np.empty(nq, dtype=np.float32)
        leaf_size_np1 = np.empty(nq, dtype=np.int32)
        min_margin = np.empty(nq, dtype=np.float32)
        candidates_scored = np.empty(nq, dtype=np.int32)

        max_centroid_cos[:] = (query_embs @ centroids.T).max(axis=1)

        print("Collecting routing diagnostics...")
        for i, q in enumerate(query_embs):
            r1 = idx.search(q, k=K, nprobe=1, return_routing=True)
            routing1 = r1.routing or {}
            leaf_size_np1[i] = routing1.get("candidates_scored", 0)
            r3 = idx.search(q, k=K, nprobe=3, return_routing=True)
            routing3 = r3.routing or {}
            mm = routing3.get("min_margin")
            min_margin[i] = mm if mm is not None else 0.0
            candidates_scored[i] = routing3.get("candidates_scored", 0)
        print("  done")

    # -------------------------------------------------------------------------
    # Out-of-fold predicted probabilities (CV so abstention is honest)
    # -------------------------------------------------------------------------
    print("\n5-fold CV predicted probabilities...")

    X_baseline = np.column_stack([top1_cos]).astype(np.float64)
    X_top5 = np.column_stack([top1_cos, top5_mean_cos]).astype(np.float64)
    X_full = np.column_stack(
        [
            top1_cos,
            max_centroid_cos,
            leaf_size_np1.astype(np.float32),
            min_margin,
            candidates_scored.astype(np.float32),
        ]
    ).astype(np.float64)

    from sklearn.metrics import roc_auc_score

    probs_baseline = cv_probs(X_baseline, hits, n_splits=5, seed=42)
    probs_top5 = cv_probs(X_top5, hits, n_splits=5, seed=42)
    probs_full = cv_probs(X_full, hits, n_splits=5, seed=42)

    auc_baseline_cv = float(roc_auc_score(hits, probs_baseline))
    auc_top5_cv = float(roc_auc_score(hits, probs_top5))
    auc_full_cv = float(roc_auc_score(hits, probs_full))
    print(f"  baseline (top1 only)    out-of-fold AUC = {auc_baseline_cv:.4f}")
    print(f"  top1+top5_mean          out-of-fold AUC = {auc_top5_cv:.4f}")
    print(f"  full (top1+dyf)         out-of-fold AUC = {auc_full_cv:.4f}")
    print(f"  Δ AUC (top5 - baseline) = {auc_top5_cv - auc_baseline_cv:+.4f}")
    print(f"  Δ AUC (full - baseline) = {auc_full_cv - auc_baseline_cv:+.4f}")

    # -------------------------------------------------------------------------
    # Abstention tables
    # -------------------------------------------------------------------------
    print("\n" + "=" * 72)
    print(f"  ABSTENTION — uniform baseline hit rate: {hit_rate:.4f}")
    print("=" * 72)

    rows_baseline = evaluate_abstention(probs_baseline, hits, qtypes, BUDGETS)
    rows_top5 = evaluate_abstention(probs_top5, hits, qtypes, BUDGETS)
    rows_full = evaluate_abstention(probs_full, hits, qtypes, BUDGETS)

    print_abstention_table(
        rows_baseline, "BASELINE gate (top1_flat_cos only)"
    )
    print_abstention_table(
        rows_top5, "TOP5 gate (top1 + top5_mean_cos, no dyf)"
    )
    print_abstention_table(rows_full, "FULL gate (top1 + dyf signals)")

    # Precision@coverage: what happens when we keep only the most confident
    # fraction? This is the "high-confidence slice" view, where a low AUC
    # can still buy a high-precision subset.
    print("\n" + "=" * 72)
    print("  PRECISION@COVERAGE — recall on the most-confident N% of queries")
    print("=" * 72)
    cov_baseline = evaluate_coverage(probs_baseline, hits, COVERAGES)
    cov_top5 = evaluate_coverage(probs_top5, hits, COVERAGES)
    cov_full = evaluate_coverage(probs_full, hits, COVERAGES)
    print_coverage_table(cov_baseline, "BASELINE gate")
    print_coverage_table(cov_top5, "TOP5 gate")
    print_coverage_table(cov_full, "FULL gate")

    # Head-to-head at 25% budget
    b25_base = rows_baseline[BUDGETS.index(0.25)]
    b25_full = rows_full[BUDGETS.index(0.25)]
    print("\n" + "=" * 72)
    print("  HEAD TO HEAD AT 25% ABSTENTION BUDGET")
    print("=" * 72)
    print(
        f"  uniform (no abstention):     recall = {hit_rate:.4f}  "
        f"(n=500)"
    )
    print(
        f"  baseline gate (top1 only):   recall = {b25_base['recall_kept']:.4f}  "
        f"(n={b25_base['n_kept']}) misses caught = {b25_base['misses_caught_frac']:.1%}"
    )
    print(
        f"  full gate (top1 + dyf):      recall = {b25_full['recall_kept']:.4f}  "
        f"(n={b25_full['n_kept']}) misses caught = {b25_full['misses_caught_frac']:.1%}"
    )
    full_gain_vs_uniform = b25_full["recall_kept"] - hit_rate
    full_gain_vs_baseline = b25_full["recall_kept"] - b25_base["recall_kept"]
    print(
        f"\n  full gate uplift vs uniform:      {full_gain_vs_uniform:+.4f}"
    )
    print(
        f"  full gate uplift over top1-only:  {full_gain_vs_baseline:+.4f}"
    )

    # Per-type recall at 25%
    print("\n  PER-TYPE recall at 25% abstention (full gate):")
    per_type_uniform: dict[str, tuple[int, float]] = {}
    types_arr = np.array(qtypes)
    for t in sorted(set(qtypes)):
        mask = types_arr == t
        per_type_uniform[t] = (int(mask.sum()), float(hits[mask].mean()))
    for t in sorted(set(qtypes)):
        kept = b25_full["per_type_kept"][t]
        n_uni, r_uni = per_type_uniform[t]
        print(
            f"    {t:30} uniform: {r_uni:.3f} (n={n_uni:3d}) -> "
            f"kept: {kept['recall']:.3f} (n={kept['n']:3d})"
        )

    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        summary = {
            "pool_size": n_pool,
            "n_questions": nq,
            "hit_rate_uniform": hit_rate,
            "cv_auc": {
                "baseline": auc_baseline_cv,
                "top5": auc_top5_cv,
                "full": auc_full_cv,
                "delta_top5_vs_baseline": auc_top5_cv - auc_baseline_cv,
                "delta_full_vs_baseline": auc_full_cv - auc_baseline_cv,
            },
            "abstention": {
                "baseline_top1_only": rows_baseline,
                "top5_top1_plus_top5mean": rows_top5,
                "full_top1_plus_dyf": rows_full,
            },
            "coverage": {
                "baseline_top1_only": cov_baseline,
                "top5_top1_plus_top5mean": cov_top5,
                "full_top1_plus_dyf": cov_full,
            },
            "budgets": BUDGETS,
            "coverages": COVERAGES,
            "config": {
                "model": args.model,
                "max_depth": args.max_depth,
                "num_bits": args.num_bits,
                "fit_method": args.fit_method,
                "pca": args.pca,
                "n_splits": 5,
                "seed": 42,
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
    p.add_argument("--pca", type=int, default=0, help="PCA-distill pool embeddings to N dims (0 = off)")
    p.add_argument("--out", default=None)
    args = p.parse_args()
    run(args)


if __name__ == "__main__":
    main()
