"""LongMemEval flat vs dyf tree: retrieval cost at scale.

Builds a single pooled corpus of all ~19K unique sessions across LongMemEval's
500 questions, embeds with MiniLM, then compares:

  - flat: brute-force cosine top-k (numpy dot product)
  - dyf.search: tree-walk LazyIndex at various nprobe
  - dyf.search_ivf: centroid-routed LazyIndex at various nprobe

Metrics:
  - recall_any@5 on gold answer_session_ids (task metric, comparable to
    longmemeval_flat.py)
  - recall@5 vs flat ground-truth ranking (ANN fidelity)
  - p50 / p95 latency per query
  - build time, index file size

Dataset:
    /tmp/longmemeval-data/longmemeval_s_cleaned.json
    (see longmemeval_flat.py for download)

Usage:
    python benchmarks/longmemeval_tree.py
    python benchmarks/longmemeval_tree.py --max-depth 5
    python benchmarks/longmemeval_tree.py --out benchmarks/results/tree.json
"""

from __future__ import annotations

import argparse
import json
import os
import tempfile
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

from dyf import EmbedderConfig, LazyIndex, build_dyf_tree, write_lazy_index

DEFAULT_DATA = "/tmp/longmemeval-data/longmemeval_s_cleaned.json"
K = 5


def build_pool(data: list[dict]) -> tuple[list[str], list[str], dict[str, int]]:
    """Pool unique sessions across all questions.

    Returns:
        pool_texts:  list of session docs (user turns joined)
        pool_ids:    parallel list of session_ids
        id_to_idx:   session_id -> pool index
    """
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


def recall_any(top_ids: list[str], correct: set[str]) -> float:
    return float(bool(set(top_ids) & correct))


def compute_latency(search_fn, queries: np.ndarray, warmup: int = 20) -> dict:
    for i in range(min(warmup, len(queries))):
        search_fn(queries[i])
    latencies = []
    for q in queries:
        t0 = time.perf_counter()
        search_fn(q)
        latencies.append(time.perf_counter() - t0)
    lat_ms = np.array(latencies) * 1000.0
    return {
        "p50_ms": float(np.percentile(lat_ms, 50)),
        "p95_ms": float(np.percentile(lat_ms, 95)),
        "p99_ms": float(np.percentile(lat_ms, 99)),
        "mean_ms": float(lat_ms.mean()),
        "qps": float(len(queries) / (lat_ms.sum() / 1000.0)),
    }


def run(args: argparse.Namespace) -> None:
    data_path = Path(args.data)
    if not data_path.exists():
        raise SystemExit(f"Dataset not found at {data_path}. See longmemeval_flat.py header.")

    print(f"Loading {data_path}...")
    with data_path.open() as f:
        data = json.load(f)
    if args.limit > 0:
        data = data[: args.limit]
    print(f"  {len(data)} questions")

    print("Pooling unique sessions...")
    pool_texts, pool_ids, id_to_idx = build_pool(data)
    n_pool = len(pool_texts)
    print(f"  pool size: {n_pool} unique session docs")

    # Filter to questions whose gold IDs all exist in the pool
    entries = []
    for e in data:
        gold = [g for g in e["answer_session_ids"] if g in id_to_idx]
        if gold:
            entries.append((e, gold))
    print(f"  {len(entries)} questions with resolvable gold IDs")

    questions = [e["question"] for e, _ in entries]
    qtypes = [e.get("question_type", "unknown") for e, _ in entries]
    gold_sets = [set(g) for _, g in entries]

    embed_cfg = getattr(EmbedderConfig, args.model)
    print(f"Embedder: {embed_cfg.name} ({embed_cfg.model_id}, {embed_cfg.dim}d)")

    print(f"Embedding {n_pool} pool docs + {len(questions)} queries...")
    t0 = time.time()
    all_texts = pool_texts + questions
    all_embs = embed_cfg.embed(all_texts, batch_size=64, verbose=True)
    all_embs = l2_normalize(all_embs.astype(np.float32))
    t_embed = time.time() - t0
    print(f"  embed done in {t_embed:.1f}s, shape={all_embs.shape}")

    pool_embs = all_embs[:n_pool]
    query_embs = all_embs[n_pool:]

    # -------------------------------------------------------------------------
    # Flat baseline (brute-force dot product)
    # -------------------------------------------------------------------------
    print("\n== FLAT (brute-force dot product) ==")

    def flat_search(q: np.ndarray) -> np.ndarray:
        sims = pool_embs @ q
        top = np.argpartition(-sims, K)[:K]
        return top[np.argsort(-sims[top])]

    flat_latency = compute_latency(flat_search, query_embs)
    print(f"  latency: p50={flat_latency['p50_ms']:.3f}ms  p95={flat_latency['p95_ms']:.3f}ms  qps={flat_latency['qps']:.0f}")

    print("  scoring recall_any@5 on gold...")
    flat_top_indices = np.empty((len(query_embs), K), dtype=np.int64)
    for i, q in enumerate(query_embs):
        flat_top_indices[i] = flat_search(q)

    flat_any5 = 0.0
    flat_per_type: dict[str, list[float]] = defaultdict(list)
    for i in range(len(query_embs)):
        top_ids = [pool_ids[j] for j in flat_top_indices[i]]
        hit = recall_any(top_ids, gold_sets[i])
        flat_any5 += hit
        flat_per_type[qtypes[i]].append(hit)
    flat_any5 /= len(query_embs)
    print(f"  recall_any@5 (gold) = {flat_any5:.4f}")

    # -------------------------------------------------------------------------
    # Dyf tree
    # -------------------------------------------------------------------------
    print("\n== DYF TREE ==")
    print(f"  Building tree (max_depth={args.max_depth}, num_bits={args.num_bits}, fit={args.fit_method})...")
    t0 = time.time()
    tree = build_dyf_tree(
        pool_embs,
        max_depth=args.max_depth,
        num_bits=args.num_bits,
        min_leaf_size=8,
        seed=42,
        fit_method=args.fit_method,
    )
    t_tree = time.time() - t0
    print(f"  tree built in {t_tree:.1f}s")

    with tempfile.TemporaryDirectory() as tmp:
        index_path = os.path.join(tmp, "lme.dyf")
        t0 = time.time()
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
        t_write = time.time() - t0
        size_mb = os.path.getsize(index_path) / (1024 * 1024)
        print(f"  index written in {t_write:.1f}s, size={size_mb:.1f}MB")

        idx = LazyIndex(index_path)

        all_variants = []

        for mode_name, search_method in [("search", idx.search), ("search_ivf", idx.search_ivf)]:
            for nprobe in args.nprobes:
                label = f"dyf.{mode_name}(nprobe={nprobe})"
                print(f"\n  {label}")

                def _fn(q, _m=search_method, _n=nprobe):
                    return _m(q, k=K, nprobe=_n)

                lat = compute_latency(_fn, query_embs)
                print(f"    latency: p50={lat['p50_ms']:.3f}ms  p95={lat['p95_ms']:.3f}ms  qps={lat['qps']:.0f}")

                any5 = 0.0
                fid5 = 0.0
                per_type: dict[str, list[float]] = defaultdict(list)
                for i, q in enumerate(query_embs):
                    result = search_method(q, k=K, nprobe=nprobe)
                    res_indices = np.asarray(result[0])
                    top_ids = [pool_ids[int(j)] for j in res_indices[:K]]
                    hit = recall_any(top_ids, gold_sets[i])
                    any5 += hit
                    per_type[qtypes[i]].append(hit)

                    flat_top = set(int(j) for j in flat_top_indices[i])
                    res_top = set(int(j) for j in res_indices[:K])
                    fid5 += len(flat_top & res_top) / K
                any5 /= len(query_embs)
                fid5 /= len(query_embs)

                print(f"    recall_any@5 (gold)  = {any5:.4f}")
                print(f"    recall@5   vs flat   = {fid5:.4f}")
                all_variants.append(
                    {
                        "label": label,
                        "mode": mode_name,
                        "nprobe": nprobe,
                        "recall_any_5": any5,
                        "fidelity_5": fid5,
                        "latency": lat,
                    }
                )

    # -------------------------------------------------------------------------
    # Summary
    # -------------------------------------------------------------------------
    print("\n" + "=" * 72)
    print(f"  SUMMARY — LongMemEval pool N={n_pool}, Q={len(query_embs)}, k={K}")
    print("=" * 72)
    print(f"  {'method':35} {'R_any@5':>9} {'fid@5':>8} {'p50ms':>9} {'p95ms':>9} {'qps':>8}")
    print(f"  {'flat brute-force':35} {flat_any5:>9.4f} {'1.0000':>8} {flat_latency['p50_ms']:>9.3f} {flat_latency['p95_ms']:>9.3f} {flat_latency['qps']:>8.0f}")
    for v in all_variants:
        print(
            f"  {v['label']:35} {v['recall_any_5']:>9.4f} {v['fidelity_5']:>8.4f} "
            f"{v['latency']['p50_ms']:>9.3f} {v['latency']['p95_ms']:>9.3f} "
            f"{v['latency']['qps']:>8.0f}"
        )

    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        summary = {
            "pool_size": n_pool,
            "n_questions": len(query_embs),
            "model": embed_cfg.name,
            "model_id": embed_cfg.model_id,
            "build": {
                "tree_s": t_tree,
                "write_s": t_write,
                "file_mb": size_mb,
                "max_depth": args.max_depth,
                "num_bits": args.num_bits,
                "fit_method": args.fit_method,
            },
            "flat": {
                "recall_any_5": flat_any5,
                "latency": flat_latency,
                "per_type": {
                    qt: float(np.mean(flat_per_type[qt])) for qt in flat_per_type
                },
            },
            "dyf": all_variants,
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
    p.add_argument("--nprobes", type=int, nargs="+", default=[1, 3, 10])
    p.add_argument("--out", default=None)
    args = p.parse_args()
    run(args)


if __name__ == "__main__":
    main()
