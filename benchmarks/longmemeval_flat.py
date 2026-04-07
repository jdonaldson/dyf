"""LongMemEval flat-retrieval baseline for dyf.

Reproduces mempalace's "raw mode" headline result against dyf's embedding
stack, with no ChromaDB dependency. One doc per session (user turns joined),
embed, cosine similarity top-k, report recall@k.

This is a *calibration* benchmark, not a dyf tree test. It answers:

  "What does flat retrieval look like on our current embedding stack?"

Later experiments can compare dyf tree-walk retrieval to this baseline.

Dataset:
    curl -fsSL -o /tmp/longmemeval-data/longmemeval_s_cleaned.json \\
      https://huggingface.co/datasets/xiaowu0162/longmemeval-cleaned/resolve/main/longmemeval_s_cleaned.json

Usage:
    python benchmarks/longmemeval_flat.py
    python benchmarks/longmemeval_flat.py --include-assistant
    python benchmarks/longmemeval_flat.py --model MEDIUM_BGE
    python benchmarks/longmemeval_flat.py --limit 50
"""

from __future__ import annotations

import argparse
import json
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

from dyf import EmbedderConfig

DEFAULT_DATA = "/tmp/longmemeval-data/longmemeval_s_cleaned.json"
KS = (1, 5, 10)


def build_session_doc(session: list[dict], include_assistant: bool) -> str:
    """Join turns in a session into a single document string.

    mempalace's winning config keeps only user turns. We expose a flag to
    toggle this so we can measure how much that trick contributes.
    """
    if include_assistant:
        turns = [t["content"] for t in session]
    else:
        turns = [t["content"] for t in session if t["role"] == "user"]
    return "\n".join(turns)


def build_all_texts(
    data: list[dict], include_assistant: bool
) -> tuple[list[str], list[tuple[int, int, list[str]]]]:
    """Flatten dataset into a single list of texts for one big embed call.

    Returns:
        all_texts: every session doc followed by every question, in order
        slices: list of (corpus_start, corpus_end, correct_ids) per question;
                the query index is implicit (corpus_end)
    """
    all_texts: list[str] = []
    slices: list[tuple[int, int, list[str]]] = []

    for entry in data:
        corpus_start = len(all_texts)
        corpus_ids: list[str] = []

        for session, sess_id in zip(
            entry["haystack_sessions"], entry["haystack_session_ids"]
        ):
            doc = build_session_doc(session, include_assistant)
            if not doc.strip():
                continue
            all_texts.append(doc)
            corpus_ids.append(sess_id)

        corpus_end = len(all_texts)
        all_texts.append(entry["question"])  # query sits right after corpus

        slices.append((corpus_start, corpus_end, entry["answer_session_ids"]))
        # Stash corpus_ids on the slice via a parallel structure would be
        # cleaner; we pass them through a side dict keyed by question index.
        _sess_id_map[len(slices) - 1] = corpus_ids

    return all_texts, slices


# Side channel: question_index -> list[session_id] for the corpus slice.
# Keeps the slices tuple tidy without inventing a dataclass for a script.
_sess_id_map: dict[int, list[str]] = {}


def l2_normalize(x: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return x / norms


def recall_any(ranked_ids: list[str], correct: set[str], k: int) -> float:
    top = set(ranked_ids[:k])
    return float(bool(top & correct))


def recall_all(ranked_ids: list[str], correct: set[str], k: int) -> float:
    top = set(ranked_ids[:k])
    return float(correct.issubset(top))


def run(args: argparse.Namespace) -> None:
    data_path = Path(args.data)
    if not data_path.exists():
        raise SystemExit(
            f"Dataset not found at {data_path}. Download with:\n"
            "  mkdir -p /tmp/longmemeval-data && curl -fsSL -o "
            "/tmp/longmemeval-data/longmemeval_s_cleaned.json "
            "https://huggingface.co/datasets/xiaowu0162/longmemeval-cleaned"
            "/resolve/main/longmemeval_s_cleaned.json"
        )

    print(f"Loading {data_path}...")
    with data_path.open() as f:
        data = json.load(f)

    if args.limit > 0:
        data = data[: args.limit]
    print(f"  {len(data)} questions")

    embed_cfg = getattr(EmbedderConfig, args.model)
    print(f"Embedder: {embed_cfg.name} ({embed_cfg.model_id}, {embed_cfg.dim}d)")
    print(f"Mode: {'user+assistant' if args.include_assistant else 'user-only'}")

    print("Building texts...")
    all_texts, slices = build_all_texts(data, args.include_assistant)
    n_corpus = sum(end - start for start, end, _ in slices)
    n_queries = len(slices)
    print(f"  {n_corpus} session docs + {n_queries} queries = {len(all_texts)} texts")

    print("Embedding...")
    t0 = time.time()
    embeddings = embed_cfg.embed(all_texts, batch_size=64, verbose=True)
    embeddings = l2_normalize(embeddings.astype(np.float32))
    print(f"  done in {time.time() - t0:.1f}s, shape={embeddings.shape}")

    print("Scoring...")
    metrics = {f"recall_any@{k}": [] for k in KS}
    metrics.update({f"recall_all@{k}": [] for k in KS})
    per_type: dict[str, dict[str, list[float]]] = defaultdict(
        lambda: {f"recall_any@{k}": [] for k in KS}
    )

    for q_idx, (corpus_start, corpus_end, correct_list) in enumerate(slices):
        corpus_ids = _sess_id_map[q_idx]
        if not corpus_ids:
            # Degenerate: no valid session docs; skip but still count as miss
            for k in KS:
                metrics[f"recall_any@{k}"].append(0.0)
                metrics[f"recall_all@{k}"].append(0.0)
            continue

        corpus_emb = embeddings[corpus_start:corpus_end]
        query_emb = embeddings[corpus_end]
        sims = corpus_emb @ query_emb
        order = np.argsort(-sims)
        ranked_ids = [corpus_ids[i] for i in order]
        correct = set(correct_list)

        qtype = data[q_idx].get("question_type", "unknown")
        for k in KS:
            ra = recall_any(ranked_ids, correct, k)
            rl = recall_all(ranked_ids, correct, k)
            metrics[f"recall_any@{k}"].append(ra)
            metrics[f"recall_all@{k}"].append(rl)
            per_type[qtype][f"recall_any@{k}"].append(ra)

    print()
    print("=" * 60)
    print(f"  LongMemEval flat baseline — {embed_cfg.name}")
    print(f"  n={len(slices)}, mode={'user+asst' if args.include_assistant else 'user-only'}")
    print("=" * 60)
    for k in KS:
        ra = float(np.mean(metrics[f"recall_any@{k}"]))
        rl = float(np.mean(metrics[f"recall_all@{k}"]))
        print(f"  recall_any@{k:<2} = {ra:.4f}    recall_all@{k:<2} = {rl:.4f}")

    print()
    print("  PER-TYPE BREAKDOWN (recall_any@5):")
    for qtype in sorted(per_type):
        vals = per_type[qtype]["recall_any@5"]
        ra = float(np.mean(vals))
        print(f"    {qtype:30} {ra:.4f}  (n={len(vals)})")

    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        summary = {
            "model": embed_cfg.name,
            "model_id": embed_cfg.model_id,
            "dim": embed_cfg.dim,
            "include_assistant": args.include_assistant,
            "n_questions": len(slices),
            **{
                f"recall_any@{k}": float(np.mean(metrics[f"recall_any@{k}"]))
                for k in KS
            },
            **{
                f"recall_all@{k}": float(np.mean(metrics[f"recall_all@{k}"]))
                for k in KS
            },
            "per_type": {
                qtype: {
                    "recall_any@5": float(np.mean(vals["recall_any@5"])),
                    "n": len(vals["recall_any@5"]),
                }
                for qtype, vals in per_type.items()
            },
        }
        with out_path.open("w") as f:
            json.dump(summary, f, indent=2)
        print(f"\n  wrote {out_path}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", default=DEFAULT_DATA, help="Path to longmemeval_s_cleaned.json")
    p.add_argument(
        "--model",
        default="LOW",
        choices=["LOW", "MEDIUM", "MEDIUM_BGE", "HIGH"],
        help="EmbedderConfig preset. LOW=MiniLM-L6-v2 (default, matches ChromaDB default).",
    )
    p.add_argument(
        "--include-assistant",
        action="store_true",
        help="Include assistant turns in session docs (mempalace drops them).",
    )
    p.add_argument("--limit", type=int, default=0, help="Limit to first N questions (0=all)")
    p.add_argument("--out", default=None, help="Optional JSON summary output path")
    args = p.parse_args()
    run(args)


if __name__ == "__main__":
    main()
