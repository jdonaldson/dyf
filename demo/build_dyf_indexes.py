"""Build .dyf indexes from parquet files for demo datasets.

Usage:
    python demo/build_dyf_indexes.py

Builds indexes for:
    - wiki_simple_50k (1024d, BAAI/bge-large-en-v1.5)
    - gudid_50k_titled (384d, all-MiniLM-L6-v2)
    - cifar100_embeddings (512d, openai/clip-vit-base-patch32)

Output: demo/*.dyf files
"""

import json
import sys
import time
from pathlib import Path

import numpy as np
import polars as pl

# Add src to path for local dyf import
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from dyf.categorical import (
    CategoryGraph, store_category_graph,
    discover_categorical_columns, diagnose_axes,
)
from dyf.dyf_tree import build_dyf_tree
from dyf.fisher import extract_fisher_labels, compute_fisher_weights, apply_fisher_weights
from dyf.lazy_index import write_lazy_index

DEMO_DIR = Path(__file__).resolve().parent

DATASETS = [
    {
        "parquet": DEMO_DIR / "wiki_simple_50k.parquet",
        "output": DEMO_DIR / "wiki_simple_50k.dyf",
        "embedding_model": "BAAI/bge-large-en-v1.5",
        "title_col": "title",
    },
    {
        "parquet": DEMO_DIR / "gudid_50k_titled.parquet",
        "output": DEMO_DIR / "gudid_50k_titled.dyf",
        "embedding_model": "all-MiniLM-L6-v2",
        "title_col": "title",
    },
    {
        "parquet": DEMO_DIR / "cifar100_embeddings.parquet",
        "output": DEMO_DIR / "cifar100_embeddings.dyf",
        "embedding_model": "openai/clip-vit-base-patch32",
        "title_col": "title",
    },
]

MAX_DEPTH = 4
NUM_BITS = 4
MIN_LEAF_SIZE = 20
SEED = 42
FIT_METHOD = "itq"
QUANTIZATION = "float16"


def build_index(cfg: dict):
    parquet_path = cfg["parquet"]
    output_path = cfg["output"]
    name = parquet_path.stem

    if not parquet_path.exists():
        print(f"  SKIP {name}: {parquet_path} not found")
        return

    print(f"\n{'='*60}")
    print(f"Building {name}")
    print(f"{'='*60}")

    # Load parquet
    t0 = time.time()
    df = pl.read_parquet(parquet_path)
    print(f"  Loaded {len(df)} rows in {time.time()-t0:.1f}s")

    # Extract embeddings as float32 numpy
    embeddings = np.array(df["embedding"].to_list(), dtype=np.float32)
    print(f"  Embeddings shape: {embeddings.shape}")

    # Extract titles
    titles = df[cfg["title_col"]].to_list()

    # Optional axis diagnostics
    diagnose_parquet = cfg.get("diagnose_parquet")
    if diagnose_parquet:
        diag_path = Path(diagnose_parquet) if not isinstance(diagnose_parquet, Path) else diagnose_parquet
        if diag_path.exists():
            diag_df = pl.read_parquet(diag_path)
            label_cols = discover_categorical_columns(diag_df, text_col="text")
            if label_cols:
                diags = diagnose_axes(embeddings, label_cols)
                print(f"  Axis diagnostics ({len(diags)} axes):")
                for d in diags:
                    flag = " ⚠ UNDER-SERVED" if d.lift < 3.0 else ""
                    print(f"    {d.name}: lift={d.lift:.1f}x  "
                          f"purity={d.knn_purity:.3f}{flag}")
            else:
                print(f"  No categorical columns found in {diag_path}")
        else:
            print(f"  WARNING: diagnose_parquet={diag_path} not found, skipping")

    # Optional Fisher dimension weighting
    extra_meta = {}
    fisher_col = cfg.get("fisher_col")
    if fisher_col and fisher_col in df.columns:
        raw_vals = df[fisher_col].to_list()
        fisher_labels = extract_fisher_labels(raw_vals)
        fisher_weights = compute_fisher_weights(embeddings, fisher_labels)
        embeddings = apply_fisher_weights(embeddings, fisher_weights)
        extra_meta["fisher_col"] = fisher_col
        extra_meta["fisher_weights"] = json.dumps(fisher_weights.tolist())
        # Store a CategoryGraph for downstream multi-level use
        graph = CategoryGraph.from_single_level(fisher_labels)
        extra_meta.update(store_category_graph(graph, fisher_col))
        print(f"  Fisher weighting applied ({fisher_col}): "
              f"top-5 dims {np.argsort(fisher_weights)[-5:][::-1]}")

    # Build tree
    t0 = time.time()
    tree = build_dyf_tree(
        embeddings,
        max_depth=MAX_DEPTH,
        num_bits=NUM_BITS,
        min_leaf_size=MIN_LEAF_SIZE,
        seed=SEED,
        fit_method=FIT_METHOD,
    )
    print(f"  Tree built in {time.time()-t0:.1f}s")

    # Write .dyf
    t0 = time.time()
    meta = {"embedding_model": cfg["embedding_model"]}
    meta.update(extra_meta)
    write_lazy_index(
        tree,
        embeddings,
        str(output_path),
        compression="zstd",
        quantization=QUANTIZATION,
        metadata=meta,
        build_params={
            "max_depth": MAX_DEPTH,
            "num_bits": NUM_BITS,
            "min_leaf_size": MIN_LEAF_SIZE,
            "seed": SEED,
        },
        stored_fields={"title": titles},
    )
    size_mb = output_path.stat().st_size / (1024 * 1024)
    print(f"  Written {output_path.name} ({size_mb:.1f} MB) in {time.time()-t0:.1f}s")


def main():
    print("Building DYF indexes from parquet files")
    print(f"  max_depth={MAX_DEPTH}, num_bits={NUM_BITS}, "
          f"fit_method={FIT_METHOD}, quantization={QUANTIZATION}")

    for cfg in DATASETS:
        build_index(cfg)

    print(f"\nDone. Built indexes in {DEMO_DIR}")


if __name__ == "__main__":
    main()
