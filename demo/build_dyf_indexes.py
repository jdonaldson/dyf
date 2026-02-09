"""Build .dyf indexes from parquet files for demo datasets.

Usage:
    python demo/build_dyf_indexes.py

Builds indexes for:
    - wiki_simple_50k (1024d, BAAI/bge-large-en-v1.5)
    - gudid_50k_titled (384d, all-MiniLM-L6-v2)
    - cifar100_embeddings (512d, openai/clip-vit-base-patch32)

Output: demo/*.dyf files
"""

import sys
import time
from pathlib import Path

import numpy as np
import polars as pl

# Add src to path for local dyf import
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from dyf.dyf_tree import build_dyf_tree
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
    write_lazy_index(
        tree,
        embeddings,
        str(output_path),
        compression="zstd",
        quantization=QUANTIZATION,
        metadata={"embedding_model": cfg["embedding_model"]},
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
