"""Build a .dyf index from the MNIST dataset (70K handwritten digits).

Usage:
    python demo/build_mnist.py

Embeddings: raw 784d pixel values (28x28), normalized to [0, 1].
Output: demo/mnist_70k.dyf
"""

import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from dyf.dyf_tree import build_dyf_tree
from dyf.lazy_index import write_lazy_index
from dyf.provenance import create_provenance, provenance_to_dict

DEMO_DIR = Path(__file__).resolve().parent
OUTPUT = DEMO_DIR / "mnist_70k.dyf"

MAX_DEPTH = 4
NUM_BITS = 4
MIN_LEAF_SIZE = 20
SEED = 42
FIT_METHOD = "itq"
QUANTIZATION = "float16"


def main():
    from sklearn.datasets import fetch_openml

    print("Fetching MNIST...")
    t0 = time.time()
    mnist = fetch_openml("mnist_784", version=1, as_frame=False, parser="auto")
    print(f"  Loaded in {time.time() - t0:.1f}s")

    # Pixel values normalized to [0, 1]
    embeddings = mnist.data.astype(np.float32) / 255.0
    labels = mnist.target.astype(int)
    n = len(embeddings)
    print(f"  {n} samples, {embeddings.shape[1]}d")

    # Titles: "Digit 7 (#12345)"
    titles = [f"Digit {labels[i]} (#{i})" for i in range(n)]

    # Build tree
    print("Building tree...")
    t0 = time.time()
    tree = build_dyf_tree(
        embeddings,
        max_depth=MAX_DEPTH,
        num_bits=NUM_BITS,
        min_leaf_size=MIN_LEAF_SIZE,
        seed=SEED,
        fit_method=FIT_METHOD,
    )
    print(f"  Tree built in {time.time() - t0:.1f}s")

    # Write .dyf
    print("Writing .dyf...")
    t0 = time.time()
    meta = {
        "embedding_model": "raw_pixels_784d",
        "dataset": "mnist_784",
        "dataset_version": "1",
        "domain": "handwritten digit images",
        "_provenance": json.dumps(provenance_to_dict(
            create_provenance(
                artifact_type="dyf",
                n_items=n,
                source_paths=["sklearn.datasets.fetch_openml('mnist_784')"],
                params={
                    "max_depth": MAX_DEPTH,
                    "num_bits": NUM_BITS,
                    "min_leaf_size": MIN_LEAF_SIZE,
                    "seed": SEED,
                    "embedding_model": "raw_pixels_784d",
                    "quantization": QUANTIZATION,
                },
            )
        )),
    }

    write_lazy_index(
        tree,
        embeddings,
        str(OUTPUT),
        compression="zstd",
        quantization=QUANTIZATION,
        metadata=meta,
        build_params={
            "max_depth": MAX_DEPTH,
            "num_bits": NUM_BITS,
            "min_leaf_size": MIN_LEAF_SIZE,
            "seed": SEED,
        },
        stored_fields={
            "title": titles,
            "digit": [str(l) for l in labels],
        },
    )
    size_mb = OUTPUT.stat().st_size / (1024 * 1024)
    print(f"  Written {OUTPUT.name} ({size_mb:.1f} MB) in {time.time() - t0:.1f}s")
    print("Done.")


if __name__ == "__main__":
    main()
