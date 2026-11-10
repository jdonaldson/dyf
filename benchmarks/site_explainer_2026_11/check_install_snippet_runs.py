"""Run the code the explainer page shows under "Use it", on CLINC150 MiniLM, so the page
never prints a snippet that has not executed.  Prints k per resolution and the result fields."""

import os
import tempfile
from pathlib import Path

import numpy as np

from dyf import LazyIndex, build_dyf_tree, louvain_cluster_leaves, write_lazy_index

embeddings = np.load(Path.home() / ".cache/dyf_intent_bench/clinc150_X.npy").astype(np.float32)

tree = build_dyf_tree(embeddings, num_bits=3, max_depth=4, min_leaf_size=20, seed=42)
path = os.path.join(tempfile.mkdtemp(dir=os.environ.get("TMPDIR")), "requests.dyf")
write_lazy_index(tree, embeddings, path)

with LazyIndex(path) as idx:
    for coords in (None, np.zeros((len(embeddings), 2), dtype=np.float32)):
        try:
            for resolution in (1.0, 32.0):
                groups = louvain_cluster_leaves(idx, coords, embeddings, resolution=resolution)
                labels = np.asarray(groups.point_labels)
                print(
                    f"coords={'None' if coords is None else 'zeros'} resolution={resolution}: "
                    f"{len(np.unique(labels))} groups, fields={[f for f in dir(groups) if not f.startswith('_')]}"
                )
        except Exception as e:  # report which form fails
            print(f"coords={'None' if coords is None else 'zeros'} FAILED: {type(e).__name__}: {e}")
    hit = idx.search(embeddings[0], k=5, nprobe="auto")
    print("search:", hit.indices, np.round(hit.scores, 3))
