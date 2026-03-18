# DYF - Density Yields Features

[![CI](https://github.com/jdonaldson/dyf/actions/workflows/ci.yml/badge.svg)](https://github.com/jdonaldson/dyf/actions/workflows/ci.yml)
[![PyPI](https://img.shields.io/pypi/v/dyf)](https://pypi.org/project/dyf/)
[![Python](https://img.shields.io/pypi/pyversions/dyf)](https://pypi.org/project/dyf/)

Discover structure in embedding spaces. DYF uses density-based LSH to reveal the natural organization of your data:

- **Dense**: Core items in well-populated semantic regions
- **Bridge**: Transitional items connecting different clusters
- **Orphan**: Unique items with no semantic neighbors

## What it does

DYF transforms raw embeddings into navigable semantic maps. Instead of just clustering, it reveals the *topology* - which regions are dense, which items bridge between concepts, and which are truly unique.

Use cases:
- **Semantic navigation**: Find paths between concepts
- **Structure discovery**: Understand how your data organizes itself
- **Anomaly detection**: Identify orphans and bridges
- **Index building**: Pre-compute structure for fast queries

## Installation

```bash
pip install dyf
```

For serialization (save/load indexes):
```bash
pip install dyf[io]
```

For full features (embedding generation, LLM labeling):
```bash
pip install dyf[full]
```

## Quick Start

### Discover Structure

```python
import numpy as np
from dyf import DensityClassifier

# Your embeddings (e.g., from sentence-transformers)
embeddings = np.random.randn(10000, 384).astype(np.float32)

# Find structure
classifier = DensityClassifier(embedding_dim=384)
classifier.fit(embeddings)

# What did we find?
print(classifier.report())
# Corpus: 10000 items
#   Dense: 9500 (95.0%)
#   Bridge: 450 (4.5%)
#   Orphan: 50 (0.5%)

# Get indices
bridges = classifier.get_bridge()  # Transitional items
orphans = classifier.get_orphans() # Unique items
```

### Build & Search Indexes

```python
from dyf import build_dyf_tree, write_lazy_index, LazyIndex

# Build tree from embeddings
tree = build_dyf_tree(embeddings, max_depth=4, num_bits=3, min_leaf_size=8)

# Write to disk (mmap-friendly, zero startup cost)
write_lazy_index(tree, embeddings, "index.dyf",
                 quantization="float16", compression="zstd",
                 stored_fields={"title": titles},
                 metadata={"model": "nomic-embed-text-v1.5"})

# Search (instant open, LRU-cached leaf access)
with LazyIndex("index.dyf") as idx:
    result = idx.search(query_embedding, k=10, nprobe=3)
    print(result.indices, result.scores)
    print(result.fields["title"])  # stored fields returned with results
```

### Adaptive Probing

Queries near decision boundaries automatically probe more leaves:

```python
from dyf import LazyIndex, AdaptiveProbeConfig

with LazyIndex("index.dyf") as idx:
    # Auto mode: margin-based probe count (default thresholds)
    result = idx.search(query, k=10, nprobe="auto", return_routing=True)
    print(result.routing["adaptive_nprobe"])  # how many leaves were probed

    # Custom thresholds
    cfg = AdaptiveProbeConfig(margin_lo=0.005, margin_hi=0.2,
                              min_probes=1, max_probes=8)
    result = idx.search(query, k=10, nprobe=cfg)
```

### Full-Featured Usage

```python
from dyf import DensityClassifierFull, EmbedderConfig, LabelerConfig

# From raw texts
classifier = DensityClassifierFull.from_texts(
    texts=documents,
    categories=categories,
)

# Label clusters with LLM
labels = classifier.label_buckets(**LabelerConfig.MEDIUM.as_kwargs())
print(labels['dense'][1234]['label'])  # "Machine Learning Papers"
```

## How It Works

Two-stage PCA-based LSH:

1. **Initial bucketing**: PCA projections create semantic buckets
2. **Density check**: Items in sparse buckets are candidates for reclassification
3. **Recovery stage**: Coarser PCA finds structure among sparse items
4. **Classification**: Dense (core), Bridge (recovered), Orphan (truly unique)

The key insight: items that appear as outliers globally often share structure at coarser resolution. Bridges are these "misplaced" items - they connect different semantic regions.

## Performance

| Dataset | Time | Per item |
|---------|------|----------|
| 60K embeddings (384d) | ~60ms | 1.0 µs |

Rust-accelerated via PyO3. ~4x faster than pure Python.

## API

### DensityClassifier

```python
DensityClassifier(
    embedding_dim: int,
    initial_bits: int = 14,      # LSH resolution
    recovery_bits: int = 8,      # Coarser recovery resolution
    dense_threshold: int = 10,   # Min bucket size for "dense"
    seed: int = 31
)

# Methods
classifier.fit(embeddings)
classifier.get_dense()           # Dense item indices
classifier.get_bridge()          # Bridge item indices
classifier.get_orphans()         # Orphan item indices
classifier.get_bucket_id(idx)    # Which bucket is item in?
classifier.report()              # Summary statistics
```

### LazyIndex

```python
from dyf import LazyIndex

with LazyIndex("index.dyf") as idx:
    # Search with fixed or adaptive probing
    result = idx.search(query, k=10, nprobe=3)       # fixed
    result = idx.search(query, k=10, nprobe="auto")   # adaptive

    # Inspect index structure
    idx.tree_summary          # metadata, dims, leaf count
    idx.total_items           # total indexed items
    idx.stored_field_names    # available stored fields

    # Extract all data
    data = idx.extract_all_fields()
    data['embeddings']        # (n, d) float32
    data['fields']            # {field_name: array}
```

## Documentation

Full documentation and API reference at [dyf.io](https://dyf.io).

## License

MIT
