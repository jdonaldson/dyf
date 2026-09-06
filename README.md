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

# Search — instant open, LRU-bounded leaf cache, fast Rust backend
with LazyIndex("index.dyf") as idx:
    result = idx.search(query_embedding, k=10, nprobe=256, backend="rust")
    print(result.indices, result.scores)
    print(result.fields["title"])  # stored fields returned with results
```

For a fully in-memory corpus, `DenseSearchIndex` builds the tree and searches via the
same Rust kernel (batched queries supported):

```python
from dyf import DenseSearchIndex

idx = DenseSearchIndex(embeddings)                  # builds tree + flattens

result = idx.search(query, k=10, nprobe=256)
result.indices, result.scores                       # same SearchResult as LazyIndex

indices, scores = idx.search(query, k=10, nprobe=256)   # also unpacks as a tuple
I, S = idx.search(query_batch, k=10, nprobe=256)        # batched -> (nq, k)
```

Both index types return the same `SearchResult`, so they are interchangeable at the call
site. It unpacks as a 2-tuple, so the positional form keeps working.

### Inspect an Index

`dyf info` describes a `.dyf` file without loading its data — item count, dimensionality,
build params, stored fields, and how far the enrichment pipeline has been run:

```bash
dyf info index.dyf
dyf info index.dyf --json    # machine-readable
```

It is backed by `LazyIndex`'s lazy-open path, so it stays cheap regardless of file size —
**0.09s on a 479 MB, 229k-item index.** Useful for deciding what to do with an artifact
before paying to open it.

> The `--json` payload carries `"schema_version": 0` and is **unstable before v1**; the
> field is there so callers can detect a change rather than be surprised by one.

### Shrink the Index: Dedup on Ingest

Real corpora repeat themselves. Index one representative per near-duplicate cluster and
carry the mapping as a stored field:

```python
from dyf import near_duplicate_clusters, build_dyf_tree, write_lazy_index

result = near_duplicate_clusters(embeddings)        # cosine > 0.99, ~2.5s / 229k points
print(f"{result.n_removed:,} duplicates ({result.removed_fraction:.1%})")

reps = embeddings[result.mask()]
tree = build_dyf_tree(reps, max_depth=4, num_bits=4)
write_lazy_index(tree, reps, "index.dyf",
                 stored_fields={"dup_members": result.member_field()})

# at query time, expand a hit back to the points it stands for
from dyf import decode_members
also_matched = decode_members(result_fields["dup_members"][0])
```

**Measure before enabling — the duplicate rate is wildly corpus-dependent:**

| corpus | dup rate | `.dyf` saving |
|---|---|---|
| CMU MoCap (adjacent frames) | 88.3% | **77.1%** |
| SEC 10-Q (legal boilerplate) | 29.4% | **25.6%** |
| news / tweets / arxiv / wikipedia | 0.0–1.0% | ~0% |

Curated document collections have almost no near-duplicates; templated or temporally
oversampled corpora have many. Where duplicates exist, file saving is reliably **~0.87× the
duplicate rate**. `near_duplicate_clusters` is itself the cheap diagnostic (~1s per 100k
points), so measure first. Retrieval quality also improves when scored on distinct content,
because the probe budget stops re-scanning near-identical vectors. No file-format change:
the mapping is an ordinary utf8 stored field.

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

Search runs on a Rust multiprobe kernel (`dyf-rs >= 0.8.0`, PyO3) — the default path for
both `LazyIndex.search` and `DenseSearchIndex`. Results are **bit-identical** to the
pure-Python reference (`backend="python"`); the kernel handles fixed *and* adaptive
`nprobe` and `return_routing`. MSMARCO MiniLM-L6 (384d), Apple Silicon, batched unless
noted:

| path | corpus / setting | latency / query | vs pure-Python |
|------|------------------|-----------------|----------------|
| `DenseSearchIndex` (in-memory, batched) | 8.84M, nprobe=256 | ~0.5 ms | ~100× |
| `LazyIndex.search` (on-disk, batched) | up to 8.84M, nprobe=256 | ~0.9 ms | ~29× |
| `LazyIndex.search` (on-disk, single query) | nprobe=256 | ~4 ms | ~6× |
| `LazyIndex.search` (with stored fields) | immich 35K | ~2 ms | ~15× |

Speedup is largest at low `nprobe`, batched queries, and without field-gather. Lazy mode
opens in ~5 ms (vs ~0.4 s preload) and bounds memory with an LRU; only PQ-compressed and
overflow indexes fall back to Python.

> **Scope.** These are Rust-vs-pure-Python speedups — recovering the cost of the per-query
> Python loop. They are **not** a claim that dyf is the fastest ANN retriever: on the pure
> recall-vs-latency frontier, mature graph libraries (pynndescent, HNSW) are faster. dyf's
> strengths are structure discovery, hierarchy, instant-open on-disk indexes, and reaching
> exact recall on a single index by raising `nprobe`.

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

- **[How It Works](https://dyf.io/how-it-works.html)** — the algorithm, metrics, and Dense/Bridge/Orphan explained
- **[Getting Started](https://dyf.io/getting-started.html)** — code recipes and examples
- **[API Reference](https://dyf.io/reference/)** — full documentation for all classes and functions

## License

MIT
