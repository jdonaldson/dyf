# DYF - Density Yields Features

[![Interactive Demo](assets/wiki_graph_header.png)](https://jdonaldson.github.io/dyf/)

*50,000 Wikipedia articles clustered by semantic similarity. Bright lines show density-based bridges connecting clusters. [Try the interactive demo →](https://jdonaldson.github.io/dyf/)*

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

### Save & Load Pre-computed Indexes

```python
from dyf import save_index, PrecomputedIndex

# Save (includes embeddings + metadata)
save_index(classifier, 'index.safetensors', embeddings,
           metadata={'model': 'all-MiniLM-L6-v2', 'created': '2026-01-12'})

# Load (no dyf-rs dependency needed!)
index = PrecomputedIndex.load('index.safetensors')
print(index.version)  # Check what version created this
print(index.metadata)  # All metadata

dense_items = index.get_dense()
bucket_5 = index.get_bucket(5)
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

### Index Serialization

```python
from dyf import save_index, load_index, PrecomputedIndex

# Save fitted classifier
save_index(classifier, 'index.safetensors', embeddings, metadata={...})

# Load as dict
data = load_index('index.safetensors')
data, metadata = load_index('index.safetensors', include_metadata=True)

# Load as object (recommended)
index = PrecomputedIndex.load('index.safetensors')
index.get_dense()
index.get_bucket(5)
index.metadata
index.version
```

## License

MIT
