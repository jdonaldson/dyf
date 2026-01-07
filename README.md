# DYF - Outlier Classification

Fast outlier classification using PCA-based LSH. Identifies three types of items in embedding spaces:

- **Dense**: Items in well-populated semantic buckets (the majority)
- **Diaspora**: Sparse items that find community via recovery PCA (misplaced by global structure)
- **Orphan**: Truly unique items with no semantic neighbors

## Installation

```bash
pip install dyf
```

For full features (embedding generation, LLM labeling):
```bash
pip install dyf[full]
```

## Quick Start

### Fast Classification (Rust-accelerated)

```python
import numpy as np
from dyf import OutlierClassifier

# Create embeddings (e.g., from sentence-transformers)
embeddings = np.random.randn(10000, 384).astype(np.float32)

# Classify outliers (~60ms for 60K samples)
classifier = OutlierClassifier(embedding_dim=384)
classifier.fit(embeddings)

# Get results
print(classifier.report())
diaspora = classifier.get_diaspora()  # Indices of diaspora items
orphans = classifier.get_orphans()    # Indices of orphan items
```

### Full-Featured Usage (with embeddings & labeling)

```python
from dyf import OutlierClassifierFull, EmbedderConfig, LabelerConfig

# From raw texts with built-in TF-IDF embeddings
classifier = OutlierClassifierFull.from_texts(
    texts=documents,
    categories=categories,  # Optional category labels
    embedding_dim=128
)

# Or use sentence-transformers
embeddings = EmbedderConfig.MEDIUM.embed(texts)  # all-mpnet-base-v2
classifier = OutlierClassifierFull(embedding_dim=768)
classifier.fit(embeddings, categories=categories, texts=texts)

# Get detailed report
print(classifier.report())

# Label buckets with local LLM
labels = classifier.label_buckets(**LabelerConfig.MEDIUM.as_kwargs())
print(labels['dense'][1234]['label'])  # "Reinforcement Learning"

# Or use keyword extraction (no LLM required)
labels = classifier.label_buckets_keywords()
```

## Performance

| Implementation | 60K samples (384d) | Per sample |
|----------------|-------------------|------------|
| DYF (Rust)     | ~60ms            | 1.0 µs     |
| Pure Python    | ~230ms           | 3.8 µs     |

**3.8x faster** than pure Python/sklearn.

## API Reference

### OutlierClassifier (Fast)

```python
OutlierClassifier(
    embedding_dim: int,
    initial_bits: int = 14,       # Bits for initial PCA LSH
    recovery_bits: int = 8,       # Bits for recovery PCA
    dense_threshold: int = 10,    # Min bucket size for "dense"
    intra_outlier_std: float = 2.0,   # Std threshold for intra-bucket outliers
    recovery_cluster_min: int = 3,    # Min cluster size for "recovered"
    seed: int = 31
)
```

**Methods:**
- `fit(embeddings)` - Fit on numpy array (n_samples, embedding_dim)
- `fit_arrow(arrow_array)` - Fit on PyArrow FixedSizeListArray (zero-copy)
- `get_diaspora()` - Get indices of diaspora items
- `get_orphans()` - Get indices of orphan items
- `get_statuses()` - Get status for all items
- `report()` - Get classification report

### EmbedderConfig Presets

| Name | Model | Dimensions | Size |
|------|-------|------------|------|
| `TFIDF` | TF-IDF + SVD | 128 | 0 MB |
| `LOW` | all-MiniLM-L6-v2 | 384 | 80 MB |
| `MEDIUM` | all-mpnet-base-v2 | 768 | 420 MB |
| `HIGH` | bge-large-en-v1.5 | 1024 | 1.3 GB |

### LabelerConfig Presets

| Name | Model | Parameters |
|------|-------|------------|
| `KEYWORDS` | TF-IDF keywords | - |
| `LOW` | phi3:mini | 3.8B |
| `MEDIUM` | qwen2.5:7b | 7B |
| `HIGH` | qwen2.5:14b | 14B |

## Algorithm

Two-stage PCA-based LSH outlier classification:

1. **Stage 1**: Random hash → bucket centroids → PCA on centroids → re-hash
2. **Outlier Detection**: Sparse buckets + intra-bucket distance outliers
3. **Stage 2**: Recovery PCA on outliers → diaspora vs orphan

The key insight: outliers from global PCA often share structure at coarser resolution.

## License

MIT
