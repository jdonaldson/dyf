"""
DYF - Outlier Classification using PCA-based LSH

Fast identification of outliers in embedding spaces:
- Dense: Items in well-populated semantic buckets
- Diaspora: Sparse items that find community via recovery PCA
- Orphan: Truly unique items with no semantic neighbors

Quick Start:
    >>> from dyf import OutlierClassifier
    >>> classifier = OutlierClassifier(embedding_dim=384)
    >>> classifier.fit(embeddings)
    >>> print(classifier.report())

Full-Featured Usage:
    >>> from dyf import OutlierClassifierFull, EmbedderConfig, LabelerConfig
    >>> classifier = OutlierClassifierFull.from_texts(texts, categories=categories)
    >>> labels = classifier.label_buckets(**LabelerConfig.MEDIUM.as_kwargs())
"""

# Fast Rust implementation (core classifier)
try:
    from dyf_rs import (
        OutlierClassifier,
        OutlierReport,
        OutlierStatus,
    )
    _HAS_RUST = True
except ImportError:
    _HAS_RUST = False
    OutlierClassifier = None
    OutlierReport = None
    OutlierStatus = None

# Python wrapper with full features (embedder configs, labeling, etc.)
from .classifier import (
    OutlierClassifier as OutlierClassifierFull,
    OutlierReport as OutlierReportFull,
    DiasporaCluster,
    EmbedderConfig,
    LabelerConfig,
    list_configs,
)

# Index serialization
from .io import save_index, load_index, PrecomputedIndex

__version__ = "0.1.2"
__all__ = [
    # Fast Rust core
    "OutlierClassifier",
    "OutlierReport",
    "OutlierStatus",
    # Full Python wrapper
    "OutlierClassifierFull",
    "OutlierReportFull",
    "DiasporaCluster",
    "EmbedderConfig",
    "LabelerConfig",
    "list_configs",
    # Serialization
    "save_index",
    "load_index",
    "PrecomputedIndex",
]

def check_rust_available():
    """Check if Rust acceleration is available."""
    return _HAS_RUST
