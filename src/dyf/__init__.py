"""
DYF - Density Yields Features

Discover structure in embedding spaces using PCA-based LSH:
- Dense: Items in well-populated semantic buckets
- Bridge: Transitional items connecting different clusters
- Orphan: Truly unique items with no semantic neighbors

Quick Start:
    >>> from dyf import DensityClassifier
    >>> classifier = DensityClassifier(embedding_dim=384)
    >>> classifier.fit(embeddings)
    >>> print(classifier.report())

Full-Featured Usage:
    >>> from dyf import DensityClassifierFull, EmbedderConfig, LabelerConfig
    >>> classifier = DensityClassifierFull.from_texts(texts, categories=categories)
    >>> labels = classifier.label_buckets(**LabelerConfig.MEDIUM.as_kwargs())
"""

# Fast Rust implementation (core classifier)
try:
    from dyf_rs import (
        DensityClassifier,
        DensityReport,
        DensityStatus,
    )
    _HAS_RUST = True
except ImportError:
    _HAS_RUST = False
    DensityClassifier = None
    DensityReport = None
    DensityStatus = None

# Python wrapper with full features (embedder configs, labeling, etc.)
from .classifier import (
    DensityClassifier as DensityClassifierFull,
    DensityReport as DensityReportFull,
    BridgeCluster,
    EmbedderConfig,
    LabelerConfig,
    list_configs,
)

# Index serialization
from .io import save_index, load_index, PrecomputedIndex

__version__ = "0.2.0"
__all__ = [
    # Fast Rust core
    "DensityClassifier",
    "DensityReport",
    "DensityStatus",
    # Full Python wrapper
    "DensityClassifierFull",
    "DensityReportFull",
    "BridgeCluster",
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
