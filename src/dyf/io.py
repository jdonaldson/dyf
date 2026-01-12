"""
Index serialization for DYF classifiers.

Save and load pre-computed LSH indices using safetensors format.
"""

import numpy as np
from pathlib import Path
from typing import Union, Optional

try:
    from safetensors.numpy import save_file, load_file
    _HAS_SAFETENSORS = True
except ImportError:
    _HAS_SAFETENSORS = False


def save_index(
    classifier,
    path: Union[str, Path],
    embeddings: Optional[np.ndarray] = None,
    include_embeddings: bool = True,
    metadata: Optional[dict] = None
) -> None:
    """
    Save a fitted classifier's index to safetensors format.

    Args:
        classifier: Fitted OutlierClassifier instance
        path: Output file path (.safetensors)
        embeddings: Original embeddings (required if include_embeddings=True)
        include_embeddings: Whether to include embeddings in the file

    Example:
        >>> classifier.fit(embeddings)
        >>> save_index(classifier, 'index.safetensors', embeddings)
    """
    if not _HAS_SAFETENSORS:
        raise ImportError("safetensors required: pip install safetensors")

    if not classifier.is_fitted():
        raise ValueError("Classifier must be fitted before saving")

    statuses = np.array([int(s) for s in classifier.get_statuses()], dtype=np.int32)
    n = len(statuses)

    # Get bucket IDs for dense items
    bucket_ids = np.full(n, -1, dtype=np.int64)
    for idx in range(n):
        if statuses[idx] == 0:  # Dense
            bid = classifier.get_bucket_id(idx)
            if bid is not None:
                bucket_ids[idx] = bid

    # Get recovery bucket IDs for bridge items
    recovery_bucket_ids = np.full(n, -1, dtype=np.int64)
    for idx in range(n):
        if statuses[idx] == 1:  # Bridge
            rid = classifier.get_recovery_bucket_id(idx)
            if rid is not None:
                recovery_bucket_ids[idx] = rid

    data = {
        'statuses': statuses,
        'bucket_ids': bucket_ids,
        'recovery_bucket_ids': recovery_bucket_ids,
    }

    if include_embeddings:
        if embeddings is None:
            raise ValueError("embeddings required when include_embeddings=True")
        data['embeddings'] = embeddings.astype(np.float32)

    # Build default metadata
    meta = {
        'dyf_version': '0.1.1',
    }
    if metadata:
        meta.update({k: str(v) for k, v in metadata.items()})

    save_file(data, str(path), metadata=meta)


def load_index(path: Union[str, Path], include_metadata: bool = False):
    """
    Load a pre-computed index from safetensors format.

    Args:
        path: Path to .safetensors file
        include_metadata: If True, return (data, metadata) tuple

    Returns:
        dict with keys: statuses, bucket_ids, recovery_bucket_ids, embeddings (if present)
        If include_metadata=True, returns (data, metadata) tuple

    Example:
        >>> index = load_index('index.safetensors')
        >>> dense_mask = index['statuses'] == 0
        >>> bridge_mask = index['statuses'] == 1
        >>>
        >>> # With metadata
        >>> data, meta = load_index('index.safetensors', include_metadata=True)
        >>> print(meta.get('dyf_version'))
    """
    if not _HAS_SAFETENSORS:
        raise ImportError("safetensors required: pip install safetensors")

    from safetensors import safe_open

    if include_metadata:
        with safe_open(str(path), framework="numpy") as f:
            metadata = f.metadata()
            data = {key: f.get_tensor(key) for key in f.keys()}
        return data, metadata or {}

    return load_file(str(path))


class PrecomputedIndex:
    """
    Work with a pre-computed DYF index without the Rust classifier.

    Useful for deployment when you don't want the dyf-rs dependency.

    Example:
        >>> index = PrecomputedIndex.load('index.safetensors')
        >>> dense_items = index.get_dense()
        >>> bridge_items = index.get_bridge()
        >>> bucket_5_items = index.get_bucket(5)
    """

    def __init__(self, data: dict, metadata: Optional[dict] = None):
        self.statuses = data['statuses']
        self.bucket_ids = data['bucket_ids']
        self.recovery_bucket_ids = data['recovery_bucket_ids']
        self.embeddings = data.get('embeddings')
        self.metadata = metadata or {}

    @classmethod
    def load(cls, path: Union[str, Path]) -> 'PrecomputedIndex':
        """Load index from safetensors file."""
        data, metadata = load_index(path, include_metadata=True)
        return cls(data, metadata)

    @property
    def version(self) -> Optional[str]:
        """Get the dyf version used to create this index."""
        return self.metadata.get('dyf_version') or self.metadata.get('dyf_rs_version')

    def __len__(self) -> int:
        return len(self.statuses)

    def get_dense(self) -> np.ndarray:
        """Get indices of dense items."""
        return np.where(self.statuses == 0)[0]

    def get_bridge(self) -> np.ndarray:
        """Get indices of bridge items."""
        return np.where(self.statuses == 1)[0]

    def get_orphans(self) -> np.ndarray:
        """Get indices of orphan items."""
        return np.where(self.statuses == 2)[0]

    def get_bucket(self, bucket_id: int) -> np.ndarray:
        """Get indices of items in a specific bucket."""
        return np.where(self.bucket_ids == bucket_id)[0]

    def get_recovery_bucket(self, bucket_id: int) -> np.ndarray:
        """Get indices of items in a specific recovery bucket."""
        return np.where(self.recovery_bucket_ids == bucket_id)[0]

    def get_unique_buckets(self) -> np.ndarray:
        """Get unique bucket IDs (excluding -1)."""
        return np.unique(self.bucket_ids[self.bucket_ids >= 0])

    def get_unique_recovery_buckets(self) -> np.ndarray:
        """Get unique recovery bucket IDs (excluding -1)."""
        return np.unique(self.recovery_bucket_ids[self.recovery_bucket_ids >= 0])

    def report(self) -> dict:
        """Get summary statistics."""
        return {
            'total': len(self),
            'dense': int((self.statuses == 0).sum()),
            'bridge': int((self.statuses == 1).sum()),
            'orphan': int((self.statuses == 2).sum()),
            'num_buckets': len(self.get_unique_buckets()),
            'num_recovery_buckets': len(self.get_unique_recovery_buckets()),
        }
