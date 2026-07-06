"""Array boundary guards for the dyf-rs (Rust) API surface.

dyf-rs functions have typed PyO3 ``f32`` signatures: passing float64 (numpy's
default dtype) raises ``TypeError: 'ndarray' object is not an instance of
'ndarray'`` — a confusing message, and worse, several callers historically
swallowed it inside graceful-degradation ``except`` blocks, producing
degenerate-but-valid-looking results (e.g. a single-leaf tree whose one
cluster scores perfect purity downstream).

Two rules enforced here and at the call sites:
  1. Every array crossing into dyf-rs goes through :func:`ensure_f32` — a
     uniform, contiguous float32 conversion with a clear error naming the
     argument when the input cannot be converted.
  2. Graceful-degradation ``except`` blocks around dyf-rs calls must re-raise
     ``TypeError`` before their broad catch: a dtype/signature error is a
     programmer bug, never a data condition to degrade on.
"""
from __future__ import annotations

import numpy as np


def ensure_f32(arr, name: str = "embeddings") -> np.ndarray:
    """Return ``arr`` as a C-contiguous float32 ndarray, or raise clearly.

    Args:
        arr: Array-like input destined for a dyf-rs (Rust) API.
        name: Argument name, used in the error message.

    Returns:
        ``np.ndarray`` with ``dtype=float32``, C-contiguous. No copy is made
        when the input already satisfies both.

    Raises:
        TypeError: If the input cannot be converted to a float32 array.
    """
    try:
        return np.ascontiguousarray(arr, dtype=np.float32)
    except (TypeError, ValueError) as e:
        raise TypeError(
            f"{name!r} must be convertible to a float32 array for the dyf-rs "
            f"API (got {type(arr).__name__}"
            + (f" with dtype {arr.dtype}" if hasattr(arr, "dtype") else "")
            + f"): {e}"
        ) from e
