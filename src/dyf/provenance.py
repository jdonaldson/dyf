"""Provenance records for pipeline artifacts — **which the caller must stamp itself**.

⚠ This module provides the *vocabulary* for artifact identity, not the behaviour. Nothing
in dyf calls `create_provenance`: `write_lazy_index` records `build_params` but no
provenance, and `Pipeline` never stamps after running a stage — it assumes `build_fn` did.
Verified 2026-09-05.

The previous version of this docstring said "each artifact carries a Provenance record",
which was simply untrue and is the kind of claim that gets believed. If you want
provenance on a `.dyf`, you write it: build a record with `create_provenance` and pass it
through `metadata=`. `Pipeline` reads either a `_provenance` key or the highest
`_provenance_level_N` (the shape the downstream `dyfviz` enrichment stages write).

A Provenance describes an artifact's identity and inputs so a consumer can check
compatibility before loading and fail loudly rather than degrade silently — see
`check_compatible`.

Note `file_hash` is a *fast partial* hash — size, mtime and the first 64 KB — so a
touched-but-unchanged file reads as changed. That is deliberate (it is a change detector,
not a content address) but it is not a checksum.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


@dataclass
class Provenance:
    """Identity fingerprint for a pipeline artifact."""

    artifact_type: str  # "parquet", "rog_cache", "dyf", "label_cache"
    n_items: int  # row / point count
    source_hash: str  # hash of source file(s) — first 12 chars of sha256
    params_hash: str  # hash of build parameters dict
    created_at: str  # ISO timestamp
    params: dict = field(default_factory=dict)  # full build params (human-readable)
    sample_seed: int | None = None  # RNG seed used for sampling
    sample_n: int | None = None  # sample size requested (None = all rows)


# ---------------------------------------------------------------------------
# Hashing helpers
# ---------------------------------------------------------------------------


def file_hash(path: str | Path) -> str:
    """Fast partial hash: first 64 KB + file size + mtime.

    Not a full SHA — just enough to detect changes quickly.
    """
    p = Path(path)
    stat = p.stat()
    h = hashlib.sha256()
    h.update(str(stat.st_size).encode())
    h.update(str(int(stat.st_mtime_ns)).encode())
    with open(p, "rb") as f:
        h.update(f.read(65_536))
    return h.hexdigest()[:12]


def params_hash(params: dict) -> str:
    """Deterministic hash of sorted JSON-serialized params."""
    canonical = json.dumps(params, sort_keys=True, default=str)
    return hashlib.sha256(canonical.encode()).hexdigest()[:12]


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


def create_provenance(
    artifact_type: str,
    n_items: int,
    source_paths: list[str | Path],
    params: dict,
    sample_seed: int | None = None,
    sample_n: int | None = None,
) -> Provenance:
    """Create a provenance record for a new artifact."""
    # Combine hashes of all source files
    h = hashlib.sha256()
    for sp in sorted(str(s) for s in source_paths):
        p = Path(sp)
        if p.exists():
            h.update(file_hash(p).encode())
        else:
            # Source may be a stage name rather than a file
            h.update(sp.encode())
    combined_source = h.hexdigest()[:12]

    return Provenance(
        artifact_type=artifact_type,
        n_items=n_items,
        source_hash=combined_source,
        params_hash=params_hash(params),
        created_at=datetime.now(timezone.utc).isoformat(),
        params=params,
        sample_seed=sample_seed,
        sample_n=sample_n,
    )


# ---------------------------------------------------------------------------
# Compatibility check
# ---------------------------------------------------------------------------


def check_compatible(
    upstream: Provenance,
    downstream_n_items: int | None = None,
    downstream_sample_n: int | None = None,
    downstream_sample_seed: int | None = None,
) -> tuple[bool, list[str]]:
    """Check if an upstream artifact is compatible with the current pipeline.

    Returns ``(ok, warnings)``.
    """
    warnings: list[str] = []

    if downstream_n_items is not None and upstream.n_items != downstream_n_items:
        warnings.append(f"n_items mismatch: cache has {upstream.n_items}, current pipeline has {downstream_n_items}")

    if downstream_sample_n is not None and upstream.sample_n != downstream_sample_n:
        warnings.append(
            f"sample_n mismatch: cache built with sample={upstream.sample_n}, "
            f"current pipeline requests sample={downstream_sample_n}"
        )

    if (
        downstream_sample_seed is not None
        and upstream.sample_seed is not None
        and upstream.sample_seed != downstream_sample_seed
    ):
        warnings.append(f"sample_seed mismatch: cache={upstream.sample_seed}, current={downstream_sample_seed}")

    ok = len(warnings) == 0
    return ok, warnings


# ---------------------------------------------------------------------------
# Serialization
# ---------------------------------------------------------------------------


def provenance_to_dict(p: Provenance) -> dict[str, Any]:
    """JSON-serializable dict."""
    return asdict(p)


def provenance_from_dict(d: dict[str, Any]) -> Provenance:
    """Reconstruct from dict."""
    return Provenance(
        artifact_type=d["artifact_type"],
        n_items=d["n_items"],
        source_hash=d["source_hash"],
        params_hash=d["params_hash"],
        created_at=d["created_at"],
        params=d.get("params", {}),
        sample_seed=d.get("sample_seed"),
        sample_n=d.get("sample_n"),
    )
