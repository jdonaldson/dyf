"""Parquet round trip for ``.dyf`` indexes: ``to_parquet`` / ``from_parquet``.

Why this exists at the *dataset boundary* rather than inside the file: a polars- or
DuckDB-centric pipeline wants embeddings sitting beside business columns, queryable with
predicate pushdown. That is an import/export concern. Making Parquet the in-file payload
encoding was considered and rejected — see ``PARQUET_NOTES.md`` — because a ``.dyf`` leaf
batch is far below Parquet's unit of benefit. Measured leaf sizes:

===========================  =======  ======  =======  =============  ======
index                         items    dim    leaves   rows/leaf p50   max
===========================  =======  ======  =======  =============  ======
``gudid_viz.dyf``              50,000    768    5,938              5     196
``products_…_n50000``          50,000    384      842             34   4,420
===========================  =======  ======  =======  =============  ======

A median leaf of 5 rows against a ~1M-row Parquet row group means per-batch footers would
dominate the payload. The same numbers are why ``to_parquet`` **does not** Hive-partition by
leaf by default: ``gudid_viz.dyf`` would emit 5,938 directories averaging 8 rows each.

Leaf locality is still worth keeping, so rows are written **in leaf order** with a ``leaf_id``
column. That is free — concatenating batches in batch-index order is already sorted — and it
lets Parquet row-group min/max statistics prune on ``leaf_id`` without any small-files cost.

Round trip is lossy in one direction by default, and the loss is in the *index*, not here:
``write_lazy_index`` quantizes to ``float16`` unless told otherwise. ``to_parquet`` exports
what the file actually holds (upcast to float32, since Parquet has no half-precision logical
type) and records the original quantization in the Parquet key/value metadata. A PQ index
stores codes rather than vectors, so exporting one requires ``allow_lossy=True`` and is
recorded as ``dyf_embeddings_reconstructed=true``.
"""

from __future__ import annotations

import hashlib
import json
import os
from datetime import datetime, timezone
from typing import Any

import numpy as np

#: Column names ``write_lazy_index`` owns in the leaf schema. A stored field cannot use these.
RESERVED_COLUMNS = ("item_index", "embedding")

#: Columns ``to_parquet`` adds that describe the *exporting* tree. They are deliberately not
#: re-imported: a ``leaf_id`` from an old build is meaningless against a freshly fitted tree,
#: and silently carrying one is the misalignment failure this module is designed to avoid.
EXPORT_ONLY_COLUMNS = ("leaf_id", "node_id")


def _sha256(path: str, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        while True:
            b = fh.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def _embedding_to_f32(col: Any, dim: int) -> Any:
    """Return ``col`` as ``fixed_size_list<float32, dim>``.

    Half-float storage is the default in ``.dyf`` and Parquet has no half-precision logical
    type, so this always needs to run. Tries Arrow's cast first and falls back to numpy,
    because fixed-size-list value casts are not supported by every pyarrow version.
    """
    import pyarrow as pa

    target = pa.list_(pa.float32(), dim)
    if col.type == target:
        return col
    try:
        return col.cast(target)
    except (pa.ArrowNotImplementedError, pa.ArrowInvalid):
        flat = np.asarray(col.values).astype(np.float32, copy=False)
        return pa.FixedSizeListArray.from_arrays(pa.array(flat, type=pa.float32()), dim)


def to_parquet(
    source: Any,
    out: str,
    *,
    row_group_size: int = 65536,
    compression: str = "zstd",
    partition_by_leaf: bool = False,
    allow_lossy: bool = False,
    fingerprint: bool = True,
    include: list[str] | None = None,
    exclude: list[str] | None = None,
) -> dict[str, Any]:
    """Export a ``.dyf`` index to Parquet, preserving leaf order.

    Args:
        source: Path to a ``.dyf`` file, or an open ``LazyIndex``.
        out: Output path. A ``.parquet`` file, or a directory when
            ``partition_by_leaf=True``.
        row_group_size: Rows per Parquet row group. Because rows are emitted in leaf order,
            row-group ``leaf_id`` statistics are what give engines pruning; smaller groups
            prune harder and cost more metadata.
        compression: Parquet codec (``zstd``, ``snappy``, ``gzip``, ``none``).
        partition_by_leaf: Emit Hive partitions ``leaf_id=<id>/``. **Off by default** — see the
            module docstring; on a high-leaf index this produces thousands of tiny files.
            Raises unless the index averages at least 64 rows per leaf, which is the point
            where a partition is worth a file.
        allow_lossy: Required to export a PQ index, whose stored codes can only be
            reconstructed approximately.
        fingerprint: Hash the source file into the Parquet metadata. O(file size).
        include: Stored fields to export. Default: all.
        exclude: Stored fields to skip.

    Returns:
        Report dict: ``rows``, ``leaves``, ``embedding_dim``, ``fields``, ``skipped``,
        ``bytes_out``, ``path``, ``reconstructed``.
    """
    import pyarrow as pa
    import pyarrow.parquet as pq

    from .lazy_index import LazyIndex

    owns = isinstance(source, str)
    idx = LazyIndex(source) if owns else source
    src_path = source if owns else str(getattr(idx, "_path", "") or "")

    try:
        if idx.is_pq and not allow_lossy:
            raise ValueError(
                "this index is PQ-quantized: stored codes reconstruct embeddings only "
                "approximately. Pass allow_lossy=True to export reconstructions (they will be "
                "marked dyf_embeddings_reconstructed=true in the Parquet metadata)."
            )

        dim = idx.embedding_dim
        n_batches = idx._index.BatchesLength()
        meta = idx._get_metadata()
        has_embeddings = meta.get("has_embeddings") != "false"

        sf_names = list(idx.stored_field_names)
        keep = [f for f in sf_names if (include is None or f in include) and (exclude is None or f not in exclude)]

        if partition_by_leaf:
            avg = idx.total_items / max(n_batches, 1)
            if avg < 64:
                raise ValueError(
                    f"refusing to Hive-partition: {n_batches} leaves averaging {avg:.1f} rows each "
                    f"would emit {n_batches} tiny files. Leave partition_by_leaf=False — rows are "
                    "already written in leaf order and row-group statistics give the same pruning."
                )

        tables = []
        for bi in range(n_batches):
            batch = idx.get_leaf(bi)
            names = batch.schema.names
            cols: list[Any] = [batch.column("item_index")]
            out_names = ["item_index"]

            if has_embeddings and "embedding" in names:
                emb = batch.column("embedding")
                if idx.is_pq:
                    m = int(meta["pq_n_subquantizers"])
                    codes = np.asarray(emb.values).reshape(len(emb), m)
                    rec = idx._pq_reconstruct(codes).astype(np.float32, copy=False)
                    emb = pa.FixedSizeListArray.from_arrays(pa.array(rec.reshape(-1), type=pa.float32()), dim)
                else:
                    emb = _embedding_to_f32(emb, dim)
                cols.append(emb)
                out_names.append("embedding")

            for f in keep:
                if f in names:
                    cols.append(batch.column(f))
                    out_names.append(f)

            cols.append(pa.array(np.full(batch.num_rows, bi, dtype=np.int32)))
            out_names.append("leaf_id")
            tables.append(pa.Table.from_arrays(cols, names=out_names))

        table = pa.concat_tables(tables) if tables else pa.table({})

        kv = {
            "dyf_source_path": os.path.abspath(src_path) if src_path else "",
            "dyf_format_version": str(idx.format_version),
            "dyf_total_items": str(idx.total_items),
            "dyf_embedding_dim": str(dim),
            "dyf_num_leaves": str(n_batches),
            "dyf_quantization": str(idx.quantization),
            "dyf_is_pq": str(bool(idx.is_pq)).lower(),
            "dyf_embeddings_reconstructed": str(bool(idx.is_pq)).lower(),
            "dyf_has_embeddings": str(bool(has_embeddings)).lower(),
            "dyf_exported_at": datetime.now(timezone.utc).isoformat(),
            "dyf_row_order": "leaf_id ascending (batch index order)",
        }
        # Carry the build params so a round trip can reproduce the SAME tree shape. Without
        # them `from_parquet` falls back to its own defaults and silently rebuilds a
        # differently shaped index — measured on a 50k/384d product index, 842 leaves became
        # 14,385 and the file went 48 MB -> 136 MB. Nothing errored; the shape just changed.
        bp = (idx.tree_summary or {}).get("build_params") or {}
        for key in ("max_depth", "num_bits", "min_leaf_size", "seed"):
            if bp.get(key) is not None:
                kv[f"dyf_build_{key}"] = str(bp[key])
        if fingerprint and src_path and os.path.exists(src_path):
            kv["dyf_sha256"] = _sha256(src_path)
        for k, v in meta.items():
            kv[f"dyf_meta__{k}"] = str(v)
        table = table.replace_schema_metadata({k.encode(): str(v).encode() for k, v in kv.items()})

        if partition_by_leaf:
            pq.write_to_dataset(table, root_path=out, partition_cols=["leaf_id"], compression=compression)
            bytes_out = sum(os.path.getsize(os.path.join(r, f)) for r, _, fs in os.walk(out) for f in fs)
        else:
            os.makedirs(os.path.dirname(os.path.abspath(out)) or ".", exist_ok=True)
            pq.write_table(table, out, row_group_size=row_group_size, compression=compression)
            bytes_out = os.path.getsize(out)

        return {
            "path": out,
            "rows": table.num_rows,
            "leaves": n_batches,
            "embedding_dim": dim,
            "fields": keep,
            "skipped": [f for f in sf_names if f not in keep],
            "bytes_out": bytes_out,
            "reconstructed": bool(idx.is_pq),
        }
    finally:
        if owns:
            idx.close()


def _coerce_stored_field(col: Any) -> tuple[Any, str] | tuple[None, str]:
    """Coerce an Arrow column to something ``write_lazy_index`` accepts.

    Returns ``(value, kind)`` on success or ``(None, reason)`` when the type has no
    representation in the stored-field schema. Skipping loudly beats coercing silently:
    a struct flattened to its repr is worse than an absent column.
    """
    import pyarrow as pa
    import pyarrow.compute as pc

    t = col.type
    if pa.types.is_string(t) or pa.types.is_large_string(t):
        return [("" if v is None else v) for v in col.to_pylist()], "utf8"
    if pa.types.is_boolean(t):
        return np.asarray(col.to_pylist(), dtype=object).astype(bool).astype(np.int32), "int32"
    if pa.types.is_binary(t) or pa.types.is_large_binary(t):
        return [(b"" if v is None else v) for v in col.to_pylist()], "binary"
    if pa.types.is_integer(t):
        wide = t.bit_width >= 64 or pa.types.is_unsigned_integer(t) and t.bit_width >= 32
        dtype = np.int64 if wide else np.int32
        return np.nan_to_num(col.to_numpy(zero_copy_only=False)).astype(dtype), ("int64" if wide else "int32")
    if pa.types.is_float64(t) or pa.types.is_decimal(t):
        return pc.cast(col, pa.float64()).to_numpy(zero_copy_only=False).astype(np.float64), "float64"
    if pa.types.is_floating(t):
        return pc.cast(col, pa.float32()).to_numpy(zero_copy_only=False).astype(np.float32), "float32"
    if pa.types.is_temporal(t):
        return [("" if v is None else v.isoformat()) for v in col.to_pylist()], "utf8(from temporal)"
    return None, f"unsupported arrow type {t}"


def from_parquet(
    source: Any,
    out: str,
    *,
    embedding_col: str = "embedding",
    carry: list[str] | None = None,
    exclude: list[str] | None = None,
    max_depth: int | None = None,
    num_bits: int | None = None,
    min_leaf_size: int | None = None,
    seed: int | None = None,
    fit_method: str = "raw_pca",
    quantization: str = "float16",
    compression: str = "none",
    format_version: int = 3,
    metadata: dict[str, str] | None = None,
    preserve_item_index: bool = True,
) -> dict[str, Any]:
    """Build a ``.dyf`` index from a Parquet file, directory, or dataset.

    The tree's hyperplanes come from a global PCA fit over every vector, so the whole
    embedding column is needed at build time. A Parquet column scan supplies exactly that,
    contiguously — the same property that makes an incremental Postgres ``aminsert`` path hard
    makes this easy.

    Args:
        source: Parquet file path, directory, glob, or list of paths.
        out: Output ``.dyf`` path.
        embedding_col: Column holding the vectors. ``pl.Array`` / Arrow
            ``FixedSizeList<float*>`` preferred; a variable ``List`` works if every row has
            the same length.
        carry: Columns to store as fields. Default: every column except the embedding,
            ``item_index``, and the export-only ``leaf_id`` / ``node_id``.
        exclude: Columns to drop.
        max_depth, num_bits, min_leaf_size, seed: Passed to ``build_dyf_tree``. Each defaults to
            the value recorded by ``to_parquet`` in the Parquet metadata, so a round trip
            reproduces the source tree's shape; an explicit argument overrides it, and the
            fallback when neither is present is ``max_depth=6, num_bits=3, min_leaf_size=4,
            seed=42``. This matters more than it looks: rebuilding a 50k/384d product index with
            mismatched params turned 842 leaves into 14,385 and tripled the file, without error.
        fit_method: Passed to ``build_dyf_tree``. Not recorded in ``.dyf``, so never inherited.
        quantization, compression, format_version, metadata: Passed to ``write_lazy_index``.
        preserve_item_index: Carry an existing ``item_index`` column through as
            ``source_item_index``. It cannot keep its own name — ``write_lazy_index`` owns
            ``item_index`` in the leaf schema — and dropping it silently would break any join
            back to the exporting index.

    Returns:
        Report dict: ``path``, ``rows``, ``embedding_dim``, ``fields``, ``skipped``,
        ``leaves``, ``bytes_out``, ``source_metadata``.
    """
    import pyarrow as pa
    import pyarrow.dataset as ds

    from .dyf_tree import build_dyf_tree
    from .lazy_index import LazyIndex, write_lazy_index

    dataset = ds.dataset(source, format="parquet")
    table = dataset.to_table()
    if embedding_col not in table.schema.names:
        raise ValueError(f"embedding column {embedding_col!r} not in {table.schema.names}")

    col = table.column(embedding_col).combine_chunks()
    if isinstance(col, pa.ChunkedArray):
        col = col.chunk(0) if col.num_chunks == 1 else pa.concat_arrays(col.chunks)

    if pa.types.is_fixed_size_list(col.type):
        dim = col.type.list_size
        emb = np.asarray(col.values).astype(np.float32, copy=False).reshape(-1, dim)
    elif pa.types.is_list(col.type) or pa.types.is_large_list(col.type):
        lengths = np.diff(np.asarray(col.offsets))
        if len(lengths) and lengths.min() != lengths.max():
            raise ValueError(
                f"{embedding_col!r} is a variable-length list ({lengths.min()}–{lengths.max()}); "
                "embeddings must be uniform. Cast to pl.Array(pl.Float32, dim) before export."
            )
        dim = int(lengths[0]) if len(lengths) else 0
        emb = np.asarray(col.values).astype(np.float32, copy=False).reshape(-1, dim)
    else:
        raise ValueError(f"{embedding_col!r} has type {col.type}; expected a fixed-size or variable list")

    dropped = {embedding_col, *EXPORT_ONLY_COLUMNS, *(exclude or ())}
    candidates = [c for c in table.schema.names if c not in dropped]
    if carry is not None:
        candidates = [c for c in candidates if c in carry]

    stored: dict[str, Any] = {}
    skipped: list[dict[str, str]] = []
    for name in candidates:
        target = name
        if name == "item_index":
            if not preserve_item_index:
                skipped.append({"column": name, "reason": "reserved name; preserve_item_index=False"})
                continue
            target = "source_item_index"
        elif name in RESERVED_COLUMNS:
            skipped.append({"column": name, "reason": f"reserved by write_lazy_index: {name}"})
            continue

        value, kind = _coerce_stored_field(table.column(name).combine_chunks())
        if value is None:
            skipped.append({"column": name, "reason": kind})
            continue
        stored[target] = value

    src_kv = {}
    raw = table.schema.metadata or {}
    for k, v in raw.items():
        key = k.decode() if isinstance(k, bytes) else str(k)
        if key.startswith("dyf_"):
            src_kv[key] = v.decode() if isinstance(v, bytes) else str(v)

    # Inherit the exporting index's build params unless the caller overrode them, so the
    # round trip is shape-preserving by default rather than silently reshaping the tree.
    defaults = {"max_depth": 6, "num_bits": 3, "min_leaf_size": 4, "seed": 42}
    explicit = {"max_depth": max_depth, "num_bits": num_bits, "min_leaf_size": min_leaf_size, "seed": seed}
    params: dict[str, int] = {}
    inherited: list[str] = []
    for name, value in explicit.items():
        if value is not None:
            params[name] = int(value)
        elif src_kv.get(f"dyf_build_{name}") is not None:
            params[name] = int(src_kv[f"dyf_build_{name}"])
            inherited.append(name)
        else:
            params[name] = defaults[name]

    tree = build_dyf_tree(emb, fit_method=fit_method, **params)

    md = {
        "parquet_source": json.dumps(source if isinstance(source, (str, list)) else str(source)),
        "parquet_imported_at": datetime.now(timezone.utc).isoformat(),
    }
    if src_kv.get("dyf_sha256"):
        md["parquet_source_dyf_sha256"] = src_kv["dyf_sha256"]
    md.update(metadata or {})

    write_lazy_index(
        tree,
        emb,
        out,
        compression=compression,
        quantization=quantization,
        metadata=md,
        stored_fields=stored or None,
        format_version=format_version,
    )

    # Read the leaf count back off the written file rather than inferring it from the tree
    # dict. `len(tree["children"])` counts top-level children, not leaves, and returns a
    # plausible-looking small integer — which is exactly how a wrong number survives review.
    with LazyIndex(out) as written:
        leaves = written.num_leaves

    return {
        "path": out,
        "rows": int(emb.shape[0]),
        "embedding_dim": int(dim),
        "fields": sorted(stored),
        "skipped": skipped,
        "leaves": int(leaves),
        "bytes_out": os.path.getsize(out),
        "build_params": params,
        "build_params_inherited": inherited,
        "source_metadata": src_kv,
    }
