"""DYF Lazy Index: FlatBuffers + Arrow IPC for mmap-based query serving.

File formats:
    DYF1: [16-byte header: magic "DYF1" + fb_size (u64) + reserved (u32)]
          [FlatBuffers section] [4KB padding] [Arrow IPC batches...]

    DYF3: [32-byte header: magic "DYF3" + fb_size (u64) + n_chunks (u16)
           + flags (u16) + reserved (16 bytes)]
          [FlatBuffers section] [4KB padding] [Arrow IPC batches...]
          Supports chunked transport: split_dyf3() splits into foo.dyf +
          foo.dyf.1, foo.dyf.2, ... for GitHub Pages / CDN hosting.

Writer:
    write_lazy_index(tree, embeddings, path, ...)

Reader:
    LazyIndex(path) — mmap, zero startup cost, LRU-cached leaf access
"""

from __future__ import annotations

import json
import mmap
import struct
from collections import deque
from collections.abc import Mapping, Sequence, Set
from dataclasses import dataclass, field
from typing import TypedDict

import numpy as np

# Stored field value types:
# - Output (from extract_all_fields): specific list type
StoredFieldValue = np.ndarray | list[str | bytes | None]
# - Input (to write_lazy_index, rewrite_lazy_index): accepts list[str], list[bytes], etc.
StoredFieldInput = np.ndarray | Sequence[str | bytes | None]


class TreeNode(TypedDict):
    """Single node from LazyIndex.get_tree_structure()."""
    node_id: int
    parent_id: int | None
    depth: int
    num_items: int
    is_leaf: bool
    batch_index: int           # -1 for internal nodes
    eigenvalues: np.ndarray | None  # (num_bits,) float32 or None


class ExtractedData(TypedDict):
    """Return type for LazyIndex.extract_all_fields()."""
    embeddings: np.ndarray     # (N, D) float32
    fields: dict[str, StoredFieldValue]
    metadata: dict[str, str]

MAGIC = b"DYF1"
MAGIC_V2 = b"DYF2"
MAGIC_V3 = b"DYF3"
HEADER_SIZE = 16  # 4 magic + 8 fb_size + 4 reserved
HEADER_SIZE_V3 = 32  # 4 magic + 8 fb_size + 2 n_chunks + 2 flags + 16 reserved
PAGE_SIZE = 4096


def detect_dyf_version(path: str) -> int:
    """Detect the DYF format version of a file.

    Returns:
        1 for DYF1 (header-based FlatBuffers),
        2 for DYF2 (footer-based FlatBuffers, append-friendly),
        3 for DYF3 (chunked, header-based like DYF1 but 32-byte header).

    Raises:
        ValueError if the file has an unknown magic.
    """
    with open(path, "rb") as f:
        magic = f.read(4)
    if magic == MAGIC:
        return 1
    elif magic == MAGIC_V2:
        return 2
    elif magic == MAGIC_V3:
        return 3
    else:
        raise ValueError(f"Unknown DYF magic: {magic!r}")


@dataclass
class SearchResult:
    """Search result with indices, scores, and optional stored fields."""
    indices: np.ndarray      # (k,) uint32
    scores: np.ndarray       # (k,) float32
    fields: dict = field(default_factory=dict)  # field_name -> (k,) values
    routing: dict | None = None  # routing diagnostics when return_routing=True

    def __iter__(self):
        """Backward-compatible unpacking: indices, scores = idx.search(...)"""
        yield self.indices
        yield self.scores

    def __getitem__(self, key):
        if isinstance(key, str):
            return self.fields[key]
        return (self.indices, self.scores)[key]

    def __len__(self):
        return 2


@dataclass
class AdaptiveProbeConfig:
    """Configuration for adaptive probe count based on routing margin.

    Queries with small routing margins (near decision boundaries) probe more
    leaves; confident queries (large margins) probe fewer. Linear interpolation
    between min_probes and max_probes based on where min_margin falls between
    margin_lo and margin_hi.
    """
    margin_lo: float = 0.01   # below this, always probe max
    margin_hi: float = 0.1    # above this, always probe min
    min_probes: int = 1
    max_probes: int = 5


def _build_flat_node(node_id, node, node_to_id, embeddings, embedding_dim, batch_idx):
    """Build a single flat node dict from a tree node.

    Args:
        node_id: BFS index of this node.
        node: Tree node dict with keys: children, indices, depth, etc.
        node_to_id: Mapping from tree node id() to BFS index.
        embeddings: (n, d) array of embedding vectors, or None.
        embedding_dim: Embedding dimension (used when embeddings is None).
        batch_idx: Current batch index for leaf assignment.

    Returns:
        Flat node dict with keys: children_ids, hyperplanes, num_bits,
        bucket_ids_to_children, centroid, num_items, batch_index, depth,
        is_leaf, indices, eigenvalues.
    """
    is_leaf = not node['children']

    # Use pre-computed centroid if available, else compute from embeddings
    indices = node['indices']
    if node.get('centroid') is not None:
        centroid = node['centroid']
    elif embeddings is not None:
        if len(indices) > 0:
            centroid = embeddings[indices].mean(axis=0).astype(np.float32)
            norm = np.linalg.norm(centroid)
            if norm > 1e-10:
                centroid /= norm
        else:
            centroid = np.zeros(embeddings.shape[1], dtype=np.float32)
    else:
        # No embeddings and no pre-computed centroid — use zero vector
        dim = embedding_dim or 0
        centroid = np.zeros(dim, dtype=np.float32)

    # Get children IDs
    children_ids = []
    for child in node['children']:
        children_ids.append(node_to_id[id(child)])

    # Hyperplanes and bucket mapping
    hp = node.get('hyperplanes')
    bid_map = node.get('bucket_id_to_child')

    if hp is not None:
        num_bits = hp.shape[0]
        hyperplanes_flat = hp.flatten().astype(np.float32)
    else:
        num_bits = 0
        hyperplanes_flat = None

    # Eigenvalues
    ev = node.get('eigenvalues')
    eigenvalues_flat = ev.astype(np.float32) if ev is not None else None

    # Build bucket_ids_to_children parallel array
    # For each child (in order), store the bucket_id that maps to it
    bucket_ids_to_children = []
    if bid_map is not None and children_ids:
        # bid_map: {bucket_id: child_index_in_parent}
        # We need parallel array: bucket_id for child 0, child 1, ...
        reverse_map = {v: k for k, v in bid_map.items()}
        for child_idx in range(len(children_ids)):
            bucket_ids_to_children.append(reverse_map.get(child_idx, 0))

    # Batch index for leaves
    bi = batch_idx if is_leaf else -1

    return {
        'children_ids': children_ids,
        'hyperplanes': hyperplanes_flat,
        'num_bits': num_bits,
        'bucket_ids_to_children': bucket_ids_to_children,
        'centroid': centroid,
        'num_items': len(indices),
        'batch_index': bi,
        'depth': node['depth'],
        'is_leaf': is_leaf,
        'indices': indices,
        'eigenvalues': eigenvalues_flat,
    }


def _flatten_tree_bfs(tree, embeddings, embedding_dim=None):
    """Flatten tree to a BFS-ordered list of node dicts with computed fields.

    Args:
        tree: Tree dict from build_dyf_tree() or _reconstruct_tree().
        embeddings: (n, d) array of embedding vectors, or None if
            embeddings are being dropped (centroids must be in tree).
        embedding_dim: Required when embeddings is None, to size zero-centroids.

    Returns:
        (flat_nodes, leaf_batch_map)

        flat_nodes: list of dicts with keys:
            children_ids, hyperplanes, num_bits, bucket_ids_to_children,
            centroid, num_items, batch_index, depth, is_leaf
        leaf_batch_map: dict mapping flat node index to batch index
    """
    # BFS assignment of node IDs
    flat_nodes = []
    node_to_id = {}
    queue = deque()

    # First pass: assign IDs via BFS
    queue.append(tree)
    while queue:
        node = queue.popleft()
        node_id = len(flat_nodes)
        node_to_id[id(node)] = node_id
        flat_nodes.append(node)
        for child in node['children']:
            queue.append(child)

    # Second pass: build flat representations
    result = []
    batch_idx = 0
    leaf_batch_map = {}

    for node_id, node in enumerate(flat_nodes):
        flat_node = _build_flat_node(
            node_id, node, node_to_id, embeddings, embedding_dim, batch_idx)
        if flat_node['is_leaf']:
            leaf_batch_map[node_id] = batch_idx
            batch_idx += 1
        result.append(flat_node)

    return result, leaf_batch_map


def _quantize_embeddings(embeddings, quantization):
    """Quantize embeddings to the requested precision."""
    if quantization == 'float32':
        return embeddings.astype(np.float32)
    elif quantization == 'float16':
        return embeddings.astype(np.float16)
    elif quantization == 'int8':
        # Scale to [-127, 127]
        max_val = np.abs(embeddings).max()
        if max_val > 0:
            scale = 127.0 / max_val
            return np.clip(np.round(embeddings * scale), -127, 127).astype(np.int8)
        return np.zeros_like(embeddings, dtype=np.int8)
    else:
        raise ValueError(f"Unknown quantization: {quantization}")


def _numpy_dtype_to_arrow(quantization):
    """Map quantization string to pyarrow type."""
    import pyarrow as pa
    if quantization == 'float32':
        return pa.float32()
    elif quantization == 'float16':
        return pa.float16()
    elif quantization == 'int8':
        return pa.int8()
    raise ValueError(f"Unknown quantization: {quantization}")


def _train_pq(embeddings, n_subquantizers):
    """Train a FAISS ProductQuantizer on (possibly subsampled) embeddings.

    Args:
        embeddings: (n, d) float32 array (will be L2-normalized).
        n_subquantizers: Number of sub-quantizers M (dim must be divisible by M).

    Returns:
        (pq, codebook) — faiss.ProductQuantizer and (M, 256, dsub) float32 codebook.
    """
    import faiss

    n, d = embeddings.shape
    dsub = d // n_subquantizers

    # L2-normalize
    emb = embeddings.astype(np.float32).copy()
    faiss.normalize_L2(emb)

    # Subsample to 1M for training if dataset is larger
    max_train = 1_000_000
    if n > max_train:
        rng = np.random.default_rng(42)
        idx = rng.choice(n, max_train, replace=False)
        train_data = emb[idx]
    else:
        train_data = emb

    pq = faiss.ProductQuantizer(d, n_subquantizers, 8)  # 8 bits = 256 centroids
    pq.train(train_data)

    # Extract codebook as (M, 256, dsub)
    centroids_flat = faiss.vector_to_array(pq.centroids)
    codebook = centroids_flat.reshape(n_subquantizers, 256, dsub).copy()

    return pq, codebook


def _encode_pq(embeddings, pq):
    """Encode embeddings to PQ codes.

    Args:
        embeddings: (n, d) float32 array (will be L2-normalized).
        pq: Trained faiss.ProductQuantizer.

    Returns:
        (n, M) uint8 codes.
    """
    import faiss

    emb = embeddings.astype(np.float32).copy()
    faiss.normalize_L2(emb)
    codes = pq.compute_codes(emb)  # (n, M) uint8
    return codes


def _serialize_codebook(codebook):
    """Serialize codebook to base64 string for metadata storage.

    Args:
        codebook: (M, 256, dsub) float32 array.

    Returns:
        Base64 encoded string.
    """
    import base64
    return base64.b64encode(codebook.astype(np.float32).tobytes()).decode('ascii')


def _deserialize_codebook(b64_str, n_subquantizers, dsub):
    """Deserialize codebook from base64 string.

    Args:
        b64_str: Base64 encoded string.
        n_subquantizers: Number of sub-quantizers M.
        dsub: Sub-vector dimension.

    Returns:
        (M, 256, dsub) float32 codebook.
    """
    import base64
    raw = base64.b64decode(b64_str)
    return np.frombuffer(raw, dtype=np.float32).reshape(n_subquantizers, 256, dsub).copy()


def _infer_arrow_type(values):
    """Infer Arrow type from a Python/numpy array of values.

    Args:
        values: array-like of field values.

    Returns:
        (pa_type, type_name) — pyarrow type and string identifier for metadata.
    """
    import pyarrow as pa

    values = np.asarray(values) if not isinstance(values, np.ndarray) else values

    # numpy arrays: match dtype
    if values.dtype == np.float32:
        return pa.float32(), 'float32'
    elif values.dtype == np.float64:
        return pa.float64(), 'float64'
    elif values.dtype == np.int32:
        return pa.int32(), 'int32'
    elif values.dtype == np.int64:
        return pa.int64(), 'int64'
    elif values.dtype.kind in ('U', 'O'):
        # String or object array — check first non-None value
        for v in values:
            if v is not None:
                if isinstance(v, bytes):
                    return pa.binary(), 'binary'
                return pa.utf8(), 'utf8'
        return pa.utf8(), 'utf8'
    elif values.dtype.kind == 'S':
        return pa.binary(), 'binary'
    else:
        raise ValueError(f"Unsupported stored field dtype: {values.dtype}")


def _resolve_arrow_schema(has_embeddings, embeddings, quantization,
                          embedding_dim, metadata, stored_fields):
    """Build the Arrow schema and prepare PQ state for write_lazy_index.

    Args:
        has_embeddings: Whether embeddings are present.
        embeddings: (n, d) float32 array, or None.
        quantization: Quantization string (e.g. 'float16', 'pq-8').
        embedding_dim: Embedding dimension.
        metadata: Mutable metadata dict (may be modified in place).
        stored_fields: Optional dict of stored field arrays.

    Returns:
        (arrow_schema, sf_types, metadata, pq_state) where pq_state is a dict
        with keys: is_pq, all_codes, n_subquantizers, q_embeddings.
    """
    import pyarrow as pa

    is_pq = has_embeddings and quantization.startswith('pq')
    pq_state = {
        'is_pq': is_pq,
        'all_codes': None,
        'n_subquantizers': 0,
        'q_embeddings': None,
    }

    if not has_embeddings:
        # No embeddings — schema has item_index + stored fields only
        arrow_schema = pa.schema([
            ('item_index', pa.uint32()),
        ])
        if metadata is None:
            metadata = {}
        metadata['has_embeddings'] = 'false'
    elif is_pq:
        if quantization == 'pq':
            # Auto-select M = dim // 4 (dsub=4, canonical PQ default)
            n_subquantizers = embedding_dim // 4
            if n_subquantizers < 1:
                raise ValueError(
                    f"embedding_dim={embedding_dim} too small for PQ "
                    f"(need at least 4)")
        else:
            # Parse M from e.g. "pq-8" -> M=8
            n_subquantizers = int(quantization.split('-')[1])
        if embedding_dim % n_subquantizers != 0:
            raise ValueError(
                f"embedding_dim={embedding_dim} must be divisible by "
                f"n_subquantizers={n_subquantizers} for PQ quantization")
        dsub = embedding_dim // n_subquantizers

        # Train PQ on full dataset, encode all embeddings
        pq, codebook = _train_pq(embeddings, n_subquantizers)
        all_codes = _encode_pq(embeddings, pq)  # (n_items, M) uint8

        # Store PQ params in metadata
        if metadata is None:
            metadata = {}
        metadata['pq_codebook'] = _serialize_codebook(codebook)
        metadata['pq_n_subquantizers'] = str(n_subquantizers)
        metadata['pq_dsub'] = str(dsub)

        # Arrow schema for PQ: codes stored as FixedSizeList(uint8, M)
        arrow_schema = pa.schema([
            ('item_index', pa.uint32()),
            ('embedding', pa.list_(pa.uint8(), n_subquantizers)),
        ])
        pq_state['all_codes'] = all_codes
        pq_state['n_subquantizers'] = n_subquantizers
    else:
        # Standard quantization path
        q_embeddings = _quantize_embeddings(embeddings, quantization)
        arrow_value_type = _numpy_dtype_to_arrow(quantization)
        arrow_schema = pa.schema([
            ('item_index', pa.uint32()),
            ('embedding', pa.list_(arrow_value_type, embedding_dim)),
        ])
        pq_state['q_embeddings'] = q_embeddings

    # Process stored fields: detect types, extend schema, store metadata
    sf_types = {}  # field_name -> (pa_type, type_name)
    if stored_fields:
        for fname, fvalues in stored_fields.items():
            pa_type, type_name = _infer_arrow_type(fvalues)
            sf_types[fname] = (pa_type, type_name)
            arrow_schema = arrow_schema.append(pa.field(fname, pa_type))
        # Store field schema in metadata
        if metadata is None:
            metadata = {}
        metadata['stored_fields'] = json.dumps(
            {fname: tname for fname, (_, tname) in sf_types.items()})

    return arrow_schema, sf_types, metadata, pq_state


def _build_leaf_batches(flat_nodes, arrow_schema, has_embeddings, quantization,
                        stored_fields, sf_types, pq_state, compression):
    """Build Arrow IPC byte buffers for each leaf node.

    Args:
        flat_nodes: BFS-ordered list of flat node dicts.
        arrow_schema: Arrow schema for the record batches.
        has_embeddings: Whether embeddings are present.
        quantization: Quantization string.
        stored_fields: Optional dict of stored field arrays.
        sf_types: Dict mapping field name to (pa_type, type_name).
        pq_state: Dict with PQ state (is_pq, all_codes, n_subquantizers,
            q_embeddings).
        compression: Compression string ('none', 'zstd', 'lz4').

    Returns:
        List of Arrow IPC buffer bytes, one per leaf.
    """
    import pyarrow as pa

    is_pq = pq_state['is_pq']
    all_codes = pq_state['all_codes']
    n_subquantizers = pq_state['n_subquantizers']
    q_embeddings = pq_state['q_embeddings']
    embedding_dim = arrow_schema.field('embedding').type.list_size if has_embeddings and not is_pq else 0

    ipc_options = None
    if compression == 'zstd':
        ipc_options = pa.ipc.IpcWriteOptions(
            compression=pa.Codec('zstd'))
    elif compression == 'lz4':
        ipc_options = pa.ipc.IpcWriteOptions(
            compression=pa.Codec('lz4'))

    batch_buffers = []
    for node in flat_nodes:
        if not node['is_leaf']:
            continue
        indices = node['indices']

        # Build Arrow RecordBatch
        item_idx_arr = pa.array(indices.astype(np.uint32), type=pa.uint32())

        columns = [item_idx_arr]

        if has_embeddings:
            if is_pq:
                # PQ path: slice pre-encoded codes for this leaf
                leaf_codes = all_codes[indices]  # (n_leaf, M) uint8
                flat_codes = leaf_codes.flatten()
                values_arr = pa.array(flat_codes, type=pa.uint8())
                emb_arr = pa.FixedSizeListArray.from_arrays(
                    values_arr, n_subquantizers)
            else:
                # Standard path: quantized embeddings
                leaf_emb = q_embeddings[indices]
                flat_values = leaf_emb.flatten()
                if quantization == 'int8':
                    values_arr = pa.array(flat_values, type=pa.int8())
                elif quantization == 'float16':
                    values_arr = pa.array(flat_values, type=pa.float16())
                else:
                    values_arr = pa.array(flat_values, type=pa.float32())
                emb_arr = pa.FixedSizeListArray.from_arrays(
                    values_arr, embedding_dim)
            columns.append(emb_arr)
        if stored_fields:
            for fname in sf_types:
                fvalues = stored_fields[fname]
                # Slice to this leaf's indices
                if isinstance(fvalues, np.ndarray):
                    leaf_vals = fvalues[indices]
                else:
                    # list-like: index manually
                    leaf_vals = [fvalues[i] for i in indices]
                pa_type = sf_types[fname][0]
                columns.append(pa.array(leaf_vals, type=pa_type))

        batch = pa.record_batch(columns, schema=arrow_schema)

        # Write to IPC stream bytes
        sink = pa.BufferOutputStream()
        writer = pa.ipc.new_stream(sink, arrow_schema,
                                   options=ipc_options)
        writer.write_batch(batch)
        writer.close()
        batch_buffers.append(sink.getvalue())

    return batch_buffers


def _build_flatbuffer_index(flat_nodes, batch_buffers, metadata, build_params,
                            tree, quantization, compression, embedding_dim,
                            n_items, n_leaves, format_version):
    """Build the FlatBuffer index bytes.

    Args:
        flat_nodes: BFS-ordered list of flat node dicts.
        batch_buffers: List of Arrow IPC buffer bytes.
        metadata: Dict of metadata key-value pairs, or None.
        build_params: Dict of build parameters, or None.
        tree: Original tree dict (for depth fallback).
        quantization: Quantization string.
        compression: Compression string.
        embedding_dim: Embedding dimension.
        n_items: Total number of items.
        n_leaves: Number of leaf nodes.
        format_version: File format version (1, 2, or 3).

    Returns:
        FlatBuffer bytes.
    """
    import flatbuffers

    from dyf.schema import (
        BatchDescriptor as FBBatch,
    )
    from dyf.schema import (
        BuildParams as FBBuildParams,
    )
    from dyf.schema import (
        Index as FBIndex,
    )
    from dyf.schema import (
        KeyValue as FBKeyValue,
    )
    from dyf.schema import (
        Node as FBNode,
    )

    builder = flatbuffers.Builder(4096)

    # Pre-build all strings and vectors
    ver_str = {1: "1.0", 2: "2.0", 3: "3.0"}.get(format_version, "1.0")
    version_off = builder.CreateString(ver_str)
    quant_off = builder.CreateString(quantization)
    comp_off = builder.CreateString(compression)

    # Metadata key-value pairs
    kv_offsets = []
    if metadata:
        for key, value in metadata.items():
            k_off = builder.CreateString(str(key))
            v_off = builder.CreateString(str(value))
            FBKeyValue.KeyValueStart(builder)
            FBKeyValue.KeyValueAddKey(builder, k_off)
            FBKeyValue.KeyValueAddValue(builder, v_off)
            kv_offsets.append(FBKeyValue.KeyValueEnd(builder))

    # Build nodes
    node_offsets = []
    for fnode in flat_nodes:
        # Children vector
        if fnode['children_ids']:
            FBNode.NodeStartChildrenVector(builder, len(fnode['children_ids']))
            for cid in reversed(fnode['children_ids']):
                builder.PrependUint32(cid)
            children_vec = builder.EndVector()
        else:
            children_vec = None

        # Hyperplanes vector
        if fnode['hyperplanes'] is not None:
            hp = fnode['hyperplanes']
            FBNode.NodeStartHyperplanesVector(builder, len(hp))
            for val in reversed(hp):
                builder.PrependFloat32(float(val))
            hp_vec = builder.EndVector()
        else:
            hp_vec = None

        # Bucket IDs to children vector
        if fnode['bucket_ids_to_children']:
            bids = fnode['bucket_ids_to_children']
            FBNode.NodeStartBucketIdsToChildrenVector(builder, len(bids))
            for bid in reversed(bids):
                builder.PrependUint64(int(bid))
            bids_vec = builder.EndVector()
        else:
            bids_vec = None

        # Centroid vector
        cent = fnode['centroid']
        FBNode.NodeStartCentroidVector(builder, len(cent))
        for val in reversed(cent):
            builder.PrependFloat32(float(val))
        cent_vec = builder.EndVector()

        # Eigenvalues vector
        if fnode.get('eigenvalues') is not None:
            ev = fnode['eigenvalues']
            FBNode.NodeStartEigenvaluesVector(builder, len(ev))
            for val in reversed(ev):
                builder.PrependFloat32(float(val))
            ev_vec = builder.EndVector()
        else:
            ev_vec = None

        # Build node
        FBNode.NodeStart(builder)
        if children_vec is not None:
            FBNode.NodeAddChildren(builder, children_vec)
        if hp_vec is not None:
            FBNode.NodeAddHyperplanes(builder, hp_vec)
        FBNode.NodeAddNumBits(builder, fnode['num_bits'])
        if bids_vec is not None:
            FBNode.NodeAddBucketIdsToChildren(builder, bids_vec)
        FBNode.NodeAddCentroid(builder, cent_vec)
        FBNode.NodeAddNumItems(builder, fnode['num_items'])
        FBNode.NodeAddBatchIndex(builder, fnode['batch_index'])
        FBNode.NodeAddDepth(builder, fnode['depth'])
        if ev_vec is not None:
            FBNode.NodeAddEigenvalues(builder, ev_vec)
        node_offsets.append(FBNode.NodeEnd(builder))

    # Nodes vector
    FBIndex.IndexStartNodesVector(builder, len(node_offsets))
    for off in reversed(node_offsets):
        builder.PrependUOffsetTRelative(off)
    nodes_vec = builder.EndVector()

    # Batch descriptors (offsets computed later — store placeholders)
    # We need to know the arrow section start to compute offsets.
    # Approach: record batch sizes, compute offsets relative to arrow start.
    batch_offsets = []
    running_offset = 0
    for buf in batch_buffers:
        batch_offsets.append(running_offset)
        running_offset += len(buf)

    batch_desc_offsets = []
    for i, buf in enumerate(batch_buffers):
        FBBatch.BatchDescriptorStart(builder)
        FBBatch.BatchDescriptorAddOffset(builder, batch_offsets[i])
        FBBatch.BatchDescriptorAddLength(builder, len(buf))
        # Count rows from the flat_nodes
        leaf_nodes = [n for n in flat_nodes if n['is_leaf']]
        FBBatch.BatchDescriptorAddNumRows(builder, len(leaf_nodes[i]['indices']))
        batch_desc_offsets.append(FBBatch.BatchDescriptorEnd(builder))

    FBIndex.IndexStartBatchesVector(builder, len(batch_desc_offsets))
    for off in reversed(batch_desc_offsets):
        builder.PrependUOffsetTRelative(off)
    batches_vec = builder.EndVector()

    # BuildParams
    bp = build_params or {}
    FBBuildParams.BuildParamsStart(builder)
    FBBuildParams.BuildParamsAddMaxDepth(builder, bp.get('max_depth', tree['depth']))
    FBBuildParams.BuildParamsAddNumBits(builder, bp.get('num_bits', 3))
    FBBuildParams.BuildParamsAddMinLeafSize(builder, bp.get('min_leaf_size', 4))
    FBBuildParams.BuildParamsAddSeed(builder, bp.get('seed', 42))
    FBBuildParams.BuildParamsAddQuantization(builder, quant_off)
    FBBuildParams.BuildParamsAddCompression(builder, comp_off)
    bp_off = FBBuildParams.BuildParamsEnd(builder)

    # Metadata vector
    if kv_offsets:
        FBIndex.IndexStartMetadataVector(builder, len(kv_offsets))
        for off in reversed(kv_offsets):
            builder.PrependUOffsetTRelative(off)
        meta_vec = builder.EndVector()
    else:
        meta_vec = None

    # Root Index table
    FBIndex.IndexStart(builder)
    FBIndex.IndexAddVersion(builder, version_off)
    FBIndex.IndexAddEmbeddingDim(builder, embedding_dim)
    FBIndex.IndexAddTotalItems(builder, n_items)
    FBIndex.IndexAddNumLeaves(builder, n_leaves)
    FBIndex.IndexAddRoot(builder, 0)  # Root is always node 0 in BFS
    FBIndex.IndexAddNodes(builder, nodes_vec)
    FBIndex.IndexAddBatches(builder, batches_vec)
    FBIndex.IndexAddBuildParams(builder, bp_off)
    if meta_vec is not None:
        FBIndex.IndexAddMetadata(builder, meta_vec)
    index_off = FBIndex.IndexEnd(builder)

    builder.Finish(index_off)
    return bytes(builder.Output())


def _write_dyf_file(path, format_version, fb_bytes, batch_buffers):
    """Write assembled DYF file in the specified format.

    Args:
        path: Output file path.
        format_version: File format version (1, 2, or 3).
        fb_bytes: Serialized FlatBuffer index bytes.
        batch_buffers: List of Arrow IPC buffer bytes.
    """
    fb_size = len(fb_bytes)

    if format_version == 2:
        # DYF2: header(8) + arrow batches + FlatBuffers + footer(16)
        with open(path, 'wb') as f:
            # 8-byte header: magic + flags
            f.write(MAGIC_V2)
            f.write(struct.pack('<I', 0))  # flags (reserved)

            # Arrow IPC batches
            for buf in batch_buffers:
                f.write(buf)

            # FlatBuffers index
            fb_offset = f.tell()
            f.write(fb_bytes)

            # 16-byte footer: fb_offset (u64 LE) + fb_size (u64 LE)
            f.write(struct.pack('<Q', fb_offset))
            f.write(struct.pack('<Q', fb_size))
    elif format_version == 3:
        # DYF3: header(32) + FlatBuffers + padding + arrow batches
        # Same layout as DYF1 but with 32-byte header supporting chunked reads
        total_header_fb = HEADER_SIZE_V3 + fb_size
        padding = (PAGE_SIZE - (total_header_fb % PAGE_SIZE)) % PAGE_SIZE

        with open(path, 'wb') as f:
            # 32-byte header
            f.write(MAGIC_V3)                      # 4: magic
            f.write(struct.pack('<Q', fb_size))     # 8: fb_size (u64 LE)
            f.write(struct.pack('<H', 1))           # 2: n_chunks (u16 LE, 1 = single file)
            f.write(struct.pack('<H', 0))           # 2: flags (reserved)
            f.write(b'\x00' * 16)                   # 16: reserved

            # FlatBuffers
            f.write(fb_bytes)

            # Padding to 4KB boundary
            if padding > 0:
                f.write(b'\x00' * padding)

            # Arrow IPC batches
            for buf in batch_buffers:
                f.write(buf)
    else:
        # DYF1: header(16) + FlatBuffers + padding + arrow batches
        total_header_fb = HEADER_SIZE + fb_size
        padding = (PAGE_SIZE - (total_header_fb % PAGE_SIZE)) % PAGE_SIZE

        with open(path, 'wb') as f:
            # Header
            f.write(MAGIC)
            f.write(struct.pack('<Q', fb_size))
            f.write(struct.pack('<I', 0))  # reserved

            # FlatBuffers
            f.write(fb_bytes)

            # Padding to 4KB boundary
            if padding > 0:
                f.write(b'\x00' * padding)

            # Arrow IPC batches
            for buf in batch_buffers:
                f.write(buf)


def write_lazy_index(
    tree: dict,
    embeddings: np.ndarray | None,
    path: str,
    compression: str = 'none',
    quantization: str = 'float16',
    metadata: dict[str, str] | None = None,
    build_params: dict[str, int] | None = None,
    stored_fields: Mapping[str, StoredFieldInput] | None = None,
    format_version: int = 1,
    embedding_dim: int | None = None,
) -> None:
    """Write a DYF lazy index file (FlatBuffers tree + Arrow IPC leaf data).

    Args:
        tree: Tree dict from build_dyf_tree() (must include hyperplanes/bucket
            mapping from updated _build_dyf_tree).
        embeddings: (n, d) array of embedding vectors, or None to write
            a viz-only file without embeddings (requires embedding_dim).
        path: Output file path (e.g. "index.dyf").
        compression: "none", "zstd", or "lz4" (default: "none").
        quantization: "float32", "float16", "int8", or "pq-M" where M is the
            number of sub-quantizers (default: "float16"). For PQ, dim must
            be divisible by M. Example: "pq-8" for 8 sub-quantizers.
        metadata: Optional dict of string key-value pairs.
        build_params: Optional dict with keys: max_depth, num_bits,
            min_leaf_size, seed. Auto-detected from tree if not provided.
        stored_fields: Optional dict mapping field name to array-like of
            length n_items. Supported types: str/list[str] (Arrow utf8),
            np.int32/int64/float32/float64 arrays, list[bytes] (Arrow binary).
        format_version: 1 (DYF1, header-based) or 2 (DYF2, footer-based,
            append-friendly). Default: 1.
        embedding_dim: Embedding dimension. Required when embeddings is None.
            Inferred from embeddings shape when provided.
    """
    has_embeddings = embeddings is not None
    if has_embeddings:
        embeddings = np.asarray(embeddings, dtype=np.float32)
        n_items, embedding_dim = embeddings.shape
    else:
        if embedding_dim is None:
            raise ValueError(
                "embedding_dim is required when embeddings is None")
        # Count items from tree leaves
        n_items = len(tree['indices'])

    # Flatten tree to BFS node list
    flat_nodes, _leaf_batch_map = _flatten_tree_bfs(
        tree, embeddings, embedding_dim=embedding_dim)
    n_leaves = sum(1 for n in flat_nodes if n['is_leaf'])

    # Build Arrow schema and prepare PQ/quantization state
    arrow_schema, sf_types, metadata, pq_state = _resolve_arrow_schema(
        has_embeddings, embeddings, quantization, embedding_dim,
        metadata, stored_fields)

    # Build Arrow IPC batches per leaf
    batch_buffers = _build_leaf_batches(
        flat_nodes, arrow_schema, has_embeddings, quantization,
        stored_fields, sf_types, pq_state, compression)

    # Build FlatBuffer index
    fb_bytes = _build_flatbuffer_index(
        flat_nodes, batch_buffers, metadata, build_params, tree,
        quantization, compression, embedding_dim, n_items, n_leaves,
        format_version)

    # Write file
    _write_dyf_file(path, format_version, fb_bytes, batch_buffers)


def split_dyf3(
    path: str,
    chunk_size: int = 95_000_000,
) -> int:
    """Split a DYF3 file into chunks for transport (e.g. GitHub Pages).

    Chunk 0 keeps the original filename (truncated to chunk_size).
    Chunks 1..N are written as ``path.1``, ``path.2``, etc. with raw
    continuation bytes (no headers).  The ``n_chunks`` field in chunk 0's
    header is patched in place.

    Args:
        path: Path to a DYF3 file (must already be format version 3).
        chunk_size: Maximum bytes per chunk.  Default 95 MB (fits GitHub's
            100 MB per-file limit with margin).

    Returns:
        Number of chunks (1 means no split was needed).

    Raises:
        ValueError: If the file is not DYF3.
    """
    import math
    import os

    with open(path, 'rb') as f:
        magic = f.read(4)
    if magic != MAGIC_V3:
        raise ValueError(f"split_dyf3 requires a DYF3 file, got magic {magic!r}")

    file_size = os.path.getsize(path)
    if file_size <= chunk_size:
        return 1

    n_chunks = math.ceil(file_size / chunk_size)
    if n_chunks > 65535:
        raise ValueError(
            f"Too many chunks ({n_chunks}); increase chunk_size or reduce file")

    # Write companion chunks (1..N) as raw byte slices
    with open(path, 'rb') as f:
        for ci in range(1, n_chunks):
            offset = ci * chunk_size
            f.seek(offset)
            data = f.read(chunk_size)
            chunk_path = f"{path}.{ci}"
            with open(chunk_path, 'wb') as cf:
                cf.write(data)

    # Truncate original to chunk_size
    with open(path, 'r+b') as f:
        f.truncate(chunk_size)

    # Patch n_chunks in header (offset 12, u16 LE)
    with open(path, 'r+b') as f:
        f.seek(12)
        f.write(struct.pack('<H', n_chunks))

    return n_chunks


def _merge_leaf_results(all_indices, all_scores, all_fields, sf_names):
    """Concatenate and deduplicate results from multiple leaf batches.

    Args:
        all_indices: List of index arrays from each leaf.
        all_scores: List of score arrays from each leaf.
        all_fields: Dict mapping field name to list of value arrays/lists.
        sf_names: List of stored field names.

    Returns:
        (deduped_indices, deduped_scores, merged_fields) with duplicates removed.
    """
    all_indices = np.concatenate(all_indices)
    all_scores = np.concatenate(all_scores)
    # Concatenate stored fields
    merged_fields = {}
    for fname in sf_names:
        parts = all_fields[fname]
        if isinstance(parts[0], list):
            merged_fields[fname] = sum(parts, [])
        else:
            merged_fields[fname] = np.concatenate(parts)

    # Deduplicate (same item may appear in overlapping leaves)
    unique_mask = np.ones(len(all_indices), dtype=bool)
    seen = set()
    for i, idx in enumerate(all_indices):
        idx_int = int(idx)
        if idx_int in seen:
            unique_mask[i] = False
        else:
            seen.add(idx_int)
    all_indices = all_indices[unique_mask]
    all_scores = all_scores[unique_mask]
    for fname in sf_names:
        vals = merged_fields[fname]
        if isinstance(vals, np.ndarray):
            merged_fields[fname] = vals[unique_mask]
        else:
            merged_fields[fname] = [v for v, m in zip(vals, unique_mask) if m]

    return all_indices, all_scores, merged_fields


def _topk_with_fields(all_indices, all_scores, merged_fields, sf_names, k):
    """Select top-k results and slice stored fields to match.

    Args:
        all_indices: Deduplicated index array.
        all_scores: Deduplicated score array.
        merged_fields: Dict mapping field name to merged values.
        sf_names: List of stored field names.
        k: Number of top results to return.

    Returns:
        (top_indices, top_scores, result_fields) sorted by descending score.
    """
    if len(all_scores) <= k:
        order = np.argsort(-all_scores)
    else:
        # Partial sort for efficiency
        part_idx = np.argpartition(-all_scores, k)[:k]
        order = part_idx[np.argsort(-all_scores[part_idx])]

    result_fields = {}
    for fname in sf_names:
        vals = merged_fields[fname]
        if isinstance(vals, np.ndarray):
            result_fields[fname] = vals[order]
        else:
            result_fields[fname] = [vals[i] for i in order]

    return all_indices[order], all_scores[order], result_fields


class LazyIndex:
    """Memory-mapped DYF index for instant-start query serving.

    Opens a .dyf file, mmap's it, parses the FlatBuffers header, and
    serves queries by traversing the tree and decompressing Arrow IPC
    batches on demand (LRU cached).

    Usage:
        idx = LazyIndex("index.dyf")
        results = idx.search(query_vector, k=10, nprobe=3)
        print(idx.tree_summary)
    """

    def __init__(self, path):
        self._path = path
        self._file = open(path, 'rb')
        self._mm = mmap.mmap(self._file.fileno(), 0, access=mmap.ACCESS_READ)
        self._extra_files = []  # track companion chunk file handles

        # Parse header — detect format version
        magic = self._mm[:4]
        if magic == MAGIC:
            self._format_version = 1
        elif magic == MAGIC_V2:
            self._format_version = 2
        elif magic == MAGIC_V3:
            self._format_version = 3
        else:
            raise ValueError(
                f"Invalid magic: {magic!r}, expected {MAGIC!r}, "
                f"{MAGIC_V2!r}, or {MAGIC_V3!r}"
            )

        if self._format_version == 3:
            # DYF3: 32-byte header, same layout as DYF1 but supports chunks
            self._fb_size = struct.unpack_from('<Q', self._mm, 4)[0]
            n_chunks = struct.unpack_from('<H', self._mm, 12)[0]

            if n_chunks > 1:
                # Multi-chunk: concatenate all chunks into a bytearray
                import os
                base_path = path
                buf = bytearray(self._mm[:])
                self._mm.close()
                self._file.close()
                self._mm = None
                self._file = None

                for ci in range(1, n_chunks):
                    chunk_path = f"{base_path}.{ci}"
                    if not os.path.exists(chunk_path):
                        raise FileNotFoundError(
                            f"DYF3 chunk {ci} not found: {chunk_path}")
                    with open(chunk_path, 'rb') as cf:
                        buf.extend(cf.read())

                self._mm = memoryview(buf)
            # else: n_chunks == 1, mmap is fine as-is

            fb_start = HEADER_SIZE_V3
            fb_buf = bytes(self._mm[fb_start:fb_start + self._fb_size])

            from dyf.schema.Index import Index as FBIndex
            self._index = FBIndex.GetRootAs(fb_buf, 0)

            total_header_fb = HEADER_SIZE_V3 + self._fb_size
            self._arrow_start = total_header_fb + (
                (PAGE_SIZE - (total_header_fb % PAGE_SIZE)) % PAGE_SIZE)

        elif self._format_version == 1:
            self._fb_size = struct.unpack_from('<Q', self._mm, 4)[0]

            # Parse FlatBuffers (at start of file)
            fb_start = HEADER_SIZE
            fb_end = fb_start + self._fb_size
            fb_buf = self._mm[fb_start:fb_end]

            from dyf.schema.Index import Index as FBIndex
            self._index = FBIndex.GetRootAs(fb_buf, 0)

            # Compute arrow section start
            total_header_fb = HEADER_SIZE + self._fb_size
            self._arrow_start = total_header_fb + (
                (PAGE_SIZE - (total_header_fb % PAGE_SIZE)) % PAGE_SIZE)
        else:
            # DYF2: FlatBuffers in footer, Arrow batches after 8-byte header
            file_len = len(self._mm)
            # Read 16-byte footer (last 16 bytes)
            fb_offset = struct.unpack_from('<Q', self._mm, file_len - 16)[0]
            self._fb_size = struct.unpack_from('<Q', self._mm, file_len - 8)[0]

            fb_buf = self._mm[fb_offset:fb_offset + self._fb_size]

            from dyf.schema.Index import Index as FBIndex
            self._index = FBIndex.GetRootAs(fb_buf, 0)

            # Arrow data starts right after the 8-byte header
            self._arrow_start = 8

        # Cache for decompressed batches
        self._batch_cache = {}
        self._cache_order = deque()
        self._cache_maxsize = 64

    def close(self):
        """Release mmap and file handle."""
        if self._mm is not None:
            if isinstance(self._mm, mmap.mmap):
                self._mm.close()
            elif isinstance(self._mm, memoryview):
                self._mm.release()
            self._mm = None
        if self._file is not None:
            self._file.close()
            self._file = None

    def __del__(self):
        self.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    @property
    def format_version(self) -> int:
        """DYF format version: 1 (header FB) or 2 (footer FB, append-friendly)."""
        return self._format_version

    @property
    def embedding_dim(self):
        return self._index.EmbeddingDim()

    @property
    def total_items(self):
        return self._index.TotalItems()

    @property
    def num_leaves(self):
        return self._index.NumLeaves()

    @property
    def num_nodes(self):
        return self._index.NodesLength()

    @property
    def stored_field_names(self):
        """List of stored field names (empty if no stored fields)."""
        meta = self._get_metadata()
        sf_json = meta.get('stored_fields')
        if not sf_json:
            return []
        return list(json.loads(sf_json).keys())

    @property
    def has_stored_fields(self):
        """True if this index has stored fields."""
        return len(self.stored_field_names) > 0

    @property
    def tree_summary(self):
        """Return tree stats without touching Arrow data."""
        bp = self._index.BuildParams()
        summary = {
            'version': self._index.Version().decode() if self._index.Version() else None,
            'embedding_dim': self.embedding_dim,
            'total_items': self.total_items,
            'num_leaves': self.num_leaves,
            'num_nodes': self.num_nodes,
            'build_params': {
                'max_depth': bp.MaxDepth() if bp else None,
                'num_bits': bp.NumBits() if bp else None,
                'min_leaf_size': bp.MinLeafSize() if bp else None,
                'seed': bp.Seed() if bp else None,
                'quantization': bp.Quantization().decode() if bp and bp.Quantization() else None,
                'compression': bp.Compression().decode() if bp and bp.Compression() else None,
            } if bp else None,
        }
        # Add PQ section if applicable
        if self.is_pq:
            meta = self._get_metadata()
            m = int(meta.get('pq_n_subquantizers', '0'))
            dsub = int(meta.get('pq_dsub', '0'))
            summary['pq'] = {
                'n_subquantizers': m,
                'dsub': dsub,
                'bytes_per_vector': m,
                'codebook_size_kb': round(m * 256 * dsub * 4 / 1024, 1),
            }
        # Add stored fields info
        sf_names = self.stored_field_names
        if sf_names:
            meta = self._get_metadata()
            sf_schema = json.loads(meta['stored_fields'])
            summary['stored_fields'] = [
                {'name': name, 'type': sf_schema[name]} for name in sf_names
            ]
        return summary

    def get_tree_structure(self) -> list[TreeNode]:
        """Export tree hierarchy for visualization (FlatBuffers only, no Arrow).

        Returns:
            list of dicts with keys: node_id, parent_id, depth, num_items,
            is_leaf, batch_index. Cached after first call.
        """
        if hasattr(self, '_tree_structure_cache'):
            return self._tree_structure_cache

        n_nodes = self._index.NodesLength()
        # Build parent map from children arrays
        parent_map = {}  # child_id -> parent_id
        for nid in range(n_nodes):
            node = self._index.Nodes(nid)
            if node is None:
                continue
            for ci in range(node.ChildrenLength()):
                child_id = node.Children(ci)
                parent_map[child_id] = nid

        result = []
        for nid in range(n_nodes):
            node = self._index.Nodes(nid)
            if node is None:
                continue
            is_leaf = node.ChildrenLength() == 0
            ev = None
            if not node.EigenvaluesIsNone() and node.EigenvaluesLength() > 0:
                ev = node.EigenvaluesAsNumpy().copy()
            result.append({
                'node_id': nid,
                'parent_id': parent_map.get(nid),  # None for root
                'depth': node.Depth(),
                'num_items': node.NumItems(),
                'is_leaf': is_leaf,
                'batch_index': node.BatchIndex() if is_leaf else -1,
                'eigenvalues': ev,
            })

        self._tree_structure_cache = result
        return result

    def get_split_hyperplanes(self) -> dict[int, np.ndarray]:
        """Return the PCA hyperplane(s) for each internal node.

        Returns:
            {node_id: (num_bits, dim) float32 array} for internal nodes only.
        """
        n_nodes = self._index.NodesLength()
        result = {}
        for nid in range(n_nodes):
            node = self._index.Nodes(nid)
            if node is None:
                continue
            hp = self._get_node_hyperplanes(node)
            if hp is not None:
                result[nid] = hp
        return result

    def get_split_eigenvalues(self) -> dict[int, np.ndarray]:
        """Return PCA eigenvalues for each internal node.

        Returns:
            {node_id: (num_bits,) float32 array} for internal nodes that
            have eigenvalues stored. Empty dict for old .dyf files.
        """
        n_nodes = self._index.NodesLength()
        result = {}
        for nid in range(n_nodes):
            node = self._index.Nodes(nid)
            if node is None:
                continue
            if not node.EigenvaluesIsNone() and node.EigenvaluesLength() > 0:
                result[nid] = np.array(
                    node.EigenvaluesAsNumpy(), dtype=np.float32)
        return result

    def _get_metadata(self):
        """Parse FlatBuffers KeyValue pairs into dict (cached)."""
        if hasattr(self, '_metadata_cache'):
            return self._metadata_cache
        meta = {}
        n = self._index.MetadataLength()
        for i in range(n):
            kv = self._index.Metadata(i)
            if kv is not None:
                key = kv.Key()
                value = kv.Value()
                if key is not None and value is not None:
                    meta[key.decode()] = value.decode()
        self._metadata_cache = meta
        return meta

    def _get_stored_field_types(self):
        """Parse stored field schema from metadata (cached).

        Returns:
            dict mapping field name to type string (e.g. 'utf8', 'float32').
            Empty dict if no stored fields.
        """
        if hasattr(self, '_sf_types_cache'):
            return self._sf_types_cache
        meta = self._get_metadata()
        sf_json = meta.get('stored_fields')
        if not sf_json:
            self._sf_types_cache = {}
        else:
            self._sf_types_cache = json.loads(sf_json)
        return self._sf_types_cache

    @property
    def quantization(self):
        """Read quantization string from BuildParams."""
        bp = self._index.BuildParams()
        if bp and bp.Quantization():
            return bp.Quantization().decode()
        return 'float16'

    @property
    def is_pq(self):
        """True if this index uses Product Quantization."""
        return self.quantization.startswith('pq')

    def _load_pq_codebook(self):
        """Deserialize PQ codebook from metadata (cached)."""
        if hasattr(self, '_pq_codebook'):
            return self._pq_codebook
        meta = self._get_metadata()
        m = int(meta['pq_n_subquantizers'])
        dsub = int(meta['pq_dsub'])
        self._pq_codebook = _deserialize_codebook(meta['pq_codebook'], m, dsub)
        return self._pq_codebook

    def _pq_precompute_tables(self, query_normed):
        """Precompute ADC distance tables for a query.

        Args:
            query_normed: (dim,) L2-normalized query vector.

        Returns:
            (M, 256) float32 inner product lookup tables.
        """
        codebook = self._load_pq_codebook()  # (M, 256, dsub)
        m = codebook.shape[0]
        dsub = codebook.shape[2]
        query_subs = query_normed.reshape(m, dsub)  # (M, dsub)
        # tables[j, c] = dot(query_sub_j, codebook[j, c])
        tables = np.einsum('md,mcd->mc', query_subs, codebook)  # (M, 256)
        return tables

    def _pq_adc_scores(self, tables, codes):
        """Compute approximate inner products via ADC table lookups.

        Args:
            tables: (M, 256) precomputed distance tables.
            codes: (n, M) uint8 PQ codes.

        Returns:
            (n,) float32 approximate cosine similarity scores.
        """
        m = tables.shape[0]
        n = codes.shape[0]
        scores = np.zeros(n, dtype=np.float32)
        for j in range(m):
            scores += tables[j, codes[:, j]]
        return scores

    def _score_leaf(self, batch, query_normed, adc_tables=None):
        """Score all items in a leaf batch against a query.

        Dispatches between PQ ADC scoring and standard matmul scoring.

        Args:
            batch: Arrow RecordBatch with columns: item_index, embedding,
                plus any stored field columns.
            query_normed: (dim,) L2-normalized query vector.
            adc_tables: (M, 256) precomputed ADC tables (required if is_pq).

        Returns:
            (item_indices, scores, field_data) — numpy arrays and dict of
            stored field arrays. field_data is {} if no stored fields.
        """
        item_indices = batch.column('item_index').to_numpy()
        emb_col = batch.column('embedding')
        n_rows = len(emb_col)
        flat_values = emb_col.values.to_numpy()

        if self.is_pq:
            meta = self._get_metadata()
            m = int(meta['pq_n_subquantizers'])
            codes = flat_values.reshape(n_rows, m)
            scores = self._pq_adc_scores(adc_tables, codes)
        else:
            dim = self.embedding_dim
            leaf_emb = flat_values.reshape(n_rows, dim).astype(np.float32)
            norms = np.linalg.norm(leaf_emb, axis=1, keepdims=True)
            leaf_emb_n = leaf_emb / np.maximum(norms, 1e-10)
            scores = leaf_emb_n @ query_normed

        # Extract stored fields
        field_data = {}
        sf_types = self._get_stored_field_types()
        for fname in sf_types:
            col = batch.column(fname)
            if sf_types[fname] in ('utf8', 'binary'):
                field_data[fname] = col.to_pylist()
            else:
                field_data[fname] = col.to_numpy()

        return item_indices, scores, field_data

    def _pq_reconstruct(self, codes):
        """Reconstruct approximate vectors from PQ codes.

        Args:
            codes: (n, M) uint8 PQ codes.

        Returns:
            (n, dim) float32 approximate vectors (NOT normalized).
        """
        codebook = self._load_pq_codebook()  # (M, 256, dsub)
        m, _, dsub = codebook.shape
        n = codes.shape[0]
        reconstructed = np.zeros((n, m * dsub), dtype=np.float32)
        for j in range(m):
            reconstructed[:, j * dsub:(j + 1) * dsub] = codebook[j, codes[:, j]]
        return reconstructed

    def get_leaf(self, batch_index):
        """Decompress and return a leaf's Arrow RecordBatch.

        Args:
            batch_index: The batch index (from node.BatchIndex()).

        Returns:
            pyarrow.RecordBatch with columns: item_index, embedding.
        """
        # Check cache
        if batch_index in self._batch_cache:
            return self._batch_cache[batch_index]

        import pyarrow as pa

        bd = self._index.Batches(batch_index)
        offset = bd.Offset()
        length = bd.Length()

        # Read compressed bytes from mmap
        start = self._arrow_start + offset
        end = start + length
        batch_bytes = self._mm[start:end]

        # Decompress via Arrow IPC
        reader = pa.ipc.open_stream(batch_bytes)
        batch = reader.read_next_batch()

        # LRU cache
        if len(self._batch_cache) >= self._cache_maxsize:
            evict_key = self._cache_order.popleft()
            self._batch_cache.pop(evict_key, None)
        self._batch_cache[batch_index] = batch
        self._cache_order.append(batch_index)

        return batch

    def _get_node_hyperplanes(self, node):
        """Extract hyperplanes from a FlatBuffers node as numpy array."""
        if node.HyperplanesIsNone() or node.HyperplanesLength() == 0:
            return None
        hp_flat = node.HyperplanesAsNumpy()
        num_bits = node.NumBits()
        if num_bits == 0:
            return None
        dim = self.embedding_dim
        return np.array(hp_flat).reshape(num_bits, dim)

    def _get_node_centroid(self, node):
        """Extract centroid from a FlatBuffers node as numpy array."""
        if node.CentroidIsNone() or node.CentroidLength() == 0:
            return None
        return np.array(node.CentroidAsNumpy())

    def _hash_query(self, query, hyperplanes):
        """Compute LSH hash of query against hyperplanes.

        Args:
            query: (dim,) float32 array.
            hyperplanes: (num_bits, dim) float32 array.

        Returns:
            (bucket_id, projections) — bucket_id as int, projections as
            (num_bits,) float32 array of signed distances to each hyperplane.
        """
        projections = hyperplanes @ query
        bits = (projections > 0).astype(np.uint64)
        bucket_id = np.uint64(0)
        for i, b in enumerate(bits):
            bucket_id |= (b << np.uint64(i))
        return int(bucket_id), projections

    def _margin_distance(self, a, b, projections, num_bits):
        """Margin-weighted distance between two bucket IDs.

        Cost of flipping bit i = |projection[i]| (distance from hyperplane).
        Bits where the query is close to the boundary are cheap to flip;
        bits where the query is far from the boundary are expensive.
        """
        xor = a ^ b
        cost = 0.0
        for i in range(num_bits):
            if xor & (1 << i):
                cost += abs(float(projections[i]))
        return cost

    def search(self, query, k=10, nprobe=3, return_routing=False, backend="python"):
        """Search the index for nearest neighbors.

        Args:
            query: (dim,) query vector.
            k: Number of results to return.
            nprobe: Number of leaf probes. Accepts:
                - int: fixed probe count (1 = single path, >1 = multi-probe)
                - "auto": adaptive probing with default AdaptiveProbeConfig
                - AdaptiveProbeConfig: adaptive probing with custom thresholds
            return_routing: If True, populate result.routing with diagnostics.
            backend: "python" (default) or "rust". The rust backend uses a
                preloaded Rust kernel (dyf_rs.DyfSearcher) — much faster, same
                result shape. It falls back to the python path when the index is
                PQ-compressed, has overflow batches, or when nprobe is adaptive
                or return_routing is requested (see _rust_eligible).

        Returns:
            SearchResult with indices, scores, and fields. Supports
            backward-compatible unpacking: indices, scores = idx.search(...)
        """
        import time
        t0 = time.perf_counter()

        query = np.asarray(query, dtype=np.float32)
        if query.ndim != 1 or len(query) != self.embedding_dim:
            raise ValueError(
                f"Query must be 1D with dim={self.embedding_dim}, "
                f"got shape {query.shape}")

        if backend == "rust" and self._rust_eligible(nprobe, return_routing):
            return self._search_rust(query, k, nprobe)

        # Normalize query for cosine similarity
        qnorm = np.linalg.norm(query)
        if qnorm > 1e-10:
            query_normed = query / qnorm
        else:
            query_normed = query

        # Find candidate leaves by tree traversal
        candidate_leaves, min_margin = self._find_candidate_leaves(query, nprobe)

        # Precompute ADC tables once per query if PQ
        adc_tables = None
        if self.is_pq:
            adc_tables = self._pq_precompute_tables(query_normed)

        # Collect results from all candidate leaves
        all_indices = []
        all_scores = []
        all_fields = {}  # field_name -> list of arrays
        sf_names = self.stored_field_names
        candidates_scored = 0

        for batch_index in candidate_leaves:
            batch = self.get_leaf(batch_index)
            item_indices, scores, field_data = self._score_leaf(
                batch, query_normed, adc_tables)
            candidates_scored += len(item_indices)
            all_indices.append(item_indices)
            all_scores.append(scores)
            for fname in sf_names:
                all_fields.setdefault(fname, []).append(field_data[fname])

        if not all_indices:
            result = SearchResult(
                np.array([], dtype=np.uint32),
                np.array([], dtype=np.float32),
                {fname: [] for fname in sf_names})
            if return_routing:
                routing = {
                    'leaves_probed': list(candidate_leaves),
                    'candidates_scored': 0,
                    'min_margin': min_margin if min_margin != float('inf') else None,
                    'elapsed_ms': (time.perf_counter() - t0) * 1000,
                }
                if not isinstance(nprobe, int):
                    routing['nprobe_mode'] = 'adaptive'
                    routing['adaptive_nprobe'] = len(candidate_leaves)
                result.routing = routing
            return result

        all_indices, all_scores, merged_fields = _merge_leaf_results(
            all_indices, all_scores, all_fields, sf_names)
        top_indices, top_scores, result_fields = _topk_with_fields(
            all_indices, all_scores, merged_fields, sf_names, k)

        routing_info = None
        if return_routing:
            routing_info = {
                'leaves_probed': list(candidate_leaves),
                'candidates_scored': candidates_scored,
                'min_margin': min_margin if min_margin != float('inf') else None,
                'elapsed_ms': (time.perf_counter() - t0) * 1000,
            }
            if not isinstance(nprobe, int):
                routing_info['nprobe_mode'] = 'adaptive'
                routing_info['adaptive_nprobe'] = len(candidate_leaves)

        return SearchResult(top_indices, top_scores, result_fields, routing_info)

    # --- Rust backend (dyf_rs.DyfSearcher, preloaded dense path) ---------------

    def _rust_eligible(self, nprobe, return_routing):
        """Whether the rust fast path can serve this query. Falls back to python
        for PQ, adaptive nprobe, return_routing, or overflow batches."""
        if self.is_pq or return_routing or not isinstance(nprobe, int):
            return False
        if getattr(self, "_has_overflow", None) is None:
            import dyf_rs
            f = dyf_rs.DyfFile.open(self._path)
            nob = f.num_overflow_batches
            if callable(nob):
                nob = nob()
            self._has_overflow = nob > 0
        return not self._has_overflow

    def _rust_searcher(self):
        if getattr(self, "_rust_search_obj", None) is None:
            import dyf_rs
            # lazy=True: instant open, bounded memory (only touched leaves), and
            # faster warm queries (per-leaf contiguous scoring) — the right fit for
            # the on-disk LazyIndex. Cold leaves pay a one-time decode on first use.
            self._rust_search_obj = dyf_rs.DyfSearcher.open(self._path, lazy=True)
        return self._rust_search_obj

    def _search_rust(self, query, k, nprobe):
        s = self._rust_searcher()
        q2 = np.ascontiguousarray(query, dtype=np.float32).reshape(1, -1)
        idx, sc = s.search_batch(q2, int(k), int(nprobe))
        idx, sc = np.asarray(idx[0]), np.asarray(sc[0])
        mask = idx >= 0
        idx = idx[mask].astype(np.uint32)
        sc = sc[mask].astype(np.float32)
        fields = self._gather_fields(idx) if self.has_stored_fields else {}
        return SearchResult(idx, sc, fields)

    def _leaf_batch_indices(self):
        if getattr(self, "_leaf_batches", None) is None:
            bs = []
            for nid in range(self._index.NodesLength()):
                node = self._index.Nodes(nid)
                if node and node.ChildrenLength() == 0 and node.BatchIndex() >= 0:
                    bs.append(node.BatchIndex())
            self._leaf_batches = bs
        return self._leaf_batches

    def _gather_fields(self, item_indices):
        """Stored-field values for the final-k items (rust path). Builds a
        per-item field cache once (one pass over base batches, matching the
        preloaded corpus), then per-query lookup is O(k)."""
        sf_types = self._get_stored_field_types()
        if not sf_types:
            return {}
        if getattr(self, "_field_cache", None) is None:
            cache = {f: {} for f in sf_types}
            for bi in self._leaf_batch_indices():
                b = self.get_leaf(bi)
                iidx = b.column("item_index").to_numpy()
                for f, ftype in sf_types.items():
                    col = b.column(f)
                    vals = col.to_pylist() if ftype in ("utf8", "binary") else col.to_numpy()
                    d = cache[f]
                    for row, it in enumerate(iidx):
                        d[int(it)] = vals[row]
            self._field_cache = cache
        out = {}
        for f, d in self._field_cache.items():
            vals = [d.get(int(it)) for it in item_indices]
            out[f] = vals if sf_types[f] in ("utf8", "binary") else np.asarray(vals)
        return out

    def _build_centroid_index(self):
        """Build a flat centroid matrix from all leaf nodes.

        Scans FlatBuffers nodes once, extracts centroids from leaves,
        returns (centroid_matrix, batch_indices) for IVF-style search.
        Cached after first call.
        """
        if hasattr(self, '_centroid_matrix'):
            return self._centroid_matrix, self._centroid_batch_indices

        centroids = []
        batch_indices = []

        for node_id in range(self._index.NodesLength()):
            node = self._index.Nodes(node_id)
            if node is None:
                continue
            # Leaf = no children, has a valid batch index
            if node.ChildrenLength() == 0 and node.BatchIndex() >= 0:
                cent = self._get_node_centroid(node)
                if cent is not None:
                    centroids.append(cent)
                    batch_indices.append(node.BatchIndex())

        self._centroid_matrix = np.array(centroids, dtype=np.float32)  # (n_leaves, dim)
        self._centroid_batch_indices = np.array(batch_indices, dtype=np.int32)
        return self._centroid_matrix, self._centroid_batch_indices

    def search_ivf(self, query, k=10, nprobe=3):
        """IVF-style search: rank leaves by centroid similarity, then score.

        Instead of LSH tree traversal, computes query similarity to all leaf
        centroids and probes the top-nprobe leaves. Combines DYF's lazy
        loading with FAISS IVF-quality routing.

        Args:
            query: (dim,) query vector.
            k: Number of results to return.
            nprobe: Number of leaves to probe.

        Returns:
            SearchResult with indices, scores, and fields. Supports
            backward-compatible unpacking: indices, scores = idx.search_ivf(...)
        """
        query = np.asarray(query, dtype=np.float32)
        if query.ndim != 1 or len(query) != self.embedding_dim:
            raise ValueError(
                f"Query must be 1D with dim={self.embedding_dim}, "
                f"got shape {query.shape}")

        # Normalize query
        qnorm = np.linalg.norm(query)
        if qnorm > 1e-10:
            query_normed = query / qnorm
        else:
            query_normed = query

        # Get centroid index (built once, cached)
        centroids, batch_indices = self._build_centroid_index()

        # Rank all leaves by centroid similarity
        sims = centroids @ query_normed  # (n_leaves,)
        if len(sims) <= nprobe:
            top_leaf_pos = np.arange(len(sims))
        else:
            top_leaf_pos = np.argpartition(-sims, nprobe)[:nprobe]

        candidate_batches = batch_indices[top_leaf_pos]

        # Precompute ADC tables once per query if PQ
        adc_tables = None
        if self.is_pq:
            adc_tables = self._pq_precompute_tables(query_normed)

        # Score candidates from selected leaves
        all_indices = []
        all_scores = []
        all_fields = {}
        sf_names = self.stored_field_names

        for batch_index in candidate_batches:
            batch = self.get_leaf(int(batch_index))
            item_indices, scores, field_data = self._score_leaf(
                batch, query_normed, adc_tables)
            all_indices.append(item_indices)
            all_scores.append(scores)
            for fname in sf_names:
                all_fields.setdefault(fname, []).append(field_data[fname])

        if not all_indices:
            return SearchResult(
                np.array([], dtype=np.uint32),
                np.array([], dtype=np.float32),
                {fname: [] for fname in sf_names})

        all_indices, all_scores, merged_fields = _merge_leaf_results(
            all_indices, all_scores, all_fields, sf_names)
        top_indices, top_scores, result_fields = _topk_with_fields(
            all_indices, all_scores, merged_fields, sf_names, k)

        return SearchResult(top_indices, top_scores, result_fields)

    def _build_item_map(self):
        """Build mapping from item_index to (batch_index, row_index).

        Scans all leaf batches once, caches the mapping for O(1) subsequent
        lookups. Called lazily on first get_item_vector() or similar call.
        """
        if hasattr(self, '_item_map'):
            return self._item_map
        item_map = {}
        n_batches = self._index.BatchesLength()
        for bi in range(n_batches):
            batch = self.get_leaf(bi)
            batch_item_ids = batch.column('item_index').to_numpy()
            for row_idx, item_id in enumerate(batch_item_ids):
                item_map[int(item_id)] = (bi, row_idx)
        self._item_map = item_map
        return item_map

    def get_item_vector(self, item_index):
        """Extract a single item's embedding vector from the index.

        Scans leaf batches to find the item (building a cached mapping on
        first call), then returns the reconstructed float32 vector.

        Args:
            item_index: The item index (as stored in the item_index column).

        Returns:
            (dim,) float32 numpy array of the item's embedding.

        Raises:
            KeyError: If item_index is not found in the index.
        """
        item_map = self._build_item_map()
        if item_index not in item_map:
            raise KeyError(f"Item index {item_index} not found in index")

        batch_index, row_idx = item_map[item_index]
        batch = self.get_leaf(batch_index)
        emb_col = batch.column('embedding')
        flat_values = emb_col.values.to_numpy()

        if self.is_pq:
            meta = self._get_metadata()
            m = int(meta['pq_n_subquantizers'])
            n_rows = len(emb_col)
            codes = flat_values.reshape(n_rows, m)
            row_codes = codes[row_idx:row_idx + 1]  # (1, M)
            return self._pq_reconstruct(row_codes)[0]  # (dim,) float32
        else:
            dim = self.embedding_dim
            n_rows = len(emb_col)
            leaf_emb = flat_values.reshape(n_rows, dim)
            return leaf_emb[row_idx].astype(np.float32)

    def get_stored_fields(self, item_indices):
        """Look up stored fields for given item indices without re-searching.

        Scans all leaves to find matching items. Less efficient than getting
        fields from search results, but useful for post-hoc exploration.

        Args:
            item_indices: array-like of item indices to look up.

        Returns:
            dict mapping field name to list of values (in same order as
            item_indices). Missing items get None.
        """
        item_indices = np.asarray(item_indices, dtype=np.uint32)
        sf_names = self.stored_field_names
        if not sf_names:
            return {}

        # Build lookup: item_index -> position in output
        target_set = {int(idx): pos for pos, idx in enumerate(item_indices)}
        n_out = len(item_indices)
        result = {fname: [None] * n_out for fname in sf_names}

        # Scan all leaves
        n_batches = self._index.BatchesLength()
        for bi in range(n_batches):
            batch = self.get_leaf(bi)
            batch_item_ids = batch.column('item_index').to_numpy()

            # Check which target items are in this batch
            for row_idx, item_id in enumerate(batch_item_ids):
                item_id_int = int(item_id)
                if item_id_int in target_set:
                    out_pos = target_set[item_id_int]
                    for fname in sf_names:
                        col = batch.column(fname)
                        result[fname][out_pos] = col[row_idx].as_py()

        return result

    def extract_all_fields(self) -> ExtractedData:
        """Bulk-read all leaf batches, concatenate, sort by item_index.

        Returns:
            dict with keys:
                'embeddings': (n, d) float32 array, or None if the file
                    was written without embeddings (PQ indexes return
                    approximate reconstructions)
                'fields': {field_name: array} for all stored fields
                'metadata': dict of metadata key-value pairs
        """
        n = self.total_items
        dim = self.embedding_dim
        meta = self._get_metadata()
        has_embeddings = meta.get('has_embeddings') != 'false'

        embeddings = np.zeros((n, dim), dtype=np.float32) if has_embeddings else None
        sf_names = self.stored_field_names
        sf_types = self._get_stored_field_types()

        # Initialize stored field arrays
        fields = {}
        for fname in sf_names:
            tname = sf_types[fname]
            if tname in ('utf8', 'binary'):
                fields[fname] = [None] * n
            elif tname == 'float32':
                fields[fname] = np.zeros(n, dtype=np.float32)
            elif tname == 'float64':
                fields[fname] = np.zeros(n, dtype=np.float64)
            elif tname == 'int32':
                fields[fname] = np.zeros(n, dtype=np.int32)
            elif tname == 'int64':
                fields[fname] = np.zeros(n, dtype=np.int64)
            else:
                fields[fname] = [None] * n

        n_batches = self._index.BatchesLength()
        for bi in range(n_batches):
            batch = self.get_leaf(bi)
            item_indices = batch.column('item_index').to_numpy()

            # Extract embeddings (if present)
            if has_embeddings:
                emb_col = batch.column('embedding')
                n_rows = len(emb_col)
                flat_values = emb_col.values.to_numpy()

                if self.is_pq:
                    m = int(meta['pq_n_subquantizers'])
                    codes = flat_values.reshape(n_rows, m)
                    leaf_emb = self._pq_reconstruct(codes)
                else:
                    leaf_emb = flat_values.reshape(n_rows, dim).astype(
                        np.float32)

                embeddings[item_indices] = leaf_emb

            # Extract stored fields
            for fname in sf_names:
                col = batch.column(fname)
                tname = sf_types[fname]
                if tname in ('utf8', 'binary'):
                    values = col.to_pylist()
                    for i, item_idx in enumerate(item_indices):
                        fields[fname][int(item_idx)] = values[i]
                else:
                    values = col.to_numpy()
                    fields[fname][item_indices] = values

        return {
            'embeddings': embeddings,
            'fields': fields,
            'metadata': dict(self._get_metadata()),
        }

    def detect_enrichment_level(self) -> int:
        """Detect enrichment level based on stored fields and metadata.

        Returns 0-3:
            0: Base (embeddings + tree only)
            1: Projected (has umap_x, umap_y, umap_z stored fields)
            2: Clustered (has community_id or cluster_* stored fields)
            3: Viz-ready (has edge_pairs or tour_narration in metadata)
        """
        sf = set(self.stored_field_names)
        meta = self._get_metadata()

        has_umap = {'umap_x', 'umap_y', 'umap_z'}.issubset(sf)
        has_clusters = (
            any(f.startswith('cluster_') for f in sf)
            or 'community_id' in sf
            or 'lsh_bucket_ids' in sf
            or 'louvain_dendrogram' in meta
        )
        has_viz = 'edge_pairs' in meta or 'tour_narration' in meta

        if has_viz:
            return 3
        if has_clusters:
            return 2
        if has_umap:
            return 1
        return 0

    def _resolve_nprobe(self, nprobe, min_margin):
        """Resolve effective probe count from nprobe spec and routing margin.

        Args:
            nprobe: int (pass-through), "auto" (default config), or
                AdaptiveProbeConfig instance.
            min_margin: Minimum absolute projection margin along primary path.

        Returns:
            int — effective number of probes.
        """
        if isinstance(nprobe, int):
            return nprobe

        if nprobe == "auto":
            cfg = AdaptiveProbeConfig()
        elif isinstance(nprobe, AdaptiveProbeConfig):
            cfg = nprobe
        else:
            raise ValueError(
                f"nprobe must be int, 'auto', or AdaptiveProbeConfig, got {nprobe!r}")

        if min_margin <= cfg.margin_lo:
            return cfg.max_probes
        if min_margin >= cfg.margin_hi:
            return cfg.min_probes

        # Linear interpolation: low margin → max probes, high margin → min probes
        t = (min_margin - cfg.margin_lo) / (cfg.margin_hi - cfg.margin_lo)
        return round(cfg.max_probes + t * (cfg.min_probes - cfg.max_probes))

    def _find_candidate_leaves(self, query, nprobe):
        """Traverse tree to find candidate leaf batch indices.

        For nprobe=1: follows the primary LSH path.
        For nprobe>1: also probes siblings with nearest Hamming-distance buckets.
        For nprobe="auto" or AdaptiveProbeConfig: two-phase traversal that
        determines effective nprobe from the primary path's routing margin.

        Returns:
            (candidates, min_margin) — list of batch indices and the minimum
            absolute projection margin observed on the primary routing path.
        """
        is_adaptive = not isinstance(nprobe, int)
        # For adaptive mode, always enqueue alternatives during phase 1
        effective_nprobe = nprobe if isinstance(nprobe, int) else float('inf')

        root_id = self._index.Root()
        candidates = set()
        min_margin = float('inf')

        # Use a priority queue of (priority, node_id) to manage probes
        # Priority 0 = primary path, 1+ = alternative paths
        probe_queue = [(0, root_id)]
        visited_leaves = set()
        first_leaf_found = False

        while probe_queue and len(candidates) < effective_nprobe:
            # Pop lowest priority (greedy)
            probe_queue.sort(key=lambda x: x[0])
            priority, node_id = probe_queue.pop(0)

            node = self._index.Nodes(node_id)
            if node is None:
                continue

            # If leaf, add to candidates
            if node.ChildrenLength() == 0:
                bi = node.BatchIndex()
                if bi >= 0 and bi not in visited_leaves:
                    candidates.add(bi)
                    visited_leaves.add(bi)

                    # Phase transition: after first leaf, resolve adaptive nprobe
                    if is_adaptive and not first_leaf_found:
                        first_leaf_found = True
                        effective_nprobe = self._resolve_nprobe(
                            nprobe, min_margin)
                continue

            # Internal node: compute hash and route
            hyperplanes = self._get_node_hyperplanes(node)
            n_children = node.ChildrenLength()

            if hyperplanes is not None:
                bucket_id, projections = self._hash_query(query, hyperplanes)
                num_bits = node.NumBits()

                # Track min margin on primary path
                if priority == 0:
                    node_min_proj = float(np.min(np.abs(projections)))
                    min_margin = min(min_margin, node_min_proj)

                # Find primary child via bucket_id_to_child mapping
                primary_child = None
                primary_bid = None

                # Build bucket_id -> child_node_id mapping
                bid_to_child = {}
                for ci in range(n_children):
                    child_node_id = node.Children(ci)
                    if not node.BucketIdsToChildrenIsNone() and ci < node.BucketIdsToChildrenLength():
                        bid = node.BucketIdsToChildren(ci)
                        bid_to_child[bid] = child_node_id

                if bucket_id in bid_to_child:
                    primary_child = bid_to_child[bucket_id]
                    primary_bid = bucket_id
                else:
                    # Fallback: find nearest bucket by margin distance
                    best_dist = float('inf')
                    for bid, child_nid in bid_to_child.items():
                        dist = self._margin_distance(bucket_id, bid, projections, num_bits)
                        if dist < best_dist:
                            best_dist = dist
                            primary_child = child_nid
                            primary_bid = bid

                if primary_child is not None:
                    probe_queue.append((priority, primary_child))

                # Enqueue alternatives when multi-probe or adaptive
                if effective_nprobe > 1 or is_adaptive:
                    alternatives = []
                    for bid, child_nid in bid_to_child.items():
                        if bid == primary_bid:
                            continue
                        dist = self._margin_distance(bucket_id, bid, projections, num_bits)
                        alternatives.append((dist, child_nid))
                    alternatives.sort(key=lambda x: x[0])
                    for dist, child_nid in alternatives:
                        probe_queue.append((priority + dist, child_nid))
            else:
                # No hyperplanes: fall back to nearest centroid
                best_sim = -2.0
                best_child = None
                for ci in range(n_children):
                    child_nid = node.Children(ci)
                    child_node = self._index.Nodes(child_nid)
                    if child_node is not None:
                        child_cent = self._get_node_centroid(child_node)
                        if child_cent is not None:
                            sim = float(query @ child_cent)
                            if sim > best_sim:
                                best_sim = sim
                                best_child = child_nid
                if best_child is not None:
                    probe_queue.append((priority, best_child))

                # For multi-probe or adaptive: also add other children
                if effective_nprobe > 1 or is_adaptive:
                    for ci in range(n_children):
                        child_nid = node.Children(ci)
                        if child_nid != best_child:
                            probe_queue.append((priority + 1, child_nid))

        return list(candidates), min_margin

    def to_faiss(self, pq_subquantizers=None, pq_bits=8):
        """Export dyf index as a FAISS IVF index.

        Uses dyf's leaf centroids as the coarse quantizer and populates
        FAISS inverted lists from dyf's leaf embeddings. This gives dyf's
        PCA-LSH partitioning with FAISS's optimized search.

        Args:
            pq_subquantizers: If set, use IndexIVFPQ with this many
                subquantizers for compression (e.g., 8 or 16). If None,
                use IndexIVFFlat (no compression).
            pq_bits: Bits per subquantizer for PQ (default: 8).

        Returns:
            faiss.IndexIVFFlat or faiss.IndexIVFPQ, ready to search.
        """
        import faiss

        dim = self.embedding_dim
        centroids, batch_indices = self._build_centroid_index()
        n_leaves = len(centroids)

        # Normalize centroids for inner product (cosine) search
        centroids_norm = centroids.copy()
        faiss.normalize_L2(centroids_norm)

        # Build coarse quantizer from dyf leaf centroids
        quantizer = faiss.IndexFlatIP(dim)
        quantizer.add(centroids_norm)

        # Create IVF index
        if pq_subquantizers is not None:
            index = faiss.IndexIVFPQ(
                quantizer, dim, n_leaves, pq_subquantizers, pq_bits,
                faiss.METRIC_INNER_PRODUCT)
        else:
            index = faiss.IndexIVFFlat(
                quantizer, dim, n_leaves, faiss.METRIC_INNER_PRODUCT)

        # Collect all embeddings and IDs from leaves
        all_embeddings = []
        all_ids = []

        if self.is_pq:
            import warnings
            warnings.warn(
                "Exporting PQ-quantized index to FAISS uses lossy "
                "reconstructed vectors. Search quality may be reduced.",
                stacklevel=2)
            meta = self._get_metadata()
            m = int(meta['pq_n_subquantizers'])
            for batch_idx in batch_indices:
                batch = self.get_leaf(int(batch_idx))
                item_indices = batch.column('item_index').to_numpy().astype(np.int64)
                emb_col = batch.column('embedding')
                flat_values = emb_col.values.to_numpy()
                codes = flat_values.reshape(len(item_indices), m)
                leaf_emb = self._pq_reconstruct(codes)
                all_embeddings.append(leaf_emb)
                all_ids.append(item_indices)
        else:
            for batch_idx in batch_indices:
                batch = self.get_leaf(int(batch_idx))
                item_indices = batch.column('item_index').to_numpy().astype(np.int64)
                emb_col = batch.column('embedding')
                flat_values = emb_col.values.to_numpy()
                leaf_emb = flat_values.reshape(len(item_indices), dim).astype(np.float32)
                all_embeddings.append(leaf_emb)
                all_ids.append(item_indices)

        all_embeddings = np.vstack(all_embeddings)
        all_ids = np.concatenate(all_ids)

        # Normalize for cosine similarity
        faiss.normalize_L2(all_embeddings)

        # Train PQ codebook if needed (IVFFlat doesn't need training)
        if pq_subquantizers is not None:
            index.train(all_embeddings)
        else:
            index.is_trained = True

        # Add vectors — FAISS assigns to nearest centroid
        index.add_with_ids(all_embeddings, all_ids)

        return index


def from_faiss(
    faiss_index,
    path: str,
    compression: str = 'zstd',
    quantization: str = 'float16',
    metadata: dict[str, str] | None = None,
    stored_fields: Mapping[str, StoredFieldInput] | None = None,
) -> LazyIndex:
    """Build a dyf lazy index file from a FAISS IVF index.

    Extracts FAISS's inverted lists and centroids, builds a single-level
    dyf tree (one node per IVF cell), and writes to the dyf file format.

    Args:
        faiss_index: A trained faiss.IndexIVFFlat or faiss.IndexIVFPQ.
        path: Output file path (e.g., "index.dyf").
        compression: "none", "zstd", or "lz4" (default: "none").
        quantization: "float32", "float16", or "int8" (default: "float16").
        metadata: Optional dict of string key-value pairs.
        stored_fields: Optional dict mapping field name to array-like of
            length n_items (indexed by FAISS ID).

    Returns:
        LazyIndex opened on the written file.
    """
    import faiss

    # Extract index parameters
    dim = faiss_index.d
    n_cells = faiss_index.nlist
    invlists = faiss_index.invlists

    # Extract coarse centroids
    quantizer = faiss.downcast_index(faiss_index.quantizer)
    centroids = faiss.rev_swig_ptr(quantizer.get_xb(), n_cells * dim)
    centroids = np.array(centroids).reshape(n_cells, dim).copy()

    # Collect all embeddings, IDs, and cell assignments
    # Enable direct map for reconstruction (needed for PQ indexes)
    faiss_index.make_direct_map()

    all_embeddings = []
    all_ids = []
    cell_sizes = []

    for cell_id in range(n_cells):
        list_size = invlists.list_size(cell_id)
        if list_size == 0:
            cell_sizes.append(0)
            continue

        ids = faiss.rev_swig_ptr(invlists.get_ids(cell_id), list_size)
        ids = np.array(ids).copy().astype(np.int64)

        # Reconstruct vectors (works for both Flat and PQ inverted lists)
        codes = np.zeros((list_size, dim), dtype=np.float32)
        for j in range(list_size):
            faiss_index.reconstruct(int(ids[j]), codes[j])

        all_embeddings.append(codes)
        all_ids.append(ids)
        cell_sizes.append(list_size)

    if not all_embeddings:
        raise ValueError("FAISS index is empty (no vectors)")
    all_embeddings = np.vstack(all_embeddings)
    all_ids = np.concatenate(all_ids)

    # Reorder embeddings so position i holds FAISS ID i.
    # This way write_lazy_index stores original FAISS IDs as item_index.
    max_id = int(all_ids.max())
    n_total = max_id + 1
    ordered_embeddings = np.zeros((n_total, dim), dtype=np.float32)
    ordered_embeddings[all_ids] = all_embeddings

    # Build a flat dyf tree: root with one child per non-empty cell
    non_empty_cells = [i for i in range(n_cells) if cell_sizes[i] > 0]

    # Rebuild cell membership using original FAISS IDs
    children = []
    cell_id_offset = 0
    for cell_id in non_empty_cells:
        list_size = cell_sizes[cell_id]
        # all_ids was built in cell order, so slice to get this cell's IDs
        cell_faiss_ids = all_ids[cell_id_offset:cell_id_offset + list_size]
        cell_id_offset += list_size
        children.append({
            'children': [],
            'indices': cell_faiss_ids.astype(np.intp),
            'depth': 0,
            'point_margin_map': None,
            'hyperplanes': None,
            'bucket_id_to_child': None,
        })

    tree = {
        'children': children,
        'indices': np.arange(n_total),
        'depth': 1,
        'point_margin_map': None,
        'hyperplanes': None,
        'bucket_id_to_child': None,
    }

    build_params = {
        'max_depth': 1,
        'num_bits': 0,
        'min_leaf_size': 1,
        'seed': 0,
    }

    if metadata is None:
        metadata = {}
    metadata['source'] = 'faiss'
    metadata['faiss_nlist'] = str(n_cells)

    write_lazy_index(tree, ordered_embeddings, path,
                     compression=compression,
                     quantization=quantization,
                     metadata=metadata,
                     build_params=build_params,
                     stored_fields=stored_fields)

    return LazyIndex(path)


def _reconstruct_tree(idx):
    """Reconstruct tree dict from LazyIndex FlatBuffers + Arrow batches.

    Rebuilds the recursive tree structure that write_lazy_index expects,
    including hyperplanes, bucket_id_to_child mappings, and per-leaf
    item indices from the Arrow batches.

    Args:
        idx: An open LazyIndex instance.

    Returns:
        dict compatible with write_lazy_index's tree parameter.
    """
    n_nodes = idx._index.NodesLength()
    dim = idx.embedding_dim

    # Parse all FlatBuffers nodes into flat dicts
    node_data = []
    for nid in range(n_nodes):
        fb_node = idx._index.Nodes(nid)
        children_ids = [fb_node.Children(i)
                        for i in range(fb_node.ChildrenLength())]

        # Hyperplanes
        hp = None
        num_bits = fb_node.NumBits()
        if (not fb_node.HyperplanesIsNone()
                and fb_node.HyperplanesLength() > 0
                and num_bits > 0):
            hp_flat = fb_node.HyperplanesAsNumpy()
            hp = np.array(hp_flat, dtype=np.float32).reshape(num_bits, dim)

        # Bucket IDs to children mapping
        # In write_lazy_index: bucket_id_to_child = {bucket_id: child_index}
        # In FlatBuffers: BucketIdsToChildren[ci] = bucket_id for child ci
        bid_to_child = None
        if (not fb_node.BucketIdsToChildrenIsNone()
                and fb_node.BucketIdsToChildrenLength() > 0):
            bid_to_child = {}
            for ci in range(fb_node.BucketIdsToChildrenLength()):
                bid = int(fb_node.BucketIdsToChildren(ci))
                bid_to_child[bid] = ci

        # Eigenvalues
        ev = None
        if not fb_node.EigenvaluesIsNone() and fb_node.EigenvaluesLength() > 0:
            ev = np.array(fb_node.EigenvaluesAsNumpy(), dtype=np.float32)

        # Centroid
        centroid = None
        if not fb_node.CentroidIsNone() and fb_node.CentroidLength() > 0:
            centroid = np.array(fb_node.CentroidAsNumpy(), dtype=np.float32)

        is_leaf = len(children_ids) == 0
        batch_index = fb_node.BatchIndex()

        # For leaves, extract item indices from Arrow batch
        indices = None
        if is_leaf and batch_index >= 0:
            batch = idx.get_leaf(batch_index)
            indices = batch.column('item_index').to_numpy().astype(np.intp)

        node_data.append({
            'children_ids': children_ids,
            'hyperplanes': hp,
            'bucket_id_to_child': bid_to_child,
            'eigenvalues': ev,
            'centroid': centroid,
            'depth': int(fb_node.Depth()),
            'is_leaf': is_leaf,
            'indices': indices,
        })

    # Build recursive tree structure (bottom-up index aggregation)
    def _build(nid):
        nd = node_data[nid]
        children = []
        all_indices = []

        for child_id in nd['children_ids']:
            child = _build(child_id)
            children.append(child)
            all_indices.append(child['indices'])

        if nd['is_leaf']:
            indices = nd['indices']
        else:
            indices = (np.concatenate(all_indices) if all_indices
                       else np.array([], dtype=np.intp))

        return {
            'children': children,
            'indices': indices,
            'depth': nd['depth'],
            'hyperplanes': nd['hyperplanes'],
            'bucket_id_to_child': nd['bucket_id_to_child'],
            'eigenvalues': nd['eigenvalues'],
            'centroid': nd['centroid'],
        }

    root_id = idx._index.Root()
    return _build(root_id)


def rewrite_lazy_index(
    path: str,
    new_stored_fields: Mapping[str, StoredFieldInput] | None = None,
    new_metadata: Mapping[str, str | None] | None = None,
    output_path: str | None = None,
    compression: str | None = None,
    drop_fields: Set[str] | list[str] | None = None,
    drop_embeddings: bool = False,
    format_version: int | None = None,
) -> None:
    """Rewrite a .dyf file with additional stored fields and/or metadata.

    Preserves the tree structure, embeddings, and existing stored fields.
    Adds new per-point stored_fields columns and/or metadata key-value pairs.

    Args:
        path: Path to existing .dyf file.
        new_stored_fields: Optional dict mapping field name to array-like
            of length total_items. Values are indexed by item_index.
        new_metadata: Optional dict of string key-value pairs to add/update.
            Values of None delete the key from existing metadata.
        output_path: Output file path. If None, overwrites the input file.
        compression: Override compression codec ('none', 'zstd', 'lz4').
            If None, preserves the original file's compression.
        drop_fields: Optional set/list of stored field names to remove.
            Applied after merging existing and new fields (exact names only).
        drop_embeddings: If True, write a viz-only file without embedding
            vectors in Arrow batches. Tree centroids are preserved.
        format_version: Output format version (1, 2, or 3). If None,
            preserves the source file's format version.

    Raises:
        ValueError: If the index uses PQ quantization (lossy round-trip).
    """
    out_path = output_path or path
    format_ver = detect_dyf_version(path)
    if format_version is not None:
        format_ver = format_version

    # Read everything while file is open
    with LazyIndex(path) as idx:
        if idx.is_pq:
            raise ValueError(
                "rewrite_lazy_index does not support PQ-quantized indexes "
                "(round-trip through float32 is lossy)")

        tree = _reconstruct_tree(idx)
        data = idx.extract_all_fields()
        src_embedding_dim = idx.embedding_dim

        bp = idx._index.BuildParams()
        build_params = {
            'max_depth': bp.MaxDepth() if bp else 8,
            'num_bits': bp.NumBits() if bp else 3,
            'min_leaf_size': bp.MinLeafSize() if bp else 4,
            'seed': bp.Seed() if bp else 42,
        }
        quantization = idx.quantization
        src_compression = (bp.Compression().decode()
                           if bp and bp.Compression() else 'none')
        if compression is None:
            compression = src_compression

    # Merge stored fields (existing + new)
    merged_sf: dict[str, StoredFieldInput] = dict(data['fields'])
    if new_stored_fields:
        merged_sf.update(new_stored_fields)
    if drop_fields:
        for fname in drop_fields:
            merged_sf.pop(fname, None)

    # Merge metadata (existing + new; None values delete keys)
    merged_meta = dict(data['metadata'])
    # Remove keys that write_lazy_index will regenerate
    merged_meta.pop('stored_fields', None)
    if new_metadata:
        for k, v in new_metadata.items():
            if v is None:
                merged_meta.pop(k, None)
            else:
                merged_meta[k] = v

    # For v2 files with only new stored fields (no drops, no output redirect),
    # use efficient append via Rust DyfFile instead of full rewrite.
    if (
        format_ver == 2
        and new_stored_fields
        and not drop_fields
        and not new_metadata
        and output_path is None
    ):
        try:
            from dyf_rs import DyfFile as RustDyfFile  # noqa: F401
            _append_fields_v2(path, new_stored_fields)
            return
        except ImportError:
            pass  # Fall through to full rewrite

    out_embeddings = None if drop_embeddings else data['embeddings']
    write_lazy_index(
        tree, out_embeddings, out_path,
        compression=compression,
        quantization=quantization,
        metadata=merged_meta if merged_meta else None,
        build_params=build_params,
        stored_fields=merged_sf if merged_sf else None,
        format_version=format_ver,
        embedding_dim=src_embedding_dim,
    )


def _append_fields_v2(
    path: str,
    new_stored_fields: Mapping[str, StoredFieldInput],
) -> None:
    """Efficiently append stored fields to a DYF2 file using Rust backend.

    Uses DyfFile.append_field_layer() which only rewrites the footer,
    avoiding the O(all_data) cost of full rewrite.
    """
    import pyarrow as pa
    from dyf_rs import DyfFile as RustDyfFile

    dyf = RustDyfFile.open(path)

    for field_name, values in new_stored_fields.items():
        arr = np.asarray(values)

        # Determine Arrow type from numpy dtype
        if arr.dtype == np.int32:
            arrow_type = "int32"
        elif arr.dtype == np.int64:
            arrow_type = "int64"
        elif arr.dtype == np.float32:
            arrow_type = "float32"
        elif arr.dtype == np.float64:
            arrow_type = "float64"
        elif arr.dtype.kind in ('U', 'O'):
            arrow_type = "utf8"
        else:
            arrow_type = str(arr.dtype)

        # Serialize as Arrow IPC batch
        if arrow_type == "utf8":
            pa_arr = pa.array([str(v) if v is not None else None for v in values])
        else:
            pa_arr = pa.array(arr)

        batch = pa.record_batch([pa_arr], names=[field_name])
        sink = pa.BufferOutputStream()
        writer = pa.ipc.new_stream(sink, batch.schema)
        writer.write_batch(batch)
        writer.close()
        batch_bytes = sink.getvalue().to_pybytes()

        dyf.append_field_layer(field_name, field_name, arrow_type, [batch_bytes])
