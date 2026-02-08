"""DYF Lazy Index: FlatBuffers + Arrow IPC for mmap-based query serving.

File format:
    [16-byte header: magic "DYF1" + fb_size (u64) + reserved (u32)]
    [FlatBuffers section (tree structure)]
    [padding to 4KB boundary]
    [Arrow IPC batch 0]
    [Arrow IPC batch 1]
    ...

Writer:
    write_lazy_index(tree, embeddings, path, ...)

Reader:
    LazyIndex(path) — mmap, zero startup cost, LRU-cached leaf access
"""

import mmap
import struct
from collections import deque

import numpy as np

MAGIC = b"DYF1"
HEADER_SIZE = 16  # 4 magic + 8 fb_size + 4 reserved
PAGE_SIZE = 4096


def _flatten_tree_bfs(tree, embeddings):
    """Flatten tree to a BFS-ordered list of node dicts with computed fields.

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
        is_leaf = not node['children']

        # Compute centroid from embeddings
        indices = node['indices']
        if len(indices) > 0:
            centroid = embeddings[indices].mean(axis=0).astype(np.float32)
            norm = np.linalg.norm(centroid)
            if norm > 1e-10:
                centroid /= norm
        else:
            centroid = np.zeros(embeddings.shape[1], dtype=np.float32)

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
        if is_leaf:
            bi = batch_idx
            leaf_batch_map[node_id] = batch_idx
            batch_idx += 1
        else:
            bi = -1

        result.append({
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
        })

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


def write_lazy_index(tree, embeddings, path, compression='zstd',
                     quantization='float16', metadata=None,
                     build_params=None):
    """Write a DYF lazy index file (FlatBuffers tree + Arrow IPC leaf data).

    Args:
        tree: Tree dict from build_dyf_tree() (must include hyperplanes/bucket
            mapping from updated _build_dyf_tree).
        embeddings: (n, d) array of embedding vectors.
        path: Output file path (e.g. "index.dyf").
        compression: "none", "zstd", or "lz4" (default: "zstd").
        quantization: "float32", "float16", "int8", or "pq-M" where M is the
            number of sub-quantizers (default: "float16"). For PQ, dim must
            be divisible by M. Example: "pq-8" for 8 sub-quantizers.
        metadata: Optional dict of string key-value pairs.
        build_params: Optional dict with keys: max_depth, num_bits,
            min_leaf_size, seed. Auto-detected from tree if not provided.
    """
    import flatbuffers
    import pyarrow as pa
    from dyf.schema import (
        Index as FBIndex,
        Node as FBNode,
        BatchDescriptor as FBBatch,
        BuildParams as FBBuildParams,
        KeyValue as FBKeyValue,
    )

    embeddings = np.asarray(embeddings, dtype=np.float32)
    n_items, embedding_dim = embeddings.shape

    # Flatten tree to BFS node list
    flat_nodes, leaf_batch_map = _flatten_tree_bfs(tree, embeddings)
    n_leaves = sum(1 for n in flat_nodes if n['is_leaf'])

    # Detect PQ quantization
    is_pq = quantization.startswith('pq')

    if is_pq:
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
    else:
        # Standard quantization path
        q_embeddings = _quantize_embeddings(embeddings, quantization)
        arrow_value_type = _numpy_dtype_to_arrow(quantization)
        arrow_schema = pa.schema([
            ('item_index', pa.uint32()),
            ('embedding', pa.list_(arrow_value_type, embedding_dim)),
        ])

    # Build Arrow IPC batches per leaf
    ipc_options = None
    if compression == 'zstd':
        ipc_options = pa.ipc.IpcWriteOptions(
            compression=pa.Codec('zstd'))
    elif compression == 'lz4':
        ipc_options = pa.ipc.IpcWriteOptions(
            compression=pa.Codec('lz4'))

    # Write each leaf batch to bytes, record offsets
    batch_buffers = []
    for node in flat_nodes:
        if not node['is_leaf']:
            continue
        indices = node['indices']

        # Build Arrow RecordBatch
        item_idx_arr = pa.array(indices.astype(np.uint32), type=pa.uint32())

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

        batch = pa.record_batch([item_idx_arr, emb_arr], schema=arrow_schema)

        # Write to IPC stream bytes
        sink = pa.BufferOutputStream()
        writer = pa.ipc.new_stream(sink, arrow_schema,
                                   options=ipc_options)
        writer.write_batch(batch)
        writer.close()
        batch_buffers.append(sink.getvalue())

    # Build FlatBuffer
    builder = flatbuffers.Builder(4096)

    # Pre-build all strings and vectors
    version_off = builder.CreateString("1.0")
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
    fb_bytes = bytes(builder.Output())

    # Write file
    fb_size = len(fb_bytes)
    # Compute padding to 4KB boundary
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

        # Parse header
        magic = self._mm[:4]
        if magic != MAGIC:
            raise ValueError(f"Invalid magic: {magic!r}, expected {MAGIC!r}")

        self._fb_size = struct.unpack_from('<Q', self._mm, 4)[0]
        # reserved = struct.unpack_from('<I', self._mm, 12)[0]

        # Parse FlatBuffers
        fb_start = HEADER_SIZE
        fb_end = fb_start + self._fb_size
        fb_buf = self._mm[fb_start:fb_end]

        from dyf.schema.Index import Index as FBIndex
        self._index = FBIndex.GetRootAs(fb_buf, 0)

        # Compute arrow section start
        total_header_fb = HEADER_SIZE + self._fb_size
        self._arrow_start = total_header_fb + (
            (PAGE_SIZE - (total_header_fb % PAGE_SIZE)) % PAGE_SIZE)

        # Cache for decompressed batches
        self._batch_cache = {}
        self._cache_order = deque()
        self._cache_maxsize = 64

    def close(self):
        """Release mmap and file handle."""
        if self._mm is not None:
            self._mm.close()
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
        return summary

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
            batch: Arrow RecordBatch with columns: item_index, embedding.
            query_normed: (dim,) L2-normalized query vector.
            adc_tables: (M, 256) precomputed ADC tables (required if is_pq).

        Returns:
            (item_indices, scores) — numpy arrays.
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

        return item_indices, scores

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

    def search(self, query, k=10, nprobe=3):
        """Search the index for nearest neighbors.

        Args:
            query: (dim,) query vector.
            k: Number of results to return.
            nprobe: Number of leaf probes (1 = single path, >1 = multi-probe).

        Returns:
            (indices, scores) — arrays of shape (min(k, total_found),).
            indices: item indices. scores: cosine similarities.
        """
        query = np.asarray(query, dtype=np.float32)
        if query.ndim != 1 or len(query) != self.embedding_dim:
            raise ValueError(
                f"Query must be 1D with dim={self.embedding_dim}, "
                f"got shape {query.shape}")

        # Normalize query for cosine similarity
        qnorm = np.linalg.norm(query)
        if qnorm > 1e-10:
            query_normed = query / qnorm
        else:
            query_normed = query

        # Find candidate leaves by tree traversal
        candidate_leaves = self._find_candidate_leaves(query, nprobe)

        # Precompute ADC tables once per query if PQ
        adc_tables = None
        if self.is_pq:
            adc_tables = self._pq_precompute_tables(query_normed)

        # Collect results from all candidate leaves
        all_indices = []
        all_scores = []

        for batch_index in candidate_leaves:
            batch = self.get_leaf(batch_index)
            item_indices, scores = self._score_leaf(
                batch, query_normed, adc_tables)
            all_indices.append(item_indices)
            all_scores.append(scores)

        if not all_indices:
            return np.array([], dtype=np.uint32), np.array([], dtype=np.float32)

        all_indices = np.concatenate(all_indices)
        all_scores = np.concatenate(all_scores)

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

        # Top-k
        if len(all_scores) <= k:
            order = np.argsort(-all_scores)
        else:
            # Partial sort for efficiency
            part_idx = np.argpartition(-all_scores, k)[:k]
            order = part_idx[np.argsort(-all_scores[part_idx])]

        return all_indices[order], all_scores[order]

    def _build_centroid_index(self):
        """Build a flat centroid matrix from all leaf nodes.

        Scans FlatBuffers nodes once, extracts centroids from leaves,
        returns (centroid_matrix, batch_indices) for IVF-style search.
        Cached after first call.
        """
        if hasattr(self, '_centroid_matrix'):
            return self._centroid_matrix, self._centroid_batch_indices

        dim = self.embedding_dim
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
            (indices, scores) — arrays of shape (min(k, total_found),).
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

        for batch_index in candidate_batches:
            batch = self.get_leaf(int(batch_index))
            item_indices, scores = self._score_leaf(
                batch, query_normed, adc_tables)
            all_indices.append(item_indices)
            all_scores.append(scores)

        if not all_indices:
            return np.array([], dtype=np.uint32), np.array([], dtype=np.float32)

        all_indices = np.concatenate(all_indices)
        all_scores = np.concatenate(all_scores)

        # Deduplicate
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

        # Top-k
        if len(all_scores) <= k:
            order = np.argsort(-all_scores)
        else:
            part_idx = np.argpartition(-all_scores, k)[:k]
            order = part_idx[np.argsort(-all_scores[part_idx])]

        return all_indices[order], all_scores[order]

    def _find_candidate_leaves(self, query, nprobe):
        """Traverse tree to find candidate leaf batch indices.

        For nprobe=1: follows the primary LSH path.
        For nprobe>1: also probes siblings with nearest Hamming-distance buckets.
        """
        root_id = self._index.Root()
        candidates = set()

        # Use a priority queue of (priority, node_id) to manage probes
        # Priority 0 = primary path, 1+ = alternative paths
        probe_queue = [(0, root_id)]
        visited_leaves = set()

        while probe_queue and len(candidates) < nprobe:
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
                continue

            # Internal node: compute hash and route
            hyperplanes = self._get_node_hyperplanes(node)
            n_children = node.ChildrenLength()

            if hyperplanes is not None:
                bucket_id, projections = self._hash_query(query, hyperplanes)
                num_bits = node.NumBits()

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

                # For nprobe > 1: add alternative children by margin distance
                if nprobe > 1:
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

                # For nprobe > 1: also add other children
                if nprobe > 1:
                    for ci in range(n_children):
                        child_nid = node.Children(ci)
                        if child_nid != best_child:
                            probe_queue.append((priority + 1, child_nid))

        return list(candidates)

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


def from_faiss(faiss_index, path, compression='zstd', quantization='float16',
               metadata=None):
    """Build a dyf lazy index file from a FAISS IVF index.

    Extracts FAISS's inverted lists and centroids, builds a single-level
    dyf tree (one node per IVF cell), and writes to the dyf file format.

    Args:
        faiss_index: A trained faiss.IndexIVFFlat or faiss.IndexIVFPQ.
        path: Output file path (e.g., "index.dyf").
        compression: "none", "zstd", or "lz4" (default: "zstd").
        quantization: "float32", "float16", or "int8" (default: "float16").
        metadata: Optional dict of string key-value pairs.

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
                     build_params=build_params)

    return LazyIndex(path)
