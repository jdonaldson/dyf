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
        quantization: "float32", "float16", or "int8" (default: "float16").
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

    # Quantize embeddings
    q_embeddings = _quantize_embeddings(embeddings, quantization)
    arrow_value_type = _numpy_dtype_to_arrow(quantization)

    # Build Arrow IPC batches per leaf
    ipc_options = None
    if compression == 'zstd':
        ipc_options = pa.ipc.IpcWriteOptions(
            compression=pa.Codec('zstd'))
    elif compression == 'lz4':
        ipc_options = pa.ipc.IpcWriteOptions(
            compression=pa.Codec('lz4'))

    # Schema for leaf batches
    arrow_schema = pa.schema([
        ('item_index', pa.uint32()),
        ('embedding', pa.list_(arrow_value_type, embedding_dim)),
    ])

    # Write each leaf batch to bytes, record offsets
    batch_buffers = []
    for node in flat_nodes:
        if not node['is_leaf']:
            continue
        indices = node['indices']
        leaf_emb = q_embeddings[indices]

        # Build Arrow RecordBatch
        item_idx_arr = pa.array(indices.astype(np.uint32), type=pa.uint32())

        # Build fixed-size list array for embeddings
        flat_values = leaf_emb.flatten()
        if quantization == 'int8':
            values_arr = pa.array(flat_values, type=pa.int8())
        elif quantization == 'float16':
            values_arr = pa.array(flat_values, type=pa.float16())
        else:
            values_arr = pa.array(flat_values, type=pa.float32())

        emb_arr = pa.FixedSizeListArray.from_arrays(values_arr, embedding_dim)
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
        return {
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
            bucket_id (uint64).
        """
        projections = hyperplanes @ query
        bits = (projections > 0).astype(np.uint64)
        bucket_id = np.uint64(0)
        for i, b in enumerate(bits):
            bucket_id |= (b << np.uint64(i))
        return int(bucket_id)

    def _hamming_distance(self, a, b, num_bits):
        """Hamming distance between two bucket IDs."""
        xor = a ^ b
        dist = 0
        for _ in range(num_bits):
            dist += xor & 1
            xor >>= 1
        return dist

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

        # Collect results from all candidate leaves
        all_indices = []
        all_scores = []

        for batch_index in candidate_leaves:
            batch = self.get_leaf(batch_index)
            item_indices = batch.column('item_index').to_numpy()

            # Extract embeddings from fixed-size list
            emb_col = batch.column('embedding')
            # Convert to numpy: flatten then reshape
            n_rows = len(emb_col)
            dim = self.embedding_dim

            # Get raw values from the fixed-size list array
            flat_values = emb_col.values.to_numpy()
            leaf_emb = flat_values.reshape(n_rows, dim).astype(np.float32)

            # Normalize for cosine similarity
            norms = np.linalg.norm(leaf_emb, axis=1, keepdims=True)
            leaf_emb_n = leaf_emb / np.maximum(norms, 1e-10)

            # Cosine similarities
            scores = leaf_emb_n @ query_normed

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
                bucket_id = self._hash_query(query, hyperplanes)
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
                    # Fallback: find nearest bucket by Hamming distance
                    best_dist = num_bits + 1
                    for bid, child_nid in bid_to_child.items():
                        dist = self._hamming_distance(bucket_id, bid, num_bits)
                        if dist < best_dist:
                            best_dist = dist
                            primary_child = child_nid
                            primary_bid = bid

                if primary_child is not None:
                    probe_queue.append((priority, primary_child))

                # For nprobe > 1: add alternative children sorted by Hamming distance
                if nprobe > 1:
                    alternatives = []
                    for bid, child_nid in bid_to_child.items():
                        if bid == primary_bid:
                            continue
                        dist = self._hamming_distance(bucket_id, bid, num_bits)
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
