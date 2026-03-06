"""
MCP Server for ROG (Recursive Ontological Graph) Browser.

Provides tools to search, explore clusters, and control the deck.gl
visualization via WebSocket commands to viz_server.py.

Run: python demo/rog_mcp.py demo/gudid_50k_titled.dyf
"""

import argparse
import json
import math
import sys
from pathlib import Path

import asyncio

import numpy as np
from scipy.cluster.hierarchy import fcluster
import websockets
from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import Tool, TextContent

# -----------------------------------------------------------------------------
# Cache and State
# -----------------------------------------------------------------------------

CACHE = None
WS_URL = "ws://localhost:8766/ws"
WS_CONN = None         # Persistent WebSocket connection to viz_server
DYF_INDEX = None       # LazyIndex instance for semantic search
EMBED_MODEL = None     # Lazy-loaded embedding model for semantic_search
LOCKED_LEVEL = None    # When set, all cluster tools use this level


def load_cache(dyf_path: str):
    """Load CACHE from an enriched .dyf file (Level 1+)."""
    global CACHE, DYF_INDEX
    src_dir = str(Path(__file__).resolve().parent.parent / "src")
    if src_dir not in sys.path:
        sys.path.insert(0, src_dir)
    from dyf.lazy_index import LazyIndex

    idx = LazyIndex(dyf_path)
    level = idx.detect_enrichment_level()
    print(f"Loading .dyf: {dyf_path} (enrichment level {level})",
          file=sys.stderr)
    if level < 1:
        print("ERROR: .dyf needs at least level 1 (UMAP coords). "
              "Run 'dyf enrich project' first.",
              file=sys.stderr)
        sys.exit(1)

    data = idx.extract_all_fields()
    n = len(data['embeddings'])

    # Titles
    titles = data['fields'].get('title')
    if titles is None:
        titles = [f"Item {i}" for i in range(n)]
    if isinstance(titles, np.ndarray):
        titles = titles.tolist()

    # 2D coords
    coords_2d = np.column_stack([
        data['fields']['umap_x'],
        data['fields']['umap_y'],
    ])

    # Cluster data — parse dendrogram metadata
    dendro_json = data['metadata'].get('louvain_dendrogram')
    leaf_comm_json = data['metadata'].get('louvain_leaf_communities')
    leaf_item_json = data['metadata'].get('leaf_item_map')

    cluster_result = {
        'labels': {}, 'names': {}, 'centroids': {},
    }
    dendro_data = None
    natural_k = None

    if dendro_json and leaf_comm_json and leaf_item_json:
        dendro_data = json.loads(dendro_json)
        leaf_comm_data = json.loads(leaf_comm_json)
        leaf_item_data = json.loads(leaf_item_json)
        Z = np.array(dendro_data['Z'])
        community_names = dendro_data['community_names']
        community_centroids = dendro_data.get('community_centroids', {})
        community_sizes = dendro_data.get('community_sizes', {})
        natural_k = leaf_comm_data['natural_k']
        leaf_to_comm = leaf_comm_data['leaf_to_community']

        # Use community_id stored field if available (correct post-reassignment
        # labels), otherwise reconstruct from leaf mappings (pre-reassignment)
        if 'community_id' in data['fields']:
            point_labels = np.asarray(
                data['fields']['community_id'], dtype=np.int32)
        else:
            point_labels = np.full(n, -1, dtype=np.int32)
            for leaf_idx, items in leaf_item_data.items():
                comm_id = leaf_to_comm.get(str(leaf_idx))
                if comm_id is None:
                    continue
                for item_idx in items:
                    if item_idx < n:
                        point_labels[item_idx] = int(comm_id)
            # Handle unassigned
            unassigned = point_labels == -1
            if unassigned.any():
                point_labels[unassigned] = 0

        # Build names and centroids arrays
        max_cid = max(int(k) for k in community_names)
        names_list = [community_names.get(str(i), f"Cluster {i}")
                      for i in range(max_cid + 1)]
        centroids = np.zeros((max_cid + 1, 2), dtype=np.float32)
        for cid_str, cent in community_centroids.items():
            cid = int(cid_str)
            if cid <= max_cid:
                centroids[cid] = [cent[0], cent[1]]

        # Store natural level as the single cluster level
        cluster_result['labels'][natural_k] = point_labels
        cluster_result['names'][natural_k] = names_list
        cluster_result['centroids'][natural_k] = centroids

        print(f"  Loaded dendrogram: {natural_k} communities, "
              f"Z={Z.shape[0]} merges", file=sys.stderr)

    # Edge pairs from metadata
    cluster_pairs = {}
    edge_json = data['metadata'].get('edge_pairs')
    if edge_json:
        for src, dst, weight in json.loads(edge_json):
            cluster_pairs[(src, dst)] = weight

    CACHE = {
        'titles': titles,
        'coords_2d': coords_2d,
        'cluster_result': cluster_result,
        'cluster_pairs': cluster_pairs,
        'dendro_data': dendro_data,
        'natural_k': natural_k,
    }

    # Also set DYF_INDEX for semantic search
    DYF_INDEX = idx

    print(f"Loaded .dyf cache: {n} points, level {level}, "
          f"cluster levels: {sorted(cluster_result['labels'].keys())}",
          file=sys.stderr)



def _get_embed_model():
    """Lazy-load the embedding model based on DYF index metadata."""
    global EMBED_MODEL
    if EMBED_MODEL is not None:
        return EMBED_MODEL

    meta = DYF_INDEX._get_metadata()
    model_name = meta.get('embedding_model', '')

    if 'clip' in model_name.lower():
        print(f"Loading CLIP model: {model_name} ...", file=sys.stderr)
        from transformers import CLIPModel, CLIPProcessor
        model = CLIPModel.from_pretrained(model_name)
        processor = CLIPProcessor.from_pretrained(model_name)
        EMBED_MODEL = ('clip', model, processor)
    else:
        print(f"Loading SentenceTransformer: {model_name} ...", file=sys.stderr)
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer(model_name)
        EMBED_MODEL = ('st', model)

    print("Model loaded.", file=sys.stderr)
    return EMBED_MODEL


def _encode_query(text: str) -> np.ndarray:
    """Encode query text to embedding vector using the loaded model."""
    import torch
    model_info = _get_embed_model()

    if model_info[0] == 'clip':
        _, model, processor = model_info
        inputs = processor(text=[text], return_tensors="pt", padding=True,
                           truncation=True)
        with torch.no_grad():
            text_features = model.get_text_features(**inputs)
        vec = text_features[0].numpy().astype(np.float32)
    else:
        _, model = model_info
        vec = model.encode(text, normalize_embeddings=True).astype(np.float32)

    # L2 normalize
    norm = np.linalg.norm(vec)
    if norm > 1e-10:
        vec = vec / norm
    return vec


def _format_search_results(result, include_coords=True) -> list[dict]:
    """Format SearchResult into a list of dicts for JSON response."""
    items = []
    for i in range(len(result.indices)):
        idx = int(result.indices[i])
        item = {
            'index': idx,
            'score': round(float(result.scores[i]), 4),
        }
        # Add title from stored fields or from CACHE
        if 'title' in result.fields:
            titles = result.fields['title']
            item['title'] = titles[i] if isinstance(titles, list) else str(titles[i])
        elif CACHE is not None and idx < len(CACHE['titles']):
            item['title'] = CACHE['titles'][idx]
        # Add 2D coords from CACHE if available
        if include_coords and CACHE is not None and idx < len(CACHE['coords_2d']):
            item['x'] = float(CACHE['coords_2d'][idx, 0])
            item['y'] = float(CACHE['coords_2d'][idx, 1])
        items.append(item)
    return items


# -----------------------------------------------------------------------------
# Query Functions
# -----------------------------------------------------------------------------

def _natural_k() -> int:
    """Return the natural cluster count from Louvain."""
    if CACHE is None or CACHE.get('natural_k') is None:
        return 12  # fallback before load
    return CACHE['natural_k']


def _cluster_levels():
    """Return cached cluster levels (informational, not a constraint)."""
    if CACHE is None:
        return (5, 12, 25, 50)  # fallback before load
    return tuple(sorted(CACHE['cluster_result']['labels'].keys()))


def _default_level():
    """Default cluster level (natural Louvain k)."""
    return _natural_k()


def _max_level():
    """Maximum cluster level (natural Louvain k)."""
    return _natural_k()


def _resolve_level(requested: int | None) -> int:
    """Resolve the effective cluster level, respecting lock."""
    if LOCKED_LEVEL is not None:
        return LOCKED_LEVEL
    if requested is not None:
        return requested
    return _natural_k()


def cut_dendrogram(target_k: int) -> tuple[np.ndarray, list[str], np.ndarray] | None:
    """Cut the dendrogram to target_k clusters.

    Returns (point_labels, names_list, centroids_2d) or None on failure.
    Results are cached in CACHE['cluster_result'].
    """
    if CACHE is None or CACHE['dendro_data'] is None:
        return None

    nk = _natural_k()

    # Already cached?
    if target_k in CACHE['cluster_result']['labels']:
        return (
            CACHE['cluster_result']['labels'][target_k],
            CACHE['cluster_result']['names'][target_k],
            CACHE['cluster_result']['centroids'][target_k],
        )

    if target_k < 1 or target_k > nk:
        return None

    Z = np.array(CACHE['dendro_data']['Z'])
    community_names = CACHE['dendro_data']['community_names']
    community_centroids = CACHE['dendro_data'].get('community_centroids', {})
    community_sizes = CACHE['dendro_data'].get('community_sizes', {})

    # fcluster returns 1-indexed labels for each of the nk communities
    merged = fcluster(Z, target_k, criterion='maxclust')
    merged = merged - 1  # 0-indexed

    # Map points: each point's community_id -> merged group
    natural_labels = CACHE['cluster_result']['labels'][nk]
    point_labels = np.array([int(merged[c]) for c in natural_labels],
                            dtype=np.int32)

    # Build merged group info
    group_children = {}  # group_id -> [(comm_id, size), ...]
    for comm_id in range(nk):
        group_id = int(merged[comm_id])
        size = int(community_sizes.get(str(comm_id), 0))
        group_children.setdefault(group_id, []).append((comm_id, size))

    max_group = max(group_children.keys()) if group_children else 0
    names_list = []
    centroids_2d = np.zeros((max_group + 1, 2), dtype=np.float32)

    for group_id in range(max_group + 1):
        children = group_children.get(group_id, [])
        if not children:
            names_list.append(f"Cluster {group_id}")
            continue
        # Name from largest child community
        children.sort(key=lambda x: -x[1])
        largest_comm_id = children[0][0]
        names_list.append(
            community_names.get(str(largest_comm_id), f"Cluster {group_id}"))
        # Centroid: size-weighted average of child centroids
        total_size = sum(s for _, s in children)
        if total_size > 0:
            cx, cy = 0.0, 0.0
            for comm_id, size in children:
                cent = community_centroids.get(str(comm_id), [0.0, 0.0])
                cx += cent[0] * size
                cy += cent[1] * size
            centroids_2d[group_id] = [cx / total_size, cy / total_size]

    # Cache result
    CACHE['cluster_result']['labels'][target_k] = point_labels
    CACHE['cluster_result']['names'][target_k] = names_list
    CACHE['cluster_result']['centroids'][target_k] = centroids_2d

    return (point_labels, names_list, centroids_2d)


def _ensure_level(level: int) -> dict | None:
    """Ensure cluster data exists for the given level.

    Returns error dict if invalid, None if OK.
    """
    nk = _natural_k()
    if level < 1 or level > nk:
        return {'error': f'Invalid level {level}. Must be 1-{nk}.'}
    result = cut_dendrogram(level)
    if result is None:
        return {'error': 'No dendrogram data available.'}
    return None


def search_points(query: str, limit: int = 20) -> list[dict]:
    """Search points by title (case-insensitive substring match)."""
    query_lower = query.lower()
    level = _resolve_level(None)
    _ensure_level(level)
    results = []
    for i, title in enumerate(CACHE['titles']):
        if query_lower in title.lower():
            result = {
                'index': i,
                'title': title,
                'x': float(CACHE['coords_2d'][i, 0]),
                'y': float(CACHE['coords_2d'][i, 1]),
            }
            if level in CACHE['cluster_result']['labels']:
                result['community_id'] = int(
                    CACHE['cluster_result']['labels'][level][i])
            results.append(result)
            if len(results) >= limit:
                break
    return results


def get_cluster_info(level: int = None) -> list[dict]:
    """Get cluster information for a given level."""
    level = _resolve_level(level)
    err = _ensure_level(level)
    if err:
        return [err]

    labels = CACHE['cluster_result']['labels'][level]
    names = CACHE['cluster_result']['names'][level]
    centroids = CACHE['cluster_result']['centroids'][level]

    clusters = []
    unique_ids = sorted(set(int(x) for x in labels))
    for i in unique_ids:
        mask = labels == i
        count = int(mask.sum())
        if count == 0:
            continue
        name = names[i] if i < len(names) else f"Cluster {i}"
        cx = float(centroids[i, 0]) if i < len(centroids) else 0.0
        cy = float(centroids[i, 1]) if i < len(centroids) else 0.0
        entry = {
            'cluster_id': i,
            'name': name,
            'count': count,
            'centroid_x': cx,
            'centroid_y': cy,
        }
        clusters.append(entry)

    return sorted(clusters, key=lambda c: -c['count'])


def get_cluster_members(cluster_id: int, level: int = None, limit: int = 50) -> list[dict]:
    """Get members of a specific cluster."""
    level = _resolve_level(level)
    err = _ensure_level(level)
    if err:
        return [err]

    labels = CACHE['cluster_result']['labels'][level]
    mask = labels == cluster_id
    indices = np.where(mask)[0][:limit]

    return [{
        'index': int(i),
        'title': CACHE['titles'][i],
        'x': float(CACHE['coords_2d'][i, 0]),
        'y': float(CACHE['coords_2d'][i, 1]),
    } for i in indices]


def get_neighbors(index: int, k: int = 10) -> list[dict]:
    """Find k nearest neighbors to a point by 2D distance."""
    if index < 0 or index >= len(CACHE['titles']):
        return [{'error': f'Invalid index {index}'}]

    target = CACHE['coords_2d'][index]
    distances = np.linalg.norm(CACHE['coords_2d'] - target, axis=1)
    nearest = np.argsort(distances)[1:k+1]

    return [{
        'index': int(i),
        'title': CACHE['titles'][i],
        'distance': float(distances[i]),
        'x': float(CACHE['coords_2d'][i, 0]),
        'y': float(CACHE['coords_2d'][i, 1]),
    } for i in nearest]


def get_points_in_region(x_min: float, x_max: float, y_min: float, y_max: float, limit: int = 100) -> list[dict]:
    """Get points within a bounding box."""
    coords = CACHE['coords_2d']
    mask = (
        (coords[:, 0] >= x_min) & (coords[:, 0] <= x_max) &
        (coords[:, 1] >= y_min) & (coords[:, 1] <= y_max)
    )
    indices = np.where(mask)[0][:limit]

    return [{
        'index': int(i),
        'title': CACHE['titles'][i],
        'x': float(CACHE['coords_2d'][i, 0]),
        'y': float(CACHE['coords_2d'][i, 1]),
    } for i in indices]


def get_bucket_connections(bucket_id: int) -> dict:
    """Get bridge connections for a bucket (at finest cluster level)."""
    cluster_pairs = CACHE.get('cluster_pairs', {})
    if not cluster_pairs:
        return {'error': 'No cluster_pairs data in cache. Re-run preprocessing.'}

    ml = _max_level()
    connections = []
    for (c1, c2), count in cluster_pairs.items():
        if int(c1) == bucket_id or int(c2) == bucket_id:
            other = int(c2) if int(c1) == bucket_id else int(c1)
            names = CACHE['cluster_result'].get('names', {}).get(ml, [])
            other_name = names[other] if other < len(names) else f"Bucket {other}"
            connections.append({
                'bucket_id': other,
                'name': other_name,
                'connection_count': int(count),
            })

    names = CACHE['cluster_result'].get('names', {}).get(ml, [])
    bucket_name = names[bucket_id] if bucket_id < len(names) else f"Bucket {bucket_id}"

    return {
        'bucket_id': bucket_id,
        'name': bucket_name,
        'total_connections': len(connections),
        'connections': sorted(connections, key=lambda x: -x['connection_count']),
    }


def get_bucket_members(bucket_id: int, limit: int = 20) -> list[dict]:
    """Get sample members of a bucket (at finest cluster level)."""
    ml = _max_level()
    labels = CACHE['cluster_result']['labels'].get(ml)
    if labels is None:
        return [{'error': f'No {ml}-cluster labels in cache.'}]

    mask = labels == bucket_id
    indices = np.where(mask)[0][:limit]

    return [{
        'index': int(i),
        'title': CACHE['titles'][i],
        'x': float(CACHE['coords_2d'][i, 0]),
        'y': float(CACHE['coords_2d'][i, 1]),
    } for i in indices]


# -----------------------------------------------------------------------------
# WebSocket Control
# -----------------------------------------------------------------------------

async def send_ws(cmd: dict) -> dict:
    """Send a command to the viz_server via a persistent WebSocket."""
    global WS_CONN
    try:
        if WS_CONN is None:
            WS_CONN = await websockets.connect(WS_URL)
        await WS_CONN.send(json.dumps(cmd))
        return {"ok": True, "cmd": cmd.get("cmd")}
    except Exception:
        # Connection lost — reconnect once and retry
        WS_CONN = None
        try:
            WS_CONN = await websockets.connect(WS_URL)
            await WS_CONN.send(json.dumps(cmd))
            return {"ok": True, "cmd": cmd.get("cmd")}
        except Exception as e:
            WS_CONN = None
            return {"error": str(e), "hint": "Is viz_server.py running?"}


def _cluster_zoom_params(cluster_id: int, level: int) -> dict | None:
    """Compute target and zoom for a cluster."""
    centroids = CACHE['cluster_result']['centroids'][level]
    if cluster_id >= len(centroids):
        return None
    cx, cy = centroids[cluster_id]
    labels = CACHE['cluster_result']['labels'][level]
    mask = labels == cluster_id
    coords = CACHE['coords_2d'][mask]
    dists = np.sqrt((coords[:, 0] - cx) ** 2 + (coords[:, 1] - cy) ** 2)
    radius = float(np.percentile(dists, 95)) * 1.3
    radius = max(radius, 0.5)
    # Convert radius to deck.gl zoom: zoom ≈ log2(viewport_half / radius)
    zoom = math.log2(400 / max(radius, 0.01))
    zoom = max(1.0, min(zoom, 15.0))
    return {"target": [float(cx), float(cy), 0], "zoom": zoom}


# -----------------------------------------------------------------------------
# MCP Server
# -----------------------------------------------------------------------------

app = Server("rog-browser")


@app.list_tools()
async def list_tools():
    tools = [
        # --- Data query tools ---
        Tool(
            name="search_points",
            description="Search for points by title (case-insensitive substring match)",
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Search query"},
                    "limit": {"type": "integer", "description": "Max results (default 20)", "default": 20},
                },
                "required": ["query"],
            },
        ),
        Tool(
            name="get_cluster_info",
            description=f"Get information about clusters at a given level (default: {_natural_k()}, range 1-{_natural_k()})",
            inputSchema={
                "type": "object",
                "properties": {
                    "level": {"type": "integer", "description": f"Number of clusters (default: {_natural_k()}, range 1-{_natural_k()})", "default": _default_level()},
                },
            },
        ),
        Tool(
            name="get_cluster_members",
            description="Get sample members of a specific cluster",
            inputSchema={
                "type": "object",
                "properties": {
                    "cluster_id": {"type": "integer", "description": "Cluster ID"},
                    "level": {"type": "integer", "description": f"Number of clusters (default: {_natural_k()}, range 1-{_natural_k()})", "default": _default_level()},
                    "limit": {"type": "integer", "description": "Max results (default 50)", "default": 50},
                },
                "required": ["cluster_id"],
            },
        ),
        Tool(
            name="get_bucket_connections",
            description=f"Get bridge connections for a bucket showing which other buckets it connects to (at {_max_level()}-cluster level)",
            inputSchema={
                "type": "object",
                "properties": {
                    "bucket_id": {"type": "integer", "description": f"Bucket ID (0-{_max_level()-1})"},
                },
                "required": ["bucket_id"],
            },
        ),
        Tool(
            name="get_bucket_members",
            description=f"Get sample members of a bucket (at {_max_level()}-cluster level)",
            inputSchema={
                "type": "object",
                "properties": {
                    "bucket_id": {"type": "integer", "description": f"Bucket ID (0-{_max_level()-1})"},
                    "limit": {"type": "integer", "description": "Max results (default 20)", "default": 20},
                },
                "required": ["bucket_id"],
            },
        ),
        Tool(
            name="get_neighbors",
            description="Find nearest neighbors to a point by 2D distance",
            inputSchema={
                "type": "object",
                "properties": {
                    "index": {"type": "integer", "description": "Point index"},
                    "k": {"type": "integer", "description": "Number of neighbors (default 10)", "default": 10},
                },
                "required": ["index"],
            },
        ),
        Tool(
            name="get_points_in_region",
            description="Get points within a bounding box",
            inputSchema={
                "type": "object",
                "properties": {
                    "x_min": {"type": "number", "description": "Minimum x coordinate"},
                    "x_max": {"type": "number", "description": "Maximum x coordinate"},
                    "y_min": {"type": "number", "description": "Minimum y coordinate"},
                    "y_max": {"type": "number", "description": "Maximum y coordinate"},
                    "limit": {"type": "integer", "description": "Max results (default 100)", "default": 100},
                },
                "required": ["x_min", "x_max", "y_min", "y_max"],
            },
        ),
        # --- Camera/view tools ---
        Tool(
            name="zoom_to_cluster",
            description="Zoom the visualization to show a specific cluster",
            inputSchema={
                "type": "object",
                "properties": {
                    "cluster_id": {"type": "integer", "description": "Cluster ID to zoom to"},
                    "level": {"type": "integer", "description": f"Number of clusters (default: {_natural_k()}, range 1-{_natural_k()})", "default": _default_level()},
                },
                "required": ["cluster_id"],
            },
        ),
        Tool(
            name="zoom_to_point",
            description="Zoom the visualization to center on a specific point",
            inputSchema={
                "type": "object",
                "properties": {
                    "index": {"type": "integer", "description": "Point index to zoom to"},
                    "radius": {"type": "number", "description": "View radius around point (default 5)", "default": 5},
                },
                "required": ["index"],
            },
        ),
        Tool(
            name="reset_view",
            description="Reset the visualization to the default view",
            inputSchema={"type": "object", "properties": {}},
        ),
        Tool(
            name="start_tour",
            description="Start or stop the cluster tour. The tour visits each cluster, showing its edges and neighbors.",
            inputSchema={"type": "object", "properties": {}},
        ),
        # --- Visibility tools ---
        Tool(
            name="highlight_points",
            description="Highlight specific points in the visualization (auto-clears after 3s)",
            inputSchema={
                "type": "object",
                "properties": {
                    "indices": {"type": "array", "items": {"type": "integer"}, "description": "Point indices to highlight"},
                },
                "required": ["indices"],
            },
        ),
        Tool(
            name="highlight_bucket",
            description="Highlight a bucket and show its bridge connections to other buckets",
            inputSchema={
                "type": "object",
                "properties": {
                    "bucket_id": {"type": "integer", "description": "Bucket ID (0-49) to highlight"},
                    "zoom": {"type": "boolean", "description": "Whether to zoom to the bucket (default true)", "default": True},
                },
                "required": ["bucket_id"],
            },
        ),
        Tool(
            name="clear_highlight",
            description="Clear point highlights and restore normal colors",
            inputSchema={"type": "object", "properties": {}},
        ),
        Tool(
            name="hide_cluster",
            description="Hide a specific cluster from the visualization",
            inputSchema={
                "type": "object",
                "properties": {
                    "cluster_id": {"type": "integer", "description": "Cluster ID to hide"},
                },
                "required": ["cluster_id"],
            },
        ),
        Tool(
            name="show_cluster",
            description="Show a previously hidden cluster",
            inputSchema={
                "type": "object",
                "properties": {
                    "cluster_id": {"type": "integer", "description": "Cluster ID to show"},
                },
                "required": ["cluster_id"],
            },
        ),
        Tool(
            name="isolate_cluster",
            description="Show only a single cluster, hiding all others",
            inputSchema={
                "type": "object",
                "properties": {
                    "cluster_id": {"type": "integer", "description": "Cluster ID to isolate"},
                },
                "required": ["cluster_id"],
            },
        ),
        Tool(
            name="show_all_clusters",
            description="Show all clusters (undo hide/isolate)",
            inputSchema={"type": "object", "properties": {}},
        ),
        Tool(
            name="show_neighbors",
            description="Find and highlight the k nearest neighbors of a point, zoom to the neighborhood, and return neighbor details",
            inputSchema={
                "type": "object",
                "properties": {
                    "index": {"type": "integer", "description": "Point index to find neighbors of"},
                    "k": {"type": "integer", "description": "Number of neighbors (default 10)", "default": 10},
                },
                "required": ["index"],
            },
        ),
        # --- Display tools ---
        Tool(
            name="highlight_edges",
            description="Highlight bridge edges connected to specific clusters. Auto-switches to 2D mode and enables edges.",
            inputSchema={
                "type": "object",
                "properties": {
                    "cluster_ids": {
                        "type": "array",
                        "items": {"type": "integer"},
                        "description": "Cluster IDs whose edges to highlight",
                    },
                },
                "required": ["cluster_ids"],
            },
        ),
        Tool(
            name="clear_edge_highlight",
            description="Clear edge highlighting and restore normal edge colors",
            inputSchema={"type": "object", "properties": {}},
        ),
        Tool(
            name="toggle_edges",
            description="Show or hide bridge edges",
            inputSchema={
                "type": "object",
                "properties": {
                    "visible": {"type": "boolean", "description": "Whether edges should be visible"},
                },
                "required": ["visible"],
            },
        ),
        Tool(
            name="toggle_labels",
            description="Show or hide cluster labels",
            inputSchema={
                "type": "object",
                "properties": {
                    "visible": {"type": "boolean", "description": "Whether labels should be visible"},
                },
                "required": ["visible"],
            },
        ),
        Tool(
            name="set_point_size",
            description="Set the point size in the visualization",
            inputSchema={
                "type": "object",
                "properties": {
                    "size": {"type": "number", "description": "Point size (1-8)", "minimum": 1, "maximum": 8},
                },
                "required": ["size"],
            },
        ),
        Tool(
            name="set_mode",
            description="Switch between 2D and 3D visualization modes",
            inputSchema={
                "type": "object",
                "properties": {
                    "mode": {"type": "string", "enum": ["2d", "3d"], "description": "Visualization mode"},
                },
                "required": ["mode"],
            },
        ),
        Tool(
            name="set_theme",
            description="Switch between dark and light themes",
            inputSchema={
                "type": "object",
                "properties": {
                    "theme": {"type": "string", "enum": ["dark", "light"], "description": "Theme"},
                },
                "required": ["theme"],
            },
        ),
        Tool(
            name="set_color_mode",
            description="Switch point coloring between cluster communities and LSH buckets",
            inputSchema={
                "type": "object",
                "properties": {
                    "mode": {"type": "string", "enum": ["cluster", "bucket"],
                             "description": "Color mode"},
                },
                "required": ["mode"],
            },
        ),
        # --- Annotation tools ---
        Tool(
            name="draw_circle",
            description="Draw a highlighter circle around a cluster's convex hull. Multiple circles can be drawn; use draw_clear to remove.",
            inputSchema={
                "type": "object",
                "properties": {
                    "cluster_id": {"type": "integer", "description": "Cluster ID to circle"},
                    "color": {"type": "string", "description": "CSS rgba color (default: 'rgba(255,230,0,0.35)')"},
                    "width": {"type": "number", "description": "Stroke width in pixels (default: 18)"},
                },
                "required": ["cluster_id"],
            },
        ),
        Tool(
            name="draw_path",
            description="Draw a freeform highlighter path through data-space points",
            inputSchema={
                "type": "object",
                "properties": {
                    "points": {
                        "type": "array",
                        "items": {"type": "array", "items": {"type": "number"}},
                        "description": "Array of [x,y,z] coordinates in data space",
                    },
                    "color": {"type": "string", "description": "CSS rgba color (default: 'rgba(255,230,0,0.35)')"},
                    "width": {"type": "number", "description": "Stroke width in pixels (default: 18)"},
                },
                "required": ["points"],
            },
        ),
        Tool(
            name="draw_clear",
            description="Clear all highlighter annotations (circles and paths)",
            inputSchema={"type": "object", "properties": {}},
        ),
        # --- Lock tools ---
        Tool(
            name="lock_clusters",
            description="Lock cluster level to a specific k. All cluster tools will use this level until unlocked.",
            inputSchema={
                "type": "object",
                "properties": {
                    "level": {"type": "integer", "description": f"Cluster level to lock to (1-{_natural_k()})"},
                },
                "required": ["level"],
            },
        ),
        Tool(
            name="unlock_clusters",
            description="Unlock cluster level. Cluster tools will use their level parameter or natural_k default.",
            inputSchema={"type": "object", "properties": {}},
        ),
    ]

    # Add semantic search tools only when a DYF index is loaded
    if DYF_INDEX is not None:
        tools.extend([
            Tool(
                name="find_similar",
                description="Find semantically similar items using high-dimensional embeddings from the .dyf index. Much more accurate than 2D distance neighbors.",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "index": {"type": "integer", "description": "Point index to find similar items for"},
                        "k": {"type": "integer", "description": "Number of results (default 20)", "default": 20},
                    },
                    "required": ["index"],
                },
            ),
            Tool(
                name="semantic_search",
                description="Search for items by free-text query using the embedding model. Encodes your query and finds nearest items in embedding space.",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "query": {"type": "string", "description": "Free-text search query"},
                        "k": {"type": "integer", "description": "Number of results (default 20)", "default": 20},
                    },
                    "required": ["query"],
                },
            ),
        ])

    return tools


@app.call_tool()
async def call_tool(name: str, arguments: dict):
    global LOCKED_LEVEL
    # --- Data query tools (local, no WS) ---
    if name == "search_points":
        result = search_points(arguments["query"], arguments.get("limit", 20))
    elif name == "get_cluster_info":
        result = get_cluster_info(arguments.get("level"))
        if LOCKED_LEVEL is not None and result and 'error' not in result[0]:
            result = {"locked": True, "locked_level": LOCKED_LEVEL,
                      "clusters": result}
    elif name == "get_cluster_members":
        result = get_cluster_members(
            arguments["cluster_id"],
            arguments.get("level"),
            arguments.get("limit", 50),
        )
        if LOCKED_LEVEL is not None and result and 'error' not in result[0]:
            result = {"locked": True, "locked_level": LOCKED_LEVEL,
                      "members": result}
    elif name == "get_bucket_connections":
        result = get_bucket_connections(arguments["bucket_id"])
    elif name == "get_bucket_members":
        result = get_bucket_members(arguments["bucket_id"], arguments.get("limit", 20))
    elif name == "get_neighbors":
        result = get_neighbors(arguments["index"], arguments.get("k", 10))
    elif name == "get_points_in_region":
        result = get_points_in_region(
            arguments["x_min"], arguments["x_max"],
            arguments["y_min"], arguments["y_max"],
            arguments.get("limit", 100),
        )

    # --- Lock tools ---
    elif name == "lock_clusters":
        level = arguments["level"]
        nk = _natural_k()
        if level < 1 or level > nk:
            result = {"error": f"Invalid level {level}. Must be 1-{nk}."}
        else:
            LOCKED_LEVEL = level
            _ensure_level(level)
            await send_ws({"cmd": "set_level", "level": level})
            result = {"locked": True, "locked_level": level}
    elif name == "unlock_clusters":
        LOCKED_LEVEL = None
        await send_ws({"cmd": "set_level", "level": _natural_k()})
        result = {"locked": False}

    # --- Camera/view tools ---
    elif name == "zoom_to_cluster":
        cluster_id = arguments["cluster_id"]
        level = _resolve_level(arguments.get("level"))
        err = _ensure_level(level)
        if err:
            result = err
        else:
            # Switch browser cluster level to match
            await send_ws({"cmd": "set_level", "level": level})
            params = _cluster_zoom_params(cluster_id, level)
            if params:
                result = await send_ws({"cmd": "zoom_to", **params})
            else:
                result = {"error": f"Invalid cluster_id {cluster_id}"}
    elif name == "zoom_to_point":
        index = arguments["index"]
        radius = arguments.get("radius", 5)
        if 0 <= index < len(CACHE['coords_2d']):
            x, y = CACHE['coords_2d'][index]
            zoom = math.log2(400 / max(radius, 0.01))
            zoom = max(1.0, min(zoom, 15.0))
            result = await send_ws({"cmd": "zoom_to", "target": [float(x), float(y), 0], "zoom": zoom})
        else:
            result = {"error": f"Invalid index {index}"}
    elif name == "reset_view":
        result = await send_ws({"cmd": "reset_view"})
    elif name == "start_tour":
        result = await send_ws({"cmd": "tour"})

    # --- Visibility tools ---
    elif name == "highlight_points":
        result = await send_ws({"cmd": "highlight", "indices": arguments["indices"]})
    elif name == "highlight_bucket":
        bucket_id = arguments["bucket_id"]
        members = get_bucket_members(bucket_id, limit=200)
        if members and 'error' not in members[0]:
            indices = [m['index'] for m in members]
            result = await send_ws({"cmd": "highlight", "indices": indices})
            if arguments.get("zoom", True):
                params = _cluster_zoom_params(bucket_id, _max_level())
                if params:
                    await send_ws({"cmd": "zoom_to", **params})
            connections = get_bucket_connections(bucket_id)
            result = {"highlight": result, "connections": connections}
        else:
            result = {"error": f"Could not get members for bucket {bucket_id}"}
    elif name == "clear_highlight":
        result = await send_ws({"cmd": "clear_highlight"})
    elif name == "show_neighbors":
        index = arguments["index"]
        k = arguments.get("k", 10)
        neighbors = get_neighbors(index, k)
        if neighbors and 'error' not in neighbors[0]:
            indices = [index] + [n['index'] for n in neighbors]
            await send_ws({"cmd": "highlight", "indices": indices})
            # Zoom to bounding box of the neighborhood
            xs = [CACHE['coords_2d'][i, 0] for i in indices]
            ys = [CACHE['coords_2d'][i, 1] for i in indices]
            cx = float(sum(xs) / len(xs))
            cy = float(sum(ys) / len(ys))
            spread = max(max(xs) - min(xs), max(ys) - min(ys), 0.5)
            zoom = math.log2(400 / max(spread * 0.7, 0.01))
            zoom = max(1.0, min(zoom, 15.0))
            await send_ws({"cmd": "zoom_to", "target": [cx, cy, 0], "zoom": zoom})
            result = {"point": CACHE['titles'][index], "neighbors": neighbors}
        else:
            result = neighbors
    elif name == "hide_cluster":
        result = await send_ws({"cmd": "hide", "cluster": arguments["cluster_id"]})
    elif name == "show_cluster":
        result = await send_ws({"cmd": "show", "cluster": arguments["cluster_id"]})
    elif name == "isolate_cluster":
        result = await send_ws({"cmd": "isolate", "cluster": arguments["cluster_id"]})
    elif name == "show_all_clusters":
        result = await send_ws({"cmd": "show_all"})

    # --- Display tools ---
    elif name == "highlight_edges":
        result = await send_ws({"cmd": "highlight_edges", "clusters": arguments["cluster_ids"]})
    elif name == "clear_edge_highlight":
        result = await send_ws({"cmd": "clear_edge_highlight"})
    elif name == "toggle_edges":
        result = await send_ws({"cmd": "toggle_edges", "visible": arguments["visible"]})
    elif name == "toggle_labels":
        result = await send_ws({"cmd": "toggle_labels", "visible": arguments["visible"]})
    elif name == "set_point_size":
        result = await send_ws({"cmd": "point_size", "size": arguments["size"]})
    elif name == "set_mode":
        result = await send_ws({"cmd": "set_mode", "mode": arguments["mode"]})
    elif name == "set_theme":
        result = await send_ws({"cmd": "set_theme", "theme": arguments["theme"]})
    elif name == "set_color_mode":
        result = await send_ws({"cmd": "set_color_mode", "mode": arguments["mode"]})

    # --- Annotation tools ---
    elif name == "draw_circle":
        result = await send_ws({
            "cmd": "draw_circle",
            "cluster": arguments["cluster_id"],
            **({"color": arguments["color"]} if "color" in arguments else {}),
            **({"width": arguments["width"]} if "width" in arguments else {}),
        })
    elif name == "draw_path":
        result = await send_ws({
            "cmd": "draw_path",
            "points": arguments["points"],
            **({"color": arguments["color"]} if "color" in arguments else {}),
            **({"width": arguments["width"]} if "width" in arguments else {}),
        })
    elif name == "draw_clear":
        result = await send_ws({"cmd": "draw_clear"})

    # --- Semantic search tools (require DYF index) ---
    elif name == "find_similar":
        if DYF_INDEX is None:
            result = {"error": "No DYF index loaded. Start with --dyf-index."}
        else:
            index = arguments["index"]
            k = arguments.get("k", 20)
            try:
                vec = DYF_INDEX.get_item_vector(index)
                search_result = DYF_INDEX.search(vec, k=k + 1, nprobe=5)
                items = _format_search_results(search_result)
                # Remove the query item itself from results
                items = [it for it in items if it['index'] != index][:k]
                # Highlight + zoom in viz
                all_indices = [index] + [it['index'] for it in items]
                highlight_indices = [i for i in all_indices
                                     if CACHE is not None and i < len(CACHE['coords_2d'])]
                if highlight_indices:
                    await send_ws({"cmd": "highlight", "indices": highlight_indices})
                    xs = [CACHE['coords_2d'][i, 0] for i in highlight_indices]
                    ys = [CACHE['coords_2d'][i, 1] for i in highlight_indices]
                    cx = float(sum(xs) / len(xs))
                    cy = float(sum(ys) / len(ys))
                    spread = max(max(xs) - min(xs), max(ys) - min(ys), 0.5)
                    zoom = math.log2(400 / max(spread * 0.7, 0.01))
                    zoom = max(1.0, min(zoom, 15.0))
                    await send_ws({"cmd": "zoom_to", "target": [cx, cy, 0], "zoom": zoom})
                source_title = CACHE['titles'][index] if CACHE is not None and index < len(CACHE['titles']) else f"item {index}"
                result = {"source": source_title, "similar": items}
            except KeyError as e:
                result = {"error": str(e)}

    elif name == "semantic_search":
        if DYF_INDEX is None:
            result = {"error": "No DYF index loaded. Start with --dyf-index."}
        else:
            query = arguments["query"]
            k = arguments.get("k", 20)
            try:
                vec = _encode_query(query)
                search_result = DYF_INDEX.search(vec, k=k, nprobe=5)
                items = _format_search_results(search_result)
                # Highlight + zoom in viz
                highlight_indices = [it['index'] for it in items
                                     if CACHE is not None and it['index'] < len(CACHE['coords_2d'])]
                if highlight_indices:
                    await send_ws({"cmd": "highlight", "indices": highlight_indices})
                    xs = [CACHE['coords_2d'][i, 0] for i in highlight_indices]
                    ys = [CACHE['coords_2d'][i, 1] for i in highlight_indices]
                    cx = float(sum(xs) / len(xs))
                    cy = float(sum(ys) / len(ys))
                    spread = max(max(xs) - min(xs), max(ys) - min(ys), 0.5)
                    zoom = math.log2(400 / max(spread * 0.7, 0.01))
                    zoom = max(1.0, min(zoom, 15.0))
                    await send_ws({"cmd": "zoom_to", "target": [cx, cy, 0], "zoom": zoom})
                result = {"query": query, "results": items}
            except Exception as e:
                result = {"error": str(e)}

    else:
        result = {"error": f"Unknown tool: {name}"}

    return [TextContent(type="text", text=json.dumps(result, indent=2))]


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("dyf_path", help="Path to enriched .dyf file (Level 1+)")
    parser.add_argument("--ws-url", default="ws://localhost:8766/ws",
                        help="WebSocket URL for viz_server.py")
    args = parser.parse_args()

    global WS_URL
    WS_URL = args.ws_url

    load_cache(args.dyf_path)

    async with stdio_server() as (read_stream, write_stream):
        await app.run(read_stream, write_stream, app.create_initialization_options())


if __name__ == "__main__":
    asyncio.run(main())
