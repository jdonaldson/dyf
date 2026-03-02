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
              "Run 'python demo/dyf_enrich.py project' first.",
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

    # Cluster data — parse cluster_{k}, cluster_{k}_2d, cluster_{k}_3d
    import re as _re
    _cluster_re = _re.compile(r'^cluster_(\d+)(?:_(2d|3d))?$')
    cluster_result = {
        'labels': {}, 'names': {}, 'centroids': {},
        'labels_2d': {}, 'names_2d': {}, 'centroids_2d': {},
        'labels_3d': {}, 'names_3d': {}, 'centroids_3d': {},
    }
    cluster_fields = sorted(
        [f for f in data['fields'] if _cluster_re.match(f)],
        key=lambda f: (int(_cluster_re.match(f).group(1)),
                       _cluster_re.match(f).group(2) or ''))

    def _load_cluster_level(field_name, lvl, suffix):
        """Load labels/names/centroids for a cluster field into result."""
        labels_arr = np.asarray(data['fields'][field_name])

        # Names from metadata
        names_key = f'cluster_names_{lvl}' + (f'_{suffix}' if suffix else '')
        names_json = data['metadata'].get(names_key, '{}')
        parsed = json.loads(names_json)
        if parsed:
            max_id = max(int(k) for k in parsed)
            names_list = [parsed.get(str(i), f"Cluster {i}")
                          for i in range(max_id + 1)]
        else:
            unique = sorted(set(labels_arr.tolist()))
            names_list = [f"Cluster {i}" for i in range(max(unique) + 1)]

        # Centroids from metadata or computed
        cent_key = f'cluster_centroids_{lvl}' + (f'_{suffix}' if suffix
                                                  else '')
        cent_json = data['metadata'].get(cent_key, '{}')
        cent_parsed = json.loads(cent_json)
        if cent_parsed:
            max_cid = max(int(k) for k in cent_parsed)
            centroids = np.zeros((max_cid + 1, 2), dtype=np.float32)
            for k, v in cent_parsed.items():
                centroids[int(k)] = [v[0], v[1]]
        else:
            unique = sorted(set(labels_arr.tolist()))
            centroids = np.zeros((max(unique) + 1, 2), dtype=np.float32)
            for cid in unique:
                mask = labels_arr == cid
                centroids[cid] = coords_2d[mask].mean(axis=0)

        return labels_arr, names_list, centroids

    has_dual = False
    for cf in cluster_fields:
        m = _cluster_re.match(cf)
        lvl = int(m.group(1))
        suffix = m.group(2)  # None, '2d', or '3d'

        labels_arr, names_list, centroids = _load_cluster_level(
            cf, lvl, suffix)

        if suffix == '2d':
            has_dual = True
            cluster_result['labels_2d'][lvl] = labels_arr
            cluster_result['names_2d'][lvl] = names_list
            cluster_result['centroids_2d'][lvl] = centroids
            # 2D is the default view
            cluster_result['labels'][lvl] = labels_arr
            cluster_result['names'][lvl] = names_list
            cluster_result['centroids'][lvl] = centroids
        elif suffix == '3d':
            has_dual = True
            cluster_result['labels_3d'][lvl] = labels_arr
            cluster_result['names_3d'][lvl] = names_list
            cluster_result['centroids_3d'][lvl] = centroids
        else:
            # Bare cluster_{k} — backward compat: populate both 2d and 3d
            cluster_result['labels'][lvl] = labels_arr
            cluster_result['names'][lvl] = names_list
            cluster_result['centroids'][lvl] = centroids
            cluster_result['labels_2d'][lvl] = labels_arr
            cluster_result['names_2d'][lvl] = names_list
            cluster_result['centroids_2d'][lvl] = centroids
            cluster_result['labels_3d'][lvl] = labels_arr
            cluster_result['names_3d'][lvl] = names_list
            cluster_result['centroids_3d'][lvl] = centroids

    # If no BIRCH clusters but tree labels exist, build from tree
    if not cluster_fields:
        for mk in sorted(data['metadata'].keys()):
            if mk.startswith('tree_labels_depth_'):
                tree_data = json.loads(data['metadata'][mk])
                child_labels_map = tree_data.get('child_labels', {})
                branch_labels_map = tree_data.get('branch_labels', {})

                # Assign points to tree children
                tree_struct = idx.get_tree_structure()
                parent_of = {nd['node_id']: nd['parent_id']
                             for nd in tree_struct}
                leaf_nodes = [nd for nd in tree_struct if nd['is_leaf']]
                labeled_nids = sorted(child_labels_map.keys(), key=int)
                nid_to_cid = {int(nid): i
                              for i, nid in enumerate(labeled_nids)}

                labels_arr = np.full(n, -1, dtype=np.int32)
                for ln in leaf_nodes:
                    if ln['batch_index'] < 0:
                        continue
                    batch = idx.get_leaf(ln['batch_index'])
                    item_idx = batch.column('item_index').to_numpy()
                    nid = ln['node_id']
                    while nid is not None:
                        if str(nid) in child_labels_map:
                            labels_arr[item_idx] = nid_to_cid[nid]
                            break
                        if str(nid) in branch_labels_map:
                            if nid not in nid_to_cid:
                                nid_to_cid[nid] = len(nid_to_cid)
                                child_labels_map[str(nid)] = \
                                    branch_labels_map[str(nid)]
                            labels_arr[item_idx] = nid_to_cid[nid]
                            break
                        nid = parent_of.get(nid)

                unassigned = labels_arr == -1
                if unassigned.any():
                    fb = max(nid_to_cid.values()) + 1
                    labels_arr[unassigned] = fb
                    child_labels_map[str(fb)] = "Other"
                    nid_to_cid[-1] = fb

                n_cls = len(set(labels_arr.tolist()))
                names_list = [""] * (max(labels_arr) + 1)
                for str_nid, cid in nid_to_cid.items():
                    if cid < len(names_list):
                        names_list[cid] = child_labels_map.get(
                            str(str_nid), f"Cluster {cid}")

                centroids = np.zeros((len(names_list), 2), dtype=np.float32)
                for cid in range(len(names_list)):
                    mask = labels_arr == cid
                    if mask.any():
                        centroids[cid] = coords_2d[mask].mean(axis=0)

                # Store as the default cluster level
                for lvl in [5, 12, 25, 50]:
                    cluster_result['labels'][lvl] = labels_arr
                    cluster_result['names'][lvl] = names_list
                    cluster_result['centroids'][lvl] = centroids

                print(f"  Built {n_cls} clusters from tree labels",
                      file=sys.stderr)
                break

    # Edge pairs from metadata
    cluster_pairs = {}
    edge_json = data['metadata'].get('edge_pairs')
    if edge_json:
        for src, dst, weight in json.loads(edge_json):
            cluster_pairs[(src, dst)] = weight

    # Path labels from cluster-tree DAG
    path_labels = {}
    for mk, mv in data['metadata'].items():
        if mk.startswith('cluster_path_labels_') and mv:
            # e.g. cluster_path_labels_25_2d → level 25
            parts = mk.replace('cluster_path_labels_', '').split('_')
            if parts:
                try:
                    lvl = int(parts[0])
                    path_labels[lvl] = json.loads(mv)
                except (ValueError, json.JSONDecodeError):
                    pass
    if path_labels:
        print(f"  Loaded path labels for levels: {sorted(path_labels.keys())}",
              file=sys.stderr)

    CACHE = {
        'titles': titles,
        'coords_2d': coords_2d,
        'cluster_result': cluster_result,
        'cluster_pairs': cluster_pairs,
        'path_labels': path_labels,
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

CLUSTER_LEVELS = (5, 12, 25, 50)


def search_points(query: str, limit: int = 20) -> list[dict]:
    """Search points by title (case-insensitive substring match)."""
    query_lower = query.lower()
    results = []
    for i, title in enumerate(CACHE['titles']):
        if query_lower in title.lower():
            result = {
                'index': i,
                'title': title,
                'x': float(CACHE['coords_2d'][i, 0]),
                'y': float(CACHE['coords_2d'][i, 1]),
            }
            for level in CLUSTER_LEVELS:
                if level in CACHE['cluster_result']['labels']:
                    result[f'cluster_{level}'] = int(CACHE['cluster_result']['labels'][level][i])
            results.append(result)
            if len(results) >= limit:
                break
    return results


def get_cluster_info(level: int = 5) -> list[dict]:
    """Get cluster information for a given level."""
    if level not in CLUSTER_LEVELS:
        return [{'error': f'Invalid level {level}. Must be one of {CLUSTER_LEVELS}.'}]

    labels = CACHE['cluster_result']['labels'][level]
    names = CACHE['cluster_result']['names'][level]
    centroids = CACHE['cluster_result']['centroids'][level]
    level_path_labels = CACHE.get('path_labels', {}).get(level, {})

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
        # Include path label from cluster-tree DAG if available
        pl = level_path_labels.get(str(i), "")
        if pl:
            entry['path_label'] = pl
        clusters.append(entry)

    return sorted(clusters, key=lambda c: -c['count'])


def get_cluster_members(cluster_id: int, level: int = 5, limit: int = 50) -> list[dict]:
    """Get members of a specific cluster."""
    if level not in CLUSTER_LEVELS:
        return [{'error': f'Invalid level {level}. Must be one of {CLUSTER_LEVELS}.'}]

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
    """Get bridge connections for a bucket (at 50-cluster level)."""
    cluster_pairs = CACHE.get('cluster_pairs', {})
    if not cluster_pairs:
        return {'error': 'No cluster_pairs data in cache. Re-run preprocessing.'}

    connections = []
    for (c1, c2), count in cluster_pairs.items():
        if int(c1) == bucket_id or int(c2) == bucket_id:
            other = int(c2) if int(c1) == bucket_id else int(c1)
            names = CACHE['cluster_result'].get('names', {}).get(50, [])
            other_name = names[other] if other < len(names) else f"Bucket {other}"
            connections.append({
                'bucket_id': other,
                'name': other_name,
                'connection_count': int(count),
            })

    names = CACHE['cluster_result'].get('names', {}).get(50, [])
    bucket_name = names[bucket_id] if bucket_id < len(names) else f"Bucket {bucket_id}"

    return {
        'bucket_id': bucket_id,
        'name': bucket_name,
        'total_connections': len(connections),
        'connections': sorted(connections, key=lambda x: -x['connection_count']),
    }


def get_bucket_members(bucket_id: int, limit: int = 20) -> list[dict]:
    """Get sample members of a bucket (at 50-cluster level)."""
    labels = CACHE['cluster_result']['labels'].get(50)
    if labels is None:
        return [{'error': 'No 50-cluster labels in cache.'}]

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
            description="Get information about clusters at a given level (5, 12, 25, or 50 clusters)",
            inputSchema={
                "type": "object",
                "properties": {
                    "level": {"type": "integer", "enum": [5, 12, 25, 50], "description": "Cluster level", "default": 5},
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
                    "level": {"type": "integer", "enum": [5, 12, 25, 50], "description": "Cluster level", "default": 5},
                    "limit": {"type": "integer", "description": "Max results (default 50)", "default": 50},
                },
                "required": ["cluster_id"],
            },
        ),
        Tool(
            name="get_bucket_connections",
            description="Get bridge connections for a bucket showing which other buckets it connects to (at 50-cluster level)",
            inputSchema={
                "type": "object",
                "properties": {
                    "bucket_id": {"type": "integer", "description": "Bucket ID (0-49)"},
                },
                "required": ["bucket_id"],
            },
        ),
        Tool(
            name="get_bucket_members",
            description="Get sample members of a bucket (at 50-cluster level)",
            inputSchema={
                "type": "object",
                "properties": {
                    "bucket_id": {"type": "integer", "description": "Bucket ID (0-49)"},
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
                    "level": {"type": "integer", "enum": [5, 12, 25, 50], "description": "Cluster level", "default": 5},
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
    # --- Data query tools (local, no WS) ---
    if name == "search_points":
        result = search_points(arguments["query"], arguments.get("limit", 20))
    elif name == "get_cluster_info":
        result = get_cluster_info(arguments.get("level", 5))
    elif name == "get_cluster_members":
        result = get_cluster_members(
            arguments["cluster_id"],
            arguments.get("level", 5),
            arguments.get("limit", 50),
        )
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

    # --- Camera/view tools ---
    elif name == "zoom_to_cluster":
        cluster_id = arguments["cluster_id"]
        level = arguments.get("level", 5)
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
                params = _cluster_zoom_params(bucket_id, 50)
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
    import asyncio
    asyncio.run(main())
