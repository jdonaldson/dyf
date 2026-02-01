"""
MCP Server for ROG (Recursive Ontological Graph) Browser.

Provides tools to search, explore clusters, and control the visualization.

Run: python demo/rog_mcp.py demo/wiki_simple_50k_rog_cache.pkl
"""

import argparse
import json
import pickle
import sys
from pathlib import Path

import numpy as np
import httpx
from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import Tool, TextContent

# -----------------------------------------------------------------------------
# Cache and State
# -----------------------------------------------------------------------------

CACHE = None
CONTROL_URL = "http://localhost:5008"


def load_cache(cache_path: str):
    """Load preprocessed cache."""
    global CACHE
    with open(cache_path, 'rb') as f:
        CACHE = pickle.load(f)
    print(f"Loaded cache: {len(CACHE['titles'])} points", file=sys.stderr)


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
            # Add cluster labels for each level
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

    clusters = []
    for i in range(level):
        mask = labels == i
        count = int(mask.sum())
        clusters.append({
            'cluster_id': i,
            'name': names[i],
            'count': count,
            'centroid_x': float(centroids[i, 0]),
            'centroid_y': float(centroids[i, 1]),
        })

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
    nearest = np.argsort(distances)[1:k+1]  # Skip self

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

    # Find all connections involving this bucket
    connections = []
    for (c1, c2), count in cluster_pairs.items():
        if int(c1) == bucket_id or int(c2) == bucket_id:
            other = int(c2) if int(c1) == bucket_id else int(c1)
            # Get bucket name if available
            names = CACHE['cluster_result'].get('names', {}).get(50, [])
            other_name = names[other] if other < len(names) else f"Bucket {other}"
            connections.append({
                'bucket_id': other,
                'name': other_name,
                'connection_count': int(count),
            })

    # Get this bucket's name
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
# Visualization Control
# -----------------------------------------------------------------------------

async def control_viz(action: str, params: dict) -> dict:
    """Send control command to the ROG control server."""
    try:
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{CONTROL_URL}/control",
                json={'action': action, 'params': params},
                timeout=5.0,
            )
            return response.json()
    except Exception as e:
        return {'error': str(e), 'hint': 'Is the Bokeh server running?'}


# -----------------------------------------------------------------------------
# MCP Server
# -----------------------------------------------------------------------------

app = Server("rog-browser")


@app.list_tools()
async def list_tools():
    return [
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
            name="highlight_points",
            description="Highlight specific points in the visualization",
            inputSchema={
                "type": "object",
                "properties": {
                    "indices": {"type": "array", "items": {"type": "integer"}, "description": "Point indices to highlight"},
                },
                "required": ["indices"],
            },
        ),
        Tool(
            name="reset_view",
            description="Reset the visualization to the default view",
            inputSchema={"type": "object", "properties": {}},
        ),
        Tool(
            name="set_cluster_level",
            description="Set the cluster level displayed (5, 12, 25, or 50)",
            inputSchema={
                "type": "object",
                "properties": {
                    "level": {"type": "integer", "enum": [5, 12, 25, 50], "description": "Cluster level to display"},
                },
                "required": ["level"],
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
            description="Clear all highlights from points and edges",
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
            name="toggle_dedup",
            description="Toggle chunk deduplication overlay (hides redundant same-doc chunks in same bucket)",
            inputSchema={
                "type": "object",
                "properties": {
                    "active": {"type": "boolean", "description": "Whether dedup mode should be active"},
                },
                "required": ["active"],
            },
        ),
        Tool(
            name="toggle_redundancy",
            description="Toggle chunk redundancy overlay (dims points by number of same-doc siblings in bucket)",
            inputSchema={
                "type": "object",
                "properties": {
                    "active": {"type": "boolean", "description": "Whether redundancy mode should be active"},
                },
                "required": ["active"],
            },
        ),
        Tool(
            name="toggle_doc_spread",
            description="Toggle document spread color mode (warm=bridge docs spanning many buckets, cool=focused docs)",
            inputSchema={
                "type": "object",
                "properties": {
                    "active": {"type": "boolean", "description": "Whether doc spread mode should be active"},
                },
                "required": ["active"],
            },
        ),
    ]


@app.call_tool()
async def call_tool(name: str, arguments: dict):
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
    elif name == "zoom_to_cluster":
        cluster_id = arguments["cluster_id"]
        level = arguments.get("level", 5)
        # Get cluster centroid and compute radius from actual point spread
        centroids = CACHE['cluster_result']['centroids'][level]
        if cluster_id < len(centroids):
            cx, cy = centroids[cluster_id]
            # Compute radius from cluster member spread
            labels = CACHE['cluster_result']['labels'][level]
            mask = labels == cluster_id
            coords = CACHE['coords_2d'][mask]
            dists = np.sqrt((coords[:, 0] - cx) ** 2 + (coords[:, 1] - cy) ** 2)
            radius = float(np.percentile(dists, 95)) * 1.3  # 95th pct + padding
            radius = max(radius, 0.5)  # Floor to avoid extreme zoom
            result = await control_viz("zoom_to", {"x": float(cx), "y": float(cy), "radius": radius})
        else:
            result = {"error": f"Invalid cluster_id {cluster_id}"}
    elif name == "zoom_to_point":
        index = arguments["index"]
        radius = arguments.get("radius", 5)
        if 0 <= index < len(CACHE['coords_2d']):
            x, y = CACHE['coords_2d'][index]
            result = await control_viz("zoom_to", {"x": float(x), "y": float(y), "radius": radius})
        else:
            result = {"error": f"Invalid index {index}"}
    elif name == "highlight_points":
        result = await control_viz("highlight", {"indices": arguments["indices"]})
    elif name == "highlight_bucket":
        bucket_id = arguments["bucket_id"]
        # Get sample points from this bucket to highlight
        members = get_bucket_members(bucket_id, limit=20)
        if members and 'error' not in members[0]:
            indices = [m['index'] for m in members]
            result = await control_viz("highlight", {"indices": indices})
            # Optionally zoom to bucket
            if arguments.get("zoom", True):
                centroids = CACHE['cluster_result']['centroids'][50]
                if bucket_id < len(centroids):
                    cx, cy = centroids[bucket_id]
                    labels = CACHE['cluster_result']['labels'][50]
                    mask = labels == bucket_id
                    coords = CACHE['coords_2d'][mask]
                    dists = np.sqrt((coords[:, 0] - cx) ** 2 + (coords[:, 1] - cy) ** 2)
                    radius = float(np.percentile(dists, 95)) * 1.3
                    radius = max(radius, 0.5)
                    await control_viz("zoom_to", {"x": float(cx), "y": float(cy), "radius": radius})
            # Include connection info in result
            connections = get_bucket_connections(bucket_id)
            result = {"highlight": result, "connections": connections}
        else:
            result = {"error": f"Could not get members for bucket {bucket_id}"}
    elif name == "clear_highlight":
        result = await control_viz("clear_highlight", {})
    elif name == "reset_view":
        result = await control_viz("reset", {})
    elif name == "set_cluster_level":
        result = await control_viz("set_level", {"level": arguments["level"]})
    elif name == "toggle_edges":
        result = await control_viz("toggle_edges", {"visible": arguments["visible"]})
    elif name == "toggle_dedup":
        result = await control_viz("set_dedup_mode", {"active": arguments["active"]})
    elif name == "toggle_redundancy":
        result = await control_viz("set_redundancy_mode", {"active": arguments["active"]})
    elif name == "toggle_doc_spread":
        result = await control_viz("set_spread_mode", {"active": arguments["active"]})
    else:
        result = {"error": f"Unknown tool: {name}"}

    return [TextContent(type="text", text=json.dumps(result, indent=2))]


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("cache_path", default="demo/wiki_simple_50k_rog_cache.pkl", nargs="?")
    parser.add_argument("--control-url", default="http://localhost:5008")
    args = parser.parse_args()

    global CONTROL_URL
    CONTROL_URL = args.control_url

    load_cache(args.cache_path)

    async with stdio_server() as (read_stream, write_stream):
        await app.run(read_stream, write_stream, app.create_initialization_options())


if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
