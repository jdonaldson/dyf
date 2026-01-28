"""
MCP Server for ROG Browser.

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

def search_points(query: str, limit: int = 20) -> list[dict]:
    """Search points by title (case-insensitive substring match)."""
    query_lower = query.lower()
    results = []
    for i, title in enumerate(CACHE['titles']):
        if query_lower in title.lower():
            results.append({
                'index': i,
                'title': title,
                'x': float(CACHE['coords_2d'][i, 0]),
                'y': float(CACHE['coords_2d'][i, 1]),
                'cluster_8': int(CACHE['cluster_result']['labels'][8][i]),
                'cluster_15': int(CACHE['cluster_result']['labels'][15][i]),
                'cluster_30': int(CACHE['cluster_result']['labels'][30][i]),
            })
            if len(results) >= limit:
                break
    return results


def get_cluster_info(level: int = 8) -> list[dict]:
    """Get cluster information for a given level."""
    if level not in [8, 15, 30]:
        return [{'error': f'Invalid level {level}. Must be 8, 15, or 30.'}]

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


def get_cluster_members(cluster_id: int, level: int = 8, limit: int = 50) -> list[dict]:
    """Get members of a specific cluster."""
    if level not in [8, 15, 30]:
        return [{'error': f'Invalid level {level}. Must be 8, 15, or 30.'}]

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
            description="Get information about clusters at a given level (8, 15, or 30 clusters)",
            inputSchema={
                "type": "object",
                "properties": {
                    "level": {"type": "integer", "enum": [8, 15, 30], "description": "Cluster level", "default": 8},
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
                    "level": {"type": "integer", "enum": [8, 15, 30], "description": "Cluster level", "default": 8},
                    "limit": {"type": "integer", "description": "Max results (default 50)", "default": 50},
                },
                "required": ["cluster_id"],
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
                    "level": {"type": "integer", "enum": [8, 15, 30], "description": "Cluster level", "default": 8},
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
            description="Set the cluster level displayed (8, 15, or 30)",
            inputSchema={
                "type": "object",
                "properties": {
                    "level": {"type": "integer", "enum": [8, 15, 30], "description": "Cluster level to display"},
                },
                "required": ["level"],
            },
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
    ]


@app.call_tool()
async def call_tool(name: str, arguments: dict):
    if name == "search_points":
        result = search_points(arguments["query"], arguments.get("limit", 20))
    elif name == "get_cluster_info":
        result = get_cluster_info(arguments.get("level", 8))
    elif name == "get_cluster_members":
        result = get_cluster_members(
            arguments["cluster_id"],
            arguments.get("level", 8),
            arguments.get("limit", 50),
        )
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
        level = arguments.get("level", 8)
        # Get cluster centroid and zoom there
        centroids = CACHE['cluster_result']['centroids'][level]
        if cluster_id < len(centroids):
            cx, cy = centroids[cluster_id]
            result = await control_viz("zoom_to", {"x": float(cx), "y": float(cy), "radius": 10})
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
    elif name == "reset_view":
        result = await control_viz("reset", {})
    elif name == "set_cluster_level":
        result = await control_viz("set_level", {"level": arguments["level"]})
    elif name == "toggle_edges":
        result = await control_viz("toggle_edges", {"visible": arguments["visible"]})
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
