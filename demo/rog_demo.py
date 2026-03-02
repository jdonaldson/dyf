"""
ROG (Recursive Ontological Generation) Demo

Generates an interactive HTML browser for exploring hierarchical ontologies
built from embedding spaces using density-adaptive thresholding.

Features:
- 2D UMAP visualization of the embedding space
- Click nodes to highlight their parent/child connections
- Search and browse by most connected concepts

Usage:
    python rog_demo.py wiki_simple_50k.parquet --output rog_browser.html
    python rog_demo.py wiki_simple_50k.parquet --sample 10000  # Faster demo
"""

import argparse
import json
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import numpy as np
import polars as pl

from dyf import build_rog_ontology, ROGResult

try:
    from umap import UMAP
except ImportError:
    UMAP = None

try:
    import pandas as pd
    from datashader.bundling import hammer_bundle
    HAS_DATASHADER = True
except ImportError:
    HAS_DATASHADER = False


def load_embeddings(path: str, sample: Optional[int] = None, seed: int = 42) -> Tuple[np.ndarray, List[str], List[str]]:
    """Load embeddings from parquet file."""
    df = pl.read_parquet(path)

    if sample and sample < len(df):
        df = df.sample(n=sample, seed=seed)

    embeddings = np.array(df['embedding'].to_list(), dtype=np.float32)

    if 'title' in df.columns:
        titles = df['title'].to_list()
    elif 'text' in df.columns:
        titles = [t[:50] + '...' if len(t) > 50 else t for t in df['text'].to_list()]
    else:
        titles = [f"Item {i}" for i in range(len(embeddings))]

    texts = df['text'].to_list() if 'text' in df.columns else titles

    return embeddings, titles, texts


def project_to_2d(embeddings: np.ndarray, seed: int = 42) -> np.ndarray:
    """Project embeddings to 2D using UMAP."""
    if UMAP is None:
        raise ImportError("umap-learn required: pip install umap-learn")

    print("  Projecting to 2D with UMAP...")
    reducer = UMAP(n_components=2, random_state=seed, n_neighbors=15, min_dist=0.1)
    coords_2d = reducer.fit_transform(embeddings)
    return coords_2d


def compute_bundled_edges(
    coords_2d: np.ndarray,
    parents_map: Dict[int, List[int]],
    node_ids: set,
) -> Dict[str, List[List[float]]]:
    """Compute hammer-bundled edge paths for all parent-child relationships.

    Returns dict mapping "child_id-parent_id" to [x_coords, y_coords] path.
    """
    if not HAS_DATASHADER:
        print("  Warning: datashader not available, using straight edges")
        return {}

    print("  Computing hammer edge bundling...")

    # Build node index mapping (only nodes in ontology)
    node_list = sorted(node_ids)
    idx_to_local = {idx: i for i, idx in enumerate(node_list)}

    # Create nodes DataFrame
    nodes_data = [{'x': coords_2d[idx, 0], 'y': coords_2d[idx, 1]} for idx in node_list]
    nodes_df = pd.DataFrame(nodes_data)

    # Collect all edges
    all_edges = []
    edge_keys = []

    for child_idx, parent_list in parents_map.items():
        if child_idx not in idx_to_local:
            continue
        for parent_idx in parent_list:
            if parent_idx not in idx_to_local:
                continue
            all_edges.append({
                'source': idx_to_local[child_idx],
                'target': idx_to_local[parent_idx]
            })
            edge_keys.append(f"{child_idx}-{parent_idx}")

    if not all_edges:
        return {}

    print(f"    Bundling {len(all_edges):,} edges...")
    edges_df = pd.DataFrame(all_edges)

    # Run hammer bundling
    bundled = hammer_bundle(nodes_df, edges_df)

    # Parse bundled paths
    edge_paths = {}
    current_x = []
    current_y = []
    edge_idx = 0

    for x, y in zip(bundled['x'], bundled['y']):
        if pd.isna(x):
            if current_x and edge_idx < len(edge_keys):
                edge_paths[edge_keys[edge_idx]] = [current_x.copy(), current_y.copy()]
                edge_idx += 1
            current_x = []
            current_y = []
        else:
            current_x.append(float(x))
            current_y.append(float(y))

    # Don't forget last path
    if current_x and edge_idx < len(edge_keys):
        edge_paths[edge_keys[edge_idx]] = [current_x, current_y]

    print(f"    Bundled {len(edge_paths):,} edge paths")
    return edge_paths


def build_node_data(
    rog_result: ROGResult,
    titles: List[str],
    coords_2d: np.ndarray,
) -> Tuple[List[Dict], List[Dict], Dict[str, List[List[float]]]]:
    """Convert ROG result to node data for visualization.

    Returns:
        (nodes, top_hubs, edge_paths) where:
        - nodes: list of node dicts with coordinates and relationships
        - top_hubs: most connected concepts for sidebar
        - edge_paths: bundled edge paths keyed by "child-parent"
    """
    ontology = rog_result.ontology
    n_nodes = len(titles)

    # Get layer assignments
    layer_map = {}
    for layer in rog_result.layers:
        for idx in layer.node_indices:
            layer_map[idx] = layer.depth

    # Build parent/child relationships
    children_map = defaultdict(list)
    parents_map = defaultdict(list)

    for child_idx in range(n_nodes):
        parents = ontology.get_parents(child_idx)
        for parent_idx in parents:
            children_map[parent_idx].append(child_idx)
            parents_map[child_idx].append(parent_idx)

    # Identify convergence (many parents) and divergence (many children) points
    n_parents = [len(parents_map.get(i, [])) for i in range(n_nodes)]
    n_children = [len(children_map.get(i, [])) for i in range(n_nodes)]

    parent_threshold = np.percentile([p for p in n_parents if p > 0], 90) if any(n_parents) else 1
    child_threshold = np.percentile([c for c in n_children if c > 0], 90) if any(n_children) else 1

    nodes = []
    for i in range(n_nodes):
        layer = layer_map.get(i, -1)
        if layer < 0:
            continue  # Skip excluded nodes

        total_connections = n_parents[i] + n_children[i]
        nodes.append({
            'id': int(i),
            'title': titles[i],
            'layer': int(layer),
            'x': float(coords_2d[i, 0]),
            'y': float(coords_2d[i, 1]),
            'n_parents': int(n_parents[i]),
            'n_children': int(n_children[i]),
            'total_connections': int(total_connections),
            'parents': [int(p) for p in parents_map.get(i, [])],
            'children': [int(c) for c in children_map.get(i, [])],
            'is_convergence': bool(n_parents[i] >= parent_threshold),
            'is_divergence': bool(n_children[i] >= child_threshold),
        })

    # Find top hubs by total connections, aggregated by title
    title_stats = defaultdict(lambda: {'total': 0, 'best_node': None})
    for node in nodes:
        title = node['title']
        title_stats[title]['total'] += node['total_connections']
        if title_stats[title]['best_node'] is None or node['total_connections'] > title_stats[title]['best_node']['total_connections']:
            title_stats[title]['best_node'] = node

    # Create hub entries with aggregated totals
    top_hubs = []
    for title, stats in title_stats.items():
        hub = stats['best_node'].copy()
        hub['total_connections'] = stats['total']  # Use aggregated total
        top_hubs.append(hub)

    top_hubs = sorted(top_hubs, key=lambda x: -x['total_connections'])[:30]

    # Compute bundled edges
    node_ids = {n['id'] for n in nodes}
    edge_paths = compute_bundled_edges(coords_2d, dict(parents_map), node_ids)

    return nodes, top_hubs, edge_paths


def generate_html(
    nodes: List[Dict],
    top_hubs: List[Dict],
    edge_paths: Dict[str, List[List[float]]],
    rog_result: ROGResult,
    title: str = "ROG Ontology Browser"
) -> str:
    """Generate interactive HTML browser with UMAP visualization."""

    # Layer stats
    layer_stats = []
    for layer in rog_result.layers:
        layer_stats.append({
            'depth': layer.depth,
            'threshold': f"{layer.similarity_threshold:.2f}",
            'nodes': layer.n_nodes,
            'edges': layer.n_edges,
            'coverage': f"{layer.coverage*100:.1f}%"
        })

    # Layer colors for visualization
    layer_colors = ['#e94560', '#f9a826', '#4ecca3', '#45b7d1', '#bb8fce']

    html = f'''<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>{title}</title>
    <style>
        * {{
            box-sizing: border-box;
            margin: 0;
            padding: 0;
        }}
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background: #1a1a2e;
            color: #eee;
            height: 100vh;
            overflow: hidden;
        }}
        .container {{
            display: flex;
            height: 100vh;
        }}
        .sidebar {{
            width: 320px;
            background: #16213e;
            padding: 15px;
            overflow-y: auto;
            border-right: 1px solid #0f3460;
            flex-shrink: 0;
        }}
        .main {{
            flex: 1;
            display: flex;
            flex-direction: column;
            overflow: hidden;
            position: relative;
        }}
        .search-box {{
            padding: 10px 15px;
            background: #16213e;
            border-bottom: 1px solid #0f3460;
            position: relative;
            z-index: 100;
        }}
        .search-box input {{
            width: 100%;
            padding: 10px 14px;
            border: 1px solid #0f3460;
            border-radius: 8px;
            background: #1a1a2e;
            color: #eee;
            font-size: 14px;
        }}
        .search-box input:focus {{
            outline: none;
            border-color: #e94560;
        }}
        .search-results {{
            position: absolute;
            top: 100%;
            left: 15px;
            right: 15px;
            max-height: 300px;
            overflow-y: auto;
            background: #16213e;
            border: 1px solid #0f3460;
            border-radius: 8px;
            z-index: 101;
        }}
        .search-result {{
            padding: 8px 12px;
            cursor: pointer;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }}
        .search-result:hover {{
            background: #0f3460;
        }}
        .canvas-container {{
            flex: 1;
            position: relative;
            overflow: hidden;
        }}
        #viz-canvas {{
            position: absolute;
            top: 0;
            left: 0;
            cursor: grab;
        }}
        #viz-canvas:active {{
            cursor: grabbing;
        }}
        .tooltip {{
            position: absolute;
            background: rgba(22, 33, 62, 0.95);
            border: 1px solid #0f3460;
            border-radius: 6px;
            padding: 8px 12px;
            font-size: 13px;
            pointer-events: none;
            z-index: 1000;
            max-width: 250px;
            display: none;
        }}
        .tooltip .title {{
            font-weight: bold;
            color: #e94560;
            margin-bottom: 4px;
        }}
        .tooltip .stats {{
            color: #888;
            font-size: 11px;
        }}
        .layer-badge {{
            font-size: 11px;
            padding: 2px 6px;
            border-radius: 10px;
            background: #0f3460;
        }}
        .layer-0 {{ background: #e94560; }}
        .layer-1 {{ background: #f9a826; color: #000; }}
        .layer-2 {{ background: #4ecca3; color: #000; }}
        .layer-3 {{ background: #45b7d1; color: #000; }}
        .layer-4 {{ background: #bb8fce; }}
        h2 {{
            font-size: 16px;
            margin-bottom: 10px;
            color: #e94560;
        }}
        h3 {{
            font-size: 12px;
            margin: 15px 0 8px 0;
            color: #888;
            text-transform: uppercase;
            letter-spacing: 1px;
        }}
        .rog-info {{
            background: #0f3460;
            padding: 12px;
            border-radius: 8px;
            margin-bottom: 15px;
            font-size: 13px;
        }}
        .rog-layers {{
            margin-top: 8px;
        }}
        .rog-layer {{
            display: flex;
            justify-content: space-between;
            padding: 4px 0;
            border-bottom: 1px solid #1a1a2e;
            font-size: 12px;
        }}
        .node-list {{
            display: flex;
            flex-wrap: wrap;
            gap: 6px;
            margin: 8px 0;
        }}
        .node-chip {{
            padding: 5px 10px;
            background: #0f3460;
            border-radius: 12px;
            cursor: pointer;
            font-size: 12px;
            transition: all 0.2s;
            border: 1px solid transparent;
        }}
        .node-chip:hover {{
            background: #e94560;
        }}
        .node-chip.active {{
            background: #e94560;
            border-color: #fff;
        }}
        .node-detail {{
            background: #0f3460;
            padding: 12px;
            border-radius: 8px;
            margin-top: 15px;
            display: none;
        }}
        .node-detail.visible {{
            display: block;
        }}
        .node-detail h3 {{
            color: #e94560;
            margin: 0 0 8px 0;
            text-transform: none;
            font-size: 14px;
        }}
        .node-detail .meta {{
            font-size: 12px;
            color: #888;
            margin-bottom: 10px;
        }}
        .legend {{
            display: flex;
            gap: 8px;
            flex-wrap: wrap;
            margin-top: 10px;
            font-size: 11px;
        }}
        .legend-item {{
            display: flex;
            align-items: center;
            gap: 4px;
        }}
        .legend-dot {{
            width: 10px;
            height: 10px;
            border-radius: 50%;
        }}
        .controls {{
            position: absolute;
            bottom: 15px;
            right: 15px;
            display: flex;
            gap: 8px;
            z-index: 100;
        }}
        .control-btn {{
            background: #16213e;
            border: 1px solid #0f3460;
            color: #eee;
            padding: 8px 12px;
            border-radius: 6px;
            cursor: pointer;
            font-size: 14px;
        }}
        .control-btn:hover {{
            background: #0f3460;
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="sidebar">
            <h2>{title}</h2>
            <div class="rog-info">
                <div><strong>Coverage:</strong> {rog_result.total_coverage*100:.1f}%</div>
                <div><strong>Nodes:</strong> {len(nodes):,}</div>
                <div class="rog-layers">
                    {''.join(f'<div class="rog-layer"><span class="layer-badge layer-{l["depth"]}">L{l["depth"]}</span><span>{l["nodes"]:,} @ {l["threshold"]}</span></div>' for l in layer_stats)}
                </div>
                <div class="legend">
                    {''.join(f'<span class="legend-item"><span class="legend-dot" style="background:{layer_colors[i]}"></span>L{i}</span>' for i in range(len(layer_stats)))}
                </div>
            </div>

            <h3>Most Connected</h3>
            <div class="node-list" id="top-hubs-list"></div>

            <div class="node-detail" id="node-detail">
                <h3 id="detail-title"></h3>
                <div class="meta" id="detail-meta"></div>
                <h3 style="margin-top:10px">Parents</h3>
                <div class="node-list" id="detail-parents"></div>
                <h3 style="margin-top:10px">Children</h3>
                <div class="node-list" id="detail-children"></div>
            </div>
        </div>

        <div class="main">
            <div class="search-box">
                <input type="text" id="search" placeholder="Search nodes..." autocomplete="off">
                <div class="search-results" id="search-results"></div>
            </div>

            <div class="canvas-container" id="canvas-container">
                <canvas id="viz-canvas"></canvas>
                <div class="tooltip" id="tooltip"></div>
            </div>

            <div class="controls">
                <button class="control-btn" onclick="resetView()">Reset View</button>
                <button class="control-btn" onclick="toggleEdges()">Toggle Edges</button>
            </div>
        </div>
    </div>

    <script>
        const nodes = {json.dumps(nodes)};
        const topHubs = {json.dumps(top_hubs)};
        const layerColors = {json.dumps(layer_colors)};
        const edgePaths = {json.dumps(edge_paths)};

        const nodeMap = {{}};
        nodes.forEach(n => nodeMap[n.id] = n);

        // Canvas setup
        const canvas = document.getElementById('viz-canvas');
        const ctx = canvas.getContext('2d');
        const container = document.getElementById('canvas-container');
        const tooltip = document.getElementById('tooltip');

        let width, height;
        let transform = {{ x: 0, y: 0, scale: 1 }};
        let isDragging = false;
        let dragStart = {{ x: 0, y: 0 }};
        let selectedNode = null;
        let hoveredNode = null;
        let showEdges = true;

        // Compute data bounds
        const xExtent = [Math.min(...nodes.map(n => n.x)), Math.max(...nodes.map(n => n.x))];
        const yExtent = [Math.min(...nodes.map(n => n.y)), Math.max(...nodes.map(n => n.y))];
        const dataWidth = xExtent[1] - xExtent[0];
        const dataHeight = yExtent[1] - yExtent[0];

        function resize() {{
            width = container.clientWidth;
            height = container.clientHeight;
            canvas.width = width * window.devicePixelRatio;
            canvas.height = height * window.devicePixelRatio;
            canvas.style.width = width + 'px';
            canvas.style.height = height + 'px';
            ctx.scale(window.devicePixelRatio, window.devicePixelRatio);
            resetView();
        }}

        function resetView() {{
            const padding = 50;
            const scaleX = (width - padding * 2) / dataWidth;
            const scaleY = (height - padding * 2) / dataHeight;
            transform.scale = Math.min(scaleX, scaleY);
            transform.x = width / 2 - (xExtent[0] + dataWidth / 2) * transform.scale;
            transform.y = height / 2 - (yExtent[0] + dataHeight / 2) * transform.scale;
            render();
        }}

        function toScreen(x, y) {{
            return {{
                x: x * transform.scale + transform.x,
                y: y * transform.scale + transform.y
            }};
        }}

        function toData(x, y) {{
            return {{
                x: (x - transform.x) / transform.scale,
                y: (y - transform.y) / transform.scale
            }};
        }}

        function render() {{
            ctx.clearRect(0, 0, width, height);

            // Draw edges for selected node
            if (selectedNode && showEdges) {{
                ctx.globalAlpha = 0.7;

                // Helper to draw bundled or straight edge
                function drawEdge(fromId, toId, color) {{
                    const key = `${{fromId}}-${{toId}}`;
                    const path = edgePaths[key];

                    ctx.strokeStyle = color;
                    ctx.lineWidth = 2;
                    ctx.beginPath();

                    if (path && path[0].length > 1) {{
                        // Draw bundled path
                        const xs = path[0];
                        const ys = path[1];
                        const p0 = toScreen(xs[0], ys[0]);
                        ctx.moveTo(p0.x, p0.y);
                        for (let i = 1; i < xs.length; i++) {{
                            const p = toScreen(xs[i], ys[i]);
                            ctx.lineTo(p.x, p.y);
                        }}
                    }} else {{
                        // Fallback to straight line
                        const from = nodeMap[fromId];
                        const to = nodeMap[toId];
                        if (from && to) {{
                            const p1 = toScreen(from.x, from.y);
                            const p2 = toScreen(to.x, to.y);
                            ctx.moveTo(p1.x, p1.y);
                            ctx.lineTo(p2.x, p2.y);
                        }}
                    }}
                    ctx.stroke();
                }}

                // Parent edges (red) - edge key is "child-parent" = "selected-parent"
                selectedNode.parents.forEach(pid => {{
                    drawEdge(selectedNode.id, pid, '#e94560');
                }});

                // Child edges (green) - edge key is "child-parent" = "child-selected"
                selectedNode.children.forEach(cid => {{
                    drawEdge(cid, selectedNode.id, '#4ecca3');
                }});

                ctx.globalAlpha = 1;
            }}

            // Draw all nodes
            const baseSize = Math.max(2, 3 * transform.scale / 50);

            nodes.forEach(node => {{
                const pos = toScreen(node.x, node.y);

                // Skip if off screen
                if (pos.x < -10 || pos.x > width + 10 || pos.y < -10 || pos.y > height + 10) return;

                const isSelected = selectedNode && selectedNode.id === node.id;
                const isConnected = selectedNode && (
                    selectedNode.parents.includes(node.id) ||
                    selectedNode.children.includes(node.id)
                );
                const isHovered = hoveredNode && hoveredNode.id === node.id;

                let size = baseSize;
                let alpha = 0.7;

                if (isSelected) {{
                    size = baseSize * 2.5;
                    alpha = 1;
                }} else if (isConnected) {{
                    size = baseSize * 1.8;
                    alpha = 1;
                }} else if (selectedNode) {{
                    alpha = 0.2;
                }}

                if (isHovered && !isSelected) {{
                    size = baseSize * 1.5;
                    alpha = 1;
                }}

                ctx.globalAlpha = alpha;
                ctx.fillStyle = layerColors[node.layer] || '#888';
                ctx.beginPath();
                ctx.arc(pos.x, pos.y, size, 0, Math.PI * 2);
                ctx.fill();

                // Draw ring for connected nodes
                if (isConnected) {{
                    ctx.strokeStyle = selectedNode.parents.includes(node.id) ? '#e94560' : '#4ecca3';
                    ctx.lineWidth = 2;
                    ctx.beginPath();
                    ctx.arc(pos.x, pos.y, size + 3, 0, Math.PI * 2);
                    ctx.stroke();
                }}

                // Draw selection ring
                if (isSelected) {{
                    ctx.strokeStyle = '#fff';
                    ctx.lineWidth = 2;
                    ctx.beginPath();
                    ctx.arc(pos.x, pos.y, size + 4, 0, Math.PI * 2);
                    ctx.stroke();
                }}
            }});

            ctx.globalAlpha = 1;
        }}

        function findNodeAt(x, y) {{
            const data = toData(x, y);
            const threshold = 15 / transform.scale;

            let closest = null;
            let closestDist = threshold;

            nodes.forEach(node => {{
                const dx = node.x - data.x;
                const dy = node.y - data.y;
                const dist = Math.sqrt(dx * dx + dy * dy);
                if (dist < closestDist) {{
                    closestDist = dist;
                    closest = node;
                }}
            }});

            return closest;
        }}

        function selectNode(node) {{
            selectedNode = node;

            // Update sidebar
            const detail = document.getElementById('node-detail');
            if (node) {{
                document.getElementById('detail-title').textContent = node.title;
                document.getElementById('detail-meta').innerHTML =
                    `<span class="layer-badge layer-${{node.layer}}">Layer ${{node.layer}}</span> ` +
                    `${{node.n_parents}} parents, ${{node.n_children}} children`;

                const parents = node.parents.map(p => nodeMap[p]).filter(Boolean);
                const children = node.children.map(c => nodeMap[c]).filter(Boolean);

                document.getElementById('detail-parents').innerHTML = parents.length > 0
                    ? parents.slice(0, 20).map(p =>
                        `<span class="node-chip" onclick="selectNodeById(${{p.id}})">${{p.title}}</span>`
                      ).join('') + (parents.length > 20 ? `<span class="node-chip">+${{parents.length - 20}} more</span>` : '')
                    : '<span style="color:#666">None</span>';

                document.getElementById('detail-children').innerHTML = children.length > 0
                    ? children.slice(0, 20).map(c =>
                        `<span class="node-chip" onclick="selectNodeById(${{c.id}})">${{c.title}}</span>`
                      ).join('') + (children.length > 20 ? `<span class="node-chip">+${{children.length - 20}} more</span>` : '')
                    : '<span style="color:#666">None</span>';

                detail.classList.add('visible');

                // Update active state on hub chips
                document.querySelectorAll('#top-hubs-list .node-chip').forEach(chip => {{
                    chip.classList.remove('active');
                }});
            }} else {{
                detail.classList.remove('visible');
            }}

            render();
        }}

        function selectNodeById(id) {{
            const node = nodeMap[id];
            if (node) {{
                selectNode(node);
                // Center on node
                const pos = toScreen(node.x, node.y);
                transform.x += width / 2 - pos.x;
                transform.y += height / 2 - pos.y;
                render();
            }}
        }}

        function toggleEdges() {{
            showEdges = !showEdges;
            render();
        }}

        // Mouse handlers
        canvas.addEventListener('mousedown', (e) => {{
            isDragging = true;
            dragStart = {{ x: e.clientX, y: e.clientY }};
            canvas.style.cursor = 'grabbing';
        }});

        canvas.addEventListener('mousemove', (e) => {{
            const rect = canvas.getBoundingClientRect();
            const x = e.clientX - rect.left;
            const y = e.clientY - rect.top;

            if (isDragging) {{
                transform.x += e.clientX - dragStart.x;
                transform.y += e.clientY - dragStart.y;
                dragStart = {{ x: e.clientX, y: e.clientY }};
                render();
            }} else {{
                const node = findNodeAt(x, y);
                if (node !== hoveredNode) {{
                    hoveredNode = node;
                    render();

                    if (node) {{
                        tooltip.innerHTML = `
                            <div class="title">${{node.title}}</div>
                            <div class="stats">Layer ${{node.layer}} | ${{node.n_parents}} parents, ${{node.n_children}} children</div>
                        `;
                        tooltip.style.display = 'block';
                        tooltip.style.left = (e.clientX - rect.left + 15) + 'px';
                        tooltip.style.top = (e.clientY - rect.top + 15) + 'px';
                    }} else {{
                        tooltip.style.display = 'none';
                    }}
                }} else if (node) {{
                    tooltip.style.left = (e.clientX - rect.left + 15) + 'px';
                    tooltip.style.top = (e.clientY - rect.top + 15) + 'px';
                }}
            }}
        }});

        canvas.addEventListener('mouseup', (e) => {{
            if (isDragging) {{
                const dx = e.clientX - dragStart.x;
                const dy = e.clientY - dragStart.y;

                // If minimal movement, treat as click
                if (Math.abs(dx) < 5 && Math.abs(dy) < 5) {{
                    const rect = canvas.getBoundingClientRect();
                    const x = e.clientX - rect.left;
                    const y = e.clientY - rect.top;
                    const node = findNodeAt(x, y);
                    selectNode(node);
                }}
            }}
            isDragging = false;
            canvas.style.cursor = 'grab';
        }});

        canvas.addEventListener('mouseleave', () => {{
            isDragging = false;
            hoveredNode = null;
            tooltip.style.display = 'none';
            canvas.style.cursor = 'grab';
            render();
        }});

        canvas.addEventListener('wheel', (e) => {{
            e.preventDefault();
            const rect = canvas.getBoundingClientRect();
            const x = e.clientX - rect.left;
            const y = e.clientY - rect.top;

            const zoom = e.deltaY > 0 ? 0.9 : 1.1;
            const newScale = transform.scale * zoom;

            // Limit zoom
            if (newScale < 0.1 || newScale > 100) return;

            // Zoom toward mouse position
            transform.x = x - (x - transform.x) * zoom;
            transform.y = y - (y - transform.y) * zoom;
            transform.scale = newScale;

            render();
        }}, {{ passive: false }});

        // Populate top hubs
        document.getElementById('top-hubs-list').innerHTML = topHubs.map(n =>
            `<span class="node-chip" onclick="selectNodeById(${{n.id}})" title="${{n.total_connections}} connections">${{n.title}} <span style="opacity:0.5;font-size:10px">(${{n.total_connections}})</span></span>`
        ).join('');

        // Search functionality
        const searchInput = document.getElementById('search');
        const searchResults = document.getElementById('search-results');

        searchInput.addEventListener('input', (e) => {{
            const query = e.target.value.toLowerCase();
            if (query.length < 2) {{
                searchResults.innerHTML = '';
                return;
            }}

            const matches = nodes.filter(n => n.title.toLowerCase().includes(query)).slice(0, 15);
            searchResults.innerHTML = matches.map(n => `
                <div class="search-result" onclick="selectNodeById(${{n.id}}); searchInput.value = ''; searchResults.innerHTML = '';">
                    <span>${{n.title}}</span>
                    <span class="layer-badge layer-${{n.layer}}">L${{n.layer}}</span>
                </div>
            `).join('');
        }});

        searchInput.addEventListener('blur', () => {{
            setTimeout(() => searchResults.innerHTML = '', 200);
        }});

        // Initialize
        window.addEventListener('resize', resize);
        resize();
    </script>
</body>
</html>'''

    return html


def main():
    parser = argparse.ArgumentParser(description='Generate ROG ontology browser')
    parser.add_argument('input', help='Path to embeddings parquet file')
    parser.add_argument('--output', '-o', default='rog_browser.html', help='Output HTML file')
    parser.add_argument('--sample', type=int, help='Sample N points for faster demo')
    parser.add_argument('--initial-threshold', type=float, default=0.55)
    parser.add_argument('--min-threshold', type=float, default=0.35)
    parser.add_argument('--target-coverage', type=float, default=0.95)
    parser.add_argument('--title', default='ROG Ontology Browser')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    print(f"Loading embeddings from {args.input}...")
    embeddings, titles, texts = load_embeddings(args.input, sample=args.sample, seed=args.seed)
    print(f"  Loaded {len(embeddings):,} embeddings, dim={embeddings.shape[1]}")

    # Normalize
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    embeddings = embeddings / np.where(norms > 0, norms, 1)

    # Project to 2D
    coords_2d = project_to_2d(embeddings, seed=args.seed)

    print("Building ROG ontology...")
    rog_result = build_rog_ontology(
        embeddings,
        initial_threshold=args.initial_threshold,
        min_threshold=args.min_threshold,
        target_coverage=args.target_coverage,
        verbose=True,
    )
    print(rog_result.summary())

    print("Building node data...")
    nodes, top_hubs, edge_paths = build_node_data(rog_result, titles, coords_2d)
    print(f"  {len(nodes)} nodes in ontology")
    print(f"  {len(top_hubs)} top hub concepts")
    print(f"  {len(edge_paths)} bundled edge paths")

    print("Generating HTML...")
    html = generate_html(nodes, top_hubs, edge_paths, rog_result, title=args.title)

    output_path = Path(args.output)
    output_path.write_text(html)
    print(f"Saved to {output_path}")


if __name__ == '__main__':
    main()
