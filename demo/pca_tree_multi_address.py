"""
PCA Tree Multi-Address Visualization — Show semantically polysemous bridge
points in 2D UMAP space.

Multi-address points are boundary at multiple PCA tree depths, meaning they
straddle concepts at several levels of the hierarchy. These are high-quality
semantic connectors between domains.

Usage:
    python demo/pca_tree_multi_address.py demo/wiki_simple_50k.parquet [--sample 10000] [--max-depth 8] [--margin-pct 0.10]
"""

import argparse
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl
import plotly.graph_objects as go
from datashader.bundling import hammer_bundle

# Ensure demo/ is importable
sys.path.insert(0, str(Path(__file__).parent))

from pca_tree_knn_umap import (
    _build_pca_tree_with_margins,
    extract_multi_address,
    cut_tree_to_labels,
    build_colors_golden_ratio,
    labels_to_names_centroids,
    run_umap,
)


# ---------------------------------------------------------------------------
# Shadow path helper
# ---------------------------------------------------------------------------

def _find_shadow_centroid(tree, point_idx, boundary_depth, coords):
    """Find the 2D centroid of the shadow leaf for a multi-address edge.

    Walks the tree to the split at boundary_depth, flips to the opposite
    branch, then follows the larger child greedily to a shadow leaf.
    Returns the 2D centroid of that shadow leaf, or None.
    """
    def _walk(node, current_depth, flipped):
        if node['left'] is None and node['right'] is None:
            return coords[node['indices']].mean(axis=0)

        if node['point_margin_map'] is None:
            return coords[node['indices']].mean(axis=0)

        left_pts = set(node['left']['indices'].tolist()) if node['left'] else set()
        point_is_left = point_idx in left_pts

        if current_depth == boundary_depth and not flipped:
            if point_is_left and node['right'] is not None:
                return _walk(node['right'], current_depth + 1, True)
            elif not point_is_left and node['left'] is not None:
                return _walk(node['left'], current_depth + 1, True)
            return None
        elif not flipped:
            if point_is_left and node['left'] is not None:
                return _walk(node['left'], current_depth + 1, False)
            elif node['right'] is not None:
                return _walk(node['right'], current_depth + 1, False)
            return None
        else:
            left_n = len(node['left']['indices']) if node['left'] else 0
            right_n = len(node['right']['indices']) if node['right'] else 0
            if left_n >= right_n and node['left'] is not None:
                return _walk(node['left'], current_depth + 1, True)
            elif node['right'] is not None:
                return _walk(node['right'], current_depth + 1, True)
            return None

    return _walk(tree, 0, False)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="PCA Tree Multi-Address Visualization")
    parser.add_argument("parquet_path", help="Path to embeddings parquet")
    parser.add_argument("--sample", type=int, default=10000)
    parser.add_argument("--max-depth", type=int, default=8)
    parser.add_argument("--margin-pct", type=float, default=0.10)
    args = parser.parse_args()

    outdir = Path(args.parquet_path).parent

    # ── Load data ─────────────────────────────────────────────────────────
    print(f"Loading {args.parquet_path}...")
    df = pl.read_parquet(args.parquet_path)
    if args.sample and args.sample < len(df):
        df = df.sample(args.sample, seed=42)

    titles = df["title"].to_list()
    embeddings = np.array(df["embedding"].to_list())
    n_points = len(titles)
    print(f"  {n_points} points, {embeddings.shape[1]}d embeddings")

    # ── Build PCA tree ────────────────────────────────────────────────────
    print(f"\nBuilding PCA tree (depth={args.max_depth})...")
    t0 = time.time()
    all_idx = np.arange(n_points)
    tree = _build_pca_tree_with_margins(embeddings, all_idx, args.max_depth)
    tree_time = time.time() - t0
    print(f"  Tree built in {tree_time:.2f}s")

    # ── Extract multi-address ─────────────────────────────────────────────
    t0 = time.time()
    ma = extract_multi_address(tree, margin_pct=args.margin_pct)
    ma_time = time.time() - t0

    boundary_count = ma['boundary_count']
    boundary_depths = ma['boundary_depths']

    # Count leaves
    def _count_leaves(node):
        if node['left'] is None and node['right'] is None:
            return 1
        count = 0
        if node['left'] is not None:
            count += _count_leaves(node['left'])
        if node['right'] is not None:
            count += _count_leaves(node['right'])
        return count

    n_leaves = _count_leaves(tree)

    # ── Console output ────────────────────────────────────────────────────
    print(f"\nMulti-address analysis (depth={args.max_depth}, "
          f"margin_pct={args.margin_pct}):")
    print(f"  {n_points} points, {n_leaves} leaves")
    print(f"  Tree build: {tree_time:.2f}s")
    print(f"  Multi-address extraction: {ma_time:.2f}s")

    # Distribution
    for threshold in [0, 1, 2, 3, 4]:
        count = int(np.sum(boundary_count >= threshold))
        pct = 100 * count / n_points
        if threshold == 0:
            exact_0 = int(np.sum(boundary_count == 0))
            print(f"\n  Boundary at 0 depths: {exact_0:>5d} ({100*exact_0/n_points:.1f}%)"
                  f"  — single-address")
        elif threshold == 1:
            exact_1 = int(np.sum(boundary_count == 1))
            print(f"  Boundary at 1 depth:  {exact_1:>5d} ({100*exact_1/n_points:.1f}%)")
        elif threshold == 2:
            print(f"  Boundary at 2+ depths: {count:>4d} ({pct:.1f}%)"
                  f"  — multi-address bridges")
        elif threshold == 3:
            print(f"  Boundary at 3+ depths: {count:>4d} ({pct:.1f}%)"
                  f"  — strong bridges")
        elif threshold == 4:
            print(f"  Boundary at 4+ depths: {count:>4d} ({pct:.1f}%)"
                  f"  — super bridges")

    # Top multi-address points
    top_indices = np.argsort(-boundary_count)[:15]
    print(f"\n  Top multi-address points:")
    for idx in top_indices:
        if boundary_count[idx] < 2:
            break
        depths = sorted(boundary_depths.get(int(idx), []))
        print(f"    '{titles[idx]}' — {boundary_count[idx]} depths {depths}")

    # ── Run UMAP ──────────────────────────────────────────────────────────
    print(f"\nRunning UMAP for 2D projection...")
    t0 = time.time()
    coords = run_umap(embeddings, n_neighbors=15)
    print(f"  UMAP done in {time.time() - t0:.1f}s")

    # ── Cut tree for cluster coloring ─────────────────────────────────────
    n_clusters = 25
    labels = cut_tree_to_labels(tree, args.max_depth, n_points, n_clusters)
    micro = cut_tree_to_labels(tree, args.max_depth, n_points, 200)
    cluster_names, centroids = labels_to_names_centroids(coords, titles, labels)

    # ── Build Plotly figure ───────────────────────────────────────────────
    print(f"\nBuilding visualization...")
    fig = go.Figure()

    # Layer 1: Base scatter — all points, colored by cluster
    colors = build_colors_golden_ratio(labels, micro)
    fig.add_trace(go.Scattergl(
        x=coords[:, 0], y=coords[:, 1],
        mode="markers",
        marker=dict(size=3, color=colors, opacity=0.3),
        text=[f"{titles[i]}<br>Cluster: {cluster_names[labels[i]]}"
              for i in range(n_points)],
        hoverinfo="text",
        showlegend=False,
        name="All points",
    ))

    # Layer 2: Multi-address overlay — points with boundary_count >= 2
    ma_mask = boundary_count >= 2
    ma_indices = np.where(ma_mask)[0]

    if len(ma_indices) > 0:
        ma_counts = boundary_count[ma_indices]
        ma_sizes = 4 + ma_counts * 2

        hover_texts = []
        for idx in ma_indices:
            depths = sorted(boundary_depths.get(int(idx), []))
            hover_texts.append(
                f"Title: {titles[idx]}<br>"
                f"Multi-address: {boundary_count[idx]} depths<br>"
                f"Boundary at: depths {depths}<br>"
                f"Cluster: {cluster_names[labels[idx]]}"
            )

        fig.add_trace(go.Scattergl(
            x=coords[ma_indices, 0],
            y=coords[ma_indices, 1],
            mode="markers",
            marker=dict(
                size=ma_sizes,
                color=ma_counts,
                colorscale="Plasma",
                opacity=0.8,
                colorbar=dict(
                    title=dict(text="Boundary<br>depths",
                               font=dict(color="white")),
                    x=1.02,
                    len=0.5,
                    tickfont=dict(color="white"),
                ),
            ),
            text=hover_texts,
            hoverinfo="text",
            showlegend=False,
            name="Multi-address",
        ))

    # Layer 3: Cluster labels at centroids
    for cid, name in cluster_names.items():
        mask = labels == cid
        if not mask.any():
            continue
        cx, cy = centroids[cid]
        fig.add_annotation(
            x=cx, y=cy, text=name[:25], showarrow=False,
            font=dict(size=9, color="white", family="Arial Black"),
            bgcolor="rgba(40,40,40,0.85)",
            bordercolor="rgba(100,100,100,0.6)",
            borderwidth=1, borderpad=3,
        )

    # Layer 4: HAMMER-bundled DAG edges (datashader)
    n_dag_edges = 50
    top_ma = np.argsort(-boundary_count)[:n_dag_edges * 2]  # over-sample
    top_ma = [i for i in top_ma if boundary_count[i] >= 2][:n_dag_edges]

    # Build nodes and edges DataFrames for hammer_bundle
    # Nodes: source points + shadow leaf centroids as virtual target nodes
    node_rows = []  # (id, x, y)
    edge_rows = []  # (source, target)
    next_node_id = n_points  # virtual target nodes start after real points

    for pt_idx in top_ma:
        depths = sorted(boundary_depths.get(int(pt_idx), []))
        if not depths:
            continue
        shallowest = depths[0]

        shadow_c = _find_shadow_centroid(tree, int(pt_idx), shallowest, coords)
        if shadow_c is None:
            continue

        # Source node (real point)
        node_rows.append((int(pt_idx), float(coords[pt_idx, 0]),
                          float(coords[pt_idx, 1])))
        # Target node (virtual, at shadow centroid)
        node_rows.append((next_node_id, float(shadow_c[0]), float(shadow_c[1])))
        edge_rows.append((int(pt_idx), next_node_id))
        next_node_id += 1

    edges_drawn = len(edge_rows)

    if edge_rows:
        # Deduplicate node rows (a source point may appear in multiple edges)
        seen = {}
        unique_nodes = []
        for nid, nx, ny in node_rows:
            if nid not in seen:
                seen[nid] = True
                unique_nodes.append((nid, nx, ny))

        nodes_df = pd.DataFrame(unique_nodes, columns=["id", "x", "y"]).set_index("id")
        edges_df = pd.DataFrame(edge_rows, columns=["source", "target"])

        print(f"  Running HAMMER edge bundling on {len(edge_rows)} edges...")
        t0 = time.time()
        bundled = hammer_bundle(nodes_df, edges_df,
                                initial_bandwidth=0.05,
                                decay=0.7,
                                iterations=4)
        print(f"  HAMMER bundling done in {time.time() - t0:.1f}s")

        # bundled is a DataFrame with x, y columns; NaN separates paths
        bx = bundled["x"].values
        by = bundled["y"].values

        fig.add_trace(go.Scattergl(
            x=bx, y=by,
            mode="lines",
            line=dict(color="rgba(255,255,100,0.25)", width=1),
            hoverinfo="skip",
            showlegend=False,
            name="Shadow paths",
        ))

    print(f"  {edges_drawn} HAMMER-bundled DAG edges drawn")

    # Layout
    n_ma = int(np.sum(ma_mask))
    fig.update_layout(
        title=f"PCA Tree Multi-Address — {n_ma} bridges ({100*n_ma/n_points:.1f}%) "
              f"in {n_points} points (depth={args.max_depth})",
        plot_bgcolor="#2a2a2a",
        paper_bgcolor="#2a2a2a",
        font=dict(color="white"),
        xaxis=dict(showgrid=False, showticklabels=False, zeroline=False, title=""),
        yaxis=dict(showgrid=False, showticklabels=False, zeroline=False, title="",
                   scaleanchor="x", scaleratio=1),
        dragmode="pan",
        margin=dict(l=0, r=0, t=40, b=0),
    )

    # Write and open
    out_path = str(outdir / "rog_2d_multi_address.html")
    fig.write_html(out_path, config={"scrollZoom": True})
    print(f"\nWrote {out_path}")
    subprocess.run(["open", out_path])


if __name__ == "__main__":
    main()
