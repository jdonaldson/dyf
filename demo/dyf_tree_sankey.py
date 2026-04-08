"""
DYF Tree Sankey Visualization

Plotly Sankey showing the full DYF tree pipeline:
  tree levels → cut clusters → final clusters (after refine_clusters)

Usage:
    python demo/dyf_tree_sankey.py demo/gudid_50k_enriched_titled.parquet --sample 8000
"""

import argparse
import colorsys
from collections import defaultdict
from pathlib import Path

import numpy as np
import polars as pl
import plotly.graph_objects as go

from dyf import cut_tree_to_labels
from dyf.dyf_tree import (
    build_dyf_tree,
    refine_dyf_tree,
    _leaf_coherence,
)
from dyf_rs import DensityClassifier


# ── Helpers ──────────────────────────────────────────────────────────────


def load_embeddings(path, sample=None, seed=42):
    """Load embeddings from parquet file."""
    df = pl.read_parquet(path)
    if sample and sample < len(df):
        df = df.sample(n=sample, seed=seed)
    embeddings = np.array(df["embedding"].to_list(), dtype=np.float32)
    if "title" in df.columns:
        titles = df["title"].to_list()
    elif "text" in df.columns:
        titles = [t[:50] for t in df["text"].to_list()]
    else:
        titles = [f"Item {i}" for i in range(len(embeddings))]
    return embeddings, titles


def golden_ratio_color(i, _n=0, lightness=0.45, saturation=0.6):
    """Generate a color using golden ratio hue spacing."""
    hue = (i * 0.618033988749895) % 1.0
    r, g, b = colorsys.hls_to_rgb(hue, lightness, saturation)
    return r, g, b


def rgb_to_hex(r, g, b):
    return f"#{int(r*255):02x}{int(g*255):02x}{int(b*255):02x}"


def rgba_str(r, g, b, a=0.4):
    return f"rgba({int(r*255)},{int(g*255)},{int(b*255)},{a})"


# ── Tree traversal ──────────────────────────────────────────────────────


def collect_nodes_by_level(tree):
    """Walk the tree and collect nodes grouped by their actual level.

    Returns dict: level -> list of (node, parent_node_or_None).
    Level 0 = root, level 1 = root's children, etc.
    """
    levels = defaultdict(list)

    def _walk(node, level, parent):
        levels[level].append((node, parent))
        for child in node["children"]:
            _walk(child, level + 1, node)

    _walk(tree, 0, None)
    return dict(levels)


def assign_node_ids(nodes_by_level):
    """Give every tree node a unique integer id. Returns node->id mapping."""
    node_id = {}
    counter = 0
    for level in sorted(nodes_by_level.keys()):
        for node, _ in nodes_by_level[level]:
            node_id[id(node)] = counter
            counter += 1
    return node_id


# ── Grouping strategy for deep levels ───────────────────────────────────


def build_sankey_nodes_and_links(
    tree, n_levels, min_display_size=30, max_nodes_per_parent=8
):
    """Build Sankey node/link arrays from the DYF tree.

    Levels 0-2: show every node individually.
    Levels 3+: for each parent, keep children with >= min_display_size
    points, collapse the rest into an aggregate "{parent} other" node.

    Returns:
        sankey_nodes: list of dicts with 'label', 'level', 'point_set'
        sankey_links: list of dicts with 'source', 'target', 'value'
    """
    nodes_by_level = collect_nodes_by_level(tree)
    max_level = max(nodes_by_level.keys())

    # Clamp n_levels to actual tree depth + 1
    n_levels = min(n_levels, max_level + 1)

    # Phase 1: Decide which tree nodes get their own Sankey node vs grouped
    # sankey_node_id -> {label, level, point_set}
    sankey_nodes = []
    sankey_links = []

    # Maps tree node id(obj) -> sankey_node_index
    tree_to_sankey = {}

    for level in range(n_levels):
        if level not in nodes_by_level:
            continue

        if level <= 2:
            # Show every node individually
            for node, parent in nodes_by_level[level]:
                n_pts = len(node["indices"])
                label = f"L{level} ({n_pts})"
                idx = len(sankey_nodes)
                sankey_nodes.append({
                    "label": label,
                    "level": level,
                    "point_set": set(node["indices"].tolist()),
                })
                tree_to_sankey[id(node)] = idx

                # Link from parent
                if parent is not None and id(parent) in tree_to_sankey:
                    sankey_links.append({
                        "source": tree_to_sankey[id(parent)],
                        "target": idx,
                        "value": n_pts,
                    })
        else:
            # Group small children per parent
            # Group nodes by their parent
            children_of = defaultdict(list)
            for node, parent in nodes_by_level[level]:
                parent_key = id(parent) if parent is not None else None
                children_of[parent_key].append(node)

            for parent_key, children in children_of.items():
                if parent_key not in tree_to_sankey and parent_key is not None:
                    # Parent was itself grouped - find which sankey node
                    # contains the parent's points.  Skip for now, these
                    # will be handled by the "other" bucket of the
                    # grandparent.
                    continue

                # Sort children by size descending
                children.sort(key=lambda n: -len(n["indices"]))

                # Keep individually if large enough
                big = []
                small_pool_indices = set()
                for child in children:
                    n_pts = len(child["indices"])
                    if n_pts >= min_display_size and len(big) < max_nodes_per_parent:
                        big.append(child)
                    else:
                        small_pool_indices.update(child["indices"].tolist())

                # Create individual nodes for big children
                for child in big:
                    n_pts = len(child["indices"])
                    label = f"L{level} ({n_pts})"
                    idx = len(sankey_nodes)
                    sankey_nodes.append({
                        "label": label,
                        "level": level,
                        "point_set": set(child["indices"].tolist()),
                    })
                    tree_to_sankey[id(child)] = idx

                    if parent_key is not None and parent_key in tree_to_sankey:
                        sankey_links.append({
                            "source": tree_to_sankey[parent_key],
                            "target": idx,
                            "value": n_pts,
                        })

                # Create aggregate "other" node for small children
                if small_pool_indices:
                    n_pts = len(small_pool_indices)
                    label = f"L{level} other ({n_pts})"
                    idx = len(sankey_nodes)
                    sankey_nodes.append({
                        "label": label,
                        "level": level,
                        "point_set": small_pool_indices,
                    })
                    if parent_key is not None and parent_key in tree_to_sankey:
                        sankey_links.append({
                            "source": tree_to_sankey[parent_key],
                            "target": idx,
                            "value": n_pts,
                        })

    return sankey_nodes, sankey_links, n_levels


# ── Cut cluster + refine_clusters instrumentation ───────────────────────


def instrumented_refine_clusters(
    labels, embeddings, min_coherence=None, min_cluster_size=None,
    num_bits=6, seed_offset=2000,
):
    """Replicate refine_clusters step-by-step, capturing flow data.

    Returns:
        final_labels: refined label array
        flows: list of (source_label, target_label, n_points, flow_type)
            where flow_type is one of:
            'core_retain'  - points that stayed in their cut cluster
            'eject'        - points ejected from cut cluster to pool
            'resplit'      - points from pool assigned to new bucket
            'merge_small'  - small cluster merged into another
            'merge_large'  - small cluster merged into large cluster
    """
    labels = np.asarray(labels).copy()
    embeddings = np.asarray(embeddings)
    n_original_clusters = len(set(labels.tolist()))

    if min_cluster_size is None:
        min_cluster_size = max(10, len(labels) // max(n_original_clusters, 1) // 3)

    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    emb_normed = embeddings / np.maximum(norms, 1e-10)

    # Per-cluster coherence
    unique_labels = sorted(set(labels.tolist()))
    cluster_coherence = {}
    cluster_members = {}
    for cid in unique_labels:
        members = np.where(labels == cid)[0]
        cluster_members[cid] = members
        if len(members) >= 2:
            cluster_coherence[cid] = _leaf_coherence(members, emb_normed)
        else:
            cluster_coherence[cid] = 1.0

    coh_values = np.array(list(cluster_coherence.values()))
    if min_coherence is None:
        threshold = float(np.percentile(coh_values, 25)) if len(coh_values) > 0 else 0.0
    else:
        threshold = float(min_coherence)

    # Track per-point flow: cut_label -> final_label
    # We record: for each point, which cut cluster it came from
    cut_labels = labels.copy()

    # Phase 1: Eject periphery
    ejected_indices = []
    for cid in unique_labels:
        if cluster_coherence[cid] >= threshold:
            continue
        members = cluster_members[cid]
        if len(members) < 4:
            continue
        subset = emb_normed[members]
        centroid = subset.mean(axis=0)
        norm = np.linalg.norm(centroid)
        if norm > 1e-10:
            centroid /= norm
        sims = subset @ centroid
        median_sim = float(np.median(sims))
        periphery_mask = sims < median_sim
        periphery = members[periphery_mask]
        ejected_indices.extend(periphery.tolist())
        labels[periphery] = -1

    if not ejected_indices:
        # No ejections - every point stays in its cut cluster
        flows = []
        for cid in unique_labels:
            members = cluster_members[cid]
            flows.append((cid, cid, len(members), "core_retain"))
        return labels, flows

    ejected_indices = np.array(ejected_indices)

    # Phase 2: Re-split ejected pool
    next_label = max(unique_labels) + 1
    ejected_emb = embeddings[ejected_indices]
    dim = ejected_emb.shape[1]

    try:
        clf = DensityClassifier(embedding_dim=dim, num_bits=num_bits, seed=seed_offset)
        clf.fit(ejected_emb)
        bucket_ids = clf.get_bucket_ids()
    except Exception:
        bucket_ids = np.zeros(len(ejected_indices), dtype=int)

    unique_buckets = sorted(set(bucket_ids.tolist()))
    resplit_label_map = {}  # bucket_id -> new_label
    for bid in unique_buckets:
        mask = bucket_ids == bid
        global_indices = ejected_indices[np.where(mask)[0]]
        labels[global_indices] = next_label
        resplit_label_map[bid] = next_label
        next_label += 1

    # Phase 3: Merge small clusters
    all_cids = sorted(set(labels.tolist()))
    all_cids = [c for c in all_cids if c != -1]
    large_cids = []
    small_cids = []
    cluster_cents = {}
    cluster_sizes = {}
    for cid in all_cids:
        members = np.where(labels == cid)[0]
        if len(members) == 0:
            continue
        cent = emb_normed[members].mean(axis=0)
        norm = np.linalg.norm(cent)
        if norm > 1e-10:
            cent /= norm
        cluster_cents[cid] = cent
        cluster_sizes[cid] = len(members)
        if len(members) >= min_cluster_size:
            large_cids.append(cid)
        else:
            small_cids.append(cid)

    # Phase 3a: merge nearest small pairs
    merge_map = {}  # old_cid -> new_cid (for tracking)
    while len(small_cids) >= 2:
        best_sim = -1.0
        best_i, best_j = 0, 1
        for i in range(len(small_cids)):
            for j in range(i + 1, len(small_cids)):
                sim = float(cluster_cents[small_cids[i]] @ cluster_cents[small_cids[j]])
                if sim > best_sim:
                    best_sim = sim
                    best_i, best_j = i, j
        ci, cj = small_cids[best_i], small_cids[best_j]
        labels[labels == cj] = ci
        merge_map[cj] = ci
        cluster_sizes[ci] = cluster_sizes[ci] + cluster_sizes[cj]
        del cluster_sizes[cj]
        members_ci = np.where(labels == ci)[0]
        cent = emb_normed[members_ci].mean(axis=0)
        norm = np.linalg.norm(cent)
        if norm > 1e-10:
            cent /= norm
        cluster_cents[ci] = cent
        small_cids.pop(best_j)
        if cluster_sizes[ci] >= min_cluster_size:
            small_cids.remove(ci)
            large_cids.append(ci)

    # Phase 3b: merge remaining small into nearest large
    if small_cids and large_cids:
        for cid in small_cids:
            large_cents = np.array([cluster_cents[c] for c in large_cids])
            sims = cluster_cents[cid] @ large_cents.T
            nearest = large_cids[int(np.argmax(sims))]
            labels[labels == cid] = nearest
            merge_map[cid] = nearest

    # Compact labels
    unique_final = sorted(set(labels.tolist()))
    remap = {old: new for new, old in enumerate(unique_final)}
    labels = np.array([remap[l] for l in labels], dtype=int)

    # Build flow records by comparing cut_labels → labels per point
    # Group by (cut_cluster, final_cluster)
    flow_counts = defaultdict(int)
    for i in range(len(labels)):
        cut_c = int(cut_labels[i])
        final_c = int(labels[i])
        flow_counts[(cut_c, final_c)] += 1

    flows = []
    for (cut_c, final_c), count in flow_counts.items():
        flows.append((cut_c, final_c, count, "flow"))

    return labels, flows


# ── Sankey builder ──────────────────────────────────────────────────────


def build_sankey(
    tree, embeddings, n_clusters, n_levels,
    min_display_size=30, max_nodes_per_parent=8,
):
    """Build the complete 9-column Sankey data.

    Returns Plotly-compatible (node_labels, node_colors, node_x, node_y,
    link_sources, link_targets, link_values, link_colors).
    """
    n_points = len(embeddings)

    # Columns 1-7: Tree levels
    tree_nodes, tree_links, actual_levels = build_sankey_nodes_and_links(
        tree, n_levels,
        min_display_size=min_display_size,
        max_nodes_per_parent=max_nodes_per_parent,
    )

    print(f"  Tree Sankey: {len(tree_nodes)} nodes, {len(tree_links)} links "
          f"across {actual_levels} levels")

    # Column 8: Cut clusters
    cut_labels = cut_tree_to_labels(tree, n_points, n_clusters, embeddings=embeddings)
    n_cut = len(set(cut_labels.tolist()))
    print(f"  Cut clusters: {n_cut}")

    # Column 9: Final clusters (after refine_clusters)
    final_labels, refine_flows = instrumented_refine_clusters(
        cut_labels, embeddings
    )
    n_final = len(set(final_labels.tolist()))
    print(f"  Final clusters: {n_final}")

    # Generate colors for final clusters (golden ratio)
    final_unique = sorted(set(final_labels.tolist()))
    n_final_clusters = len(final_unique)
    final_colors = {}
    for i, cid in enumerate(final_unique):
        r, g, b = golden_ratio_color(i, n_final_clusters)
        final_colors[cid] = (r, g, b)

    # Build Sankey arrays
    # Nodes: tree_nodes + cut_cluster_nodes + final_cluster_nodes
    node_labels = []
    node_colors = []
    node_x = []
    node_y = []

    # X positions: spread across 9 columns
    n_columns = actual_levels + 2  # tree levels + cut + final
    x_positions = [i / max(n_columns - 1, 1) for i in range(n_columns)]
    # Slight padding to avoid edge clipping
    x_positions = [0.001 + x * 0.998 for x in x_positions]

    # Tree nodes
    for sn in tree_nodes:
        node_labels.append(sn["label"])
        node_colors.append("rgba(150,150,150,0.6)")
        node_x.append(x_positions[sn["level"]])
        node_y.append(0.5)  # placeholder, will be computed

    # Cut cluster nodes
    cut_unique = sorted(set(cut_labels.tolist()))
    cut_node_map = {}  # cut_label -> sankey_node_index
    for i, cid in enumerate(cut_unique):
        idx = len(node_labels)
        cut_node_map[cid] = idx
        n_pts = int((cut_labels == cid).sum())
        node_labels.append(f"Cut {cid} ({n_pts})")
        node_colors.append("rgba(150,150,150,0.6)")
        node_x.append(x_positions[actual_levels])
        node_y.append(0.5)

    # Final cluster nodes
    final_node_map = {}  # final_label -> sankey_node_index
    for i, cid in enumerate(final_unique):
        idx = len(node_labels)
        final_node_map[cid] = idx
        n_pts = int((final_labels == cid).sum())
        r, g, b = final_colors[cid]
        node_labels.append(f"Final {cid} ({n_pts})")
        node_colors.append(rgb_to_hex(r, g, b))
        node_x.append(x_positions[actual_levels + 1])
        node_y.append(0.5)

    # Links: tree internal
    link_sources = []
    link_targets = []
    link_values = []
    link_colors = []

    for tl in tree_links:
        link_sources.append(tl["source"])
        link_targets.append(tl["target"])
        link_values.append(tl["value"])
        link_colors.append("rgba(150,150,150,0.15)")

    # Links: last tree level -> cut clusters
    # For each tree node at the deepest shown level, compute overlap with
    # cut clusters
    last_tree_level = actual_levels - 1
    leaf_sankey_nodes = [
        (i, sn) for i, sn in enumerate(tree_nodes)
        if sn["level"] == last_tree_level
    ]

    # Also include nodes from earlier levels that are leaves (no children
    # linking out of them in our Sankey)
    nodes_with_outgoing = set()
    for tl in tree_links:
        nodes_with_outgoing.add(tl["source"])

    for i, sn in enumerate(tree_nodes):
        if i not in nodes_with_outgoing and sn["level"] < last_tree_level:
            leaf_sankey_nodes.append((i, sn))

    for sn_idx, sn in leaf_sankey_nodes:
        pts = sn["point_set"]
        # Count how many points go to each cut cluster
        flow = defaultdict(int)
        for p in pts:
            flow[int(cut_labels[p])] += 1
        for cut_cid, count in flow.items():
            if cut_cid in cut_node_map:
                link_sources.append(sn_idx)
                link_targets.append(cut_node_map[cut_cid])
                link_values.append(count)
                link_colors.append("rgba(150,150,150,0.15)")

    # Links: cut clusters -> final clusters
    for cut_c, final_c, count, _flow_type in refine_flows:
        if cut_c in cut_node_map and final_c in final_node_map:
            r, g, b = final_colors[final_c]
            link_sources.append(cut_node_map[cut_c])
            link_targets.append(final_node_map[final_c])
            link_values.append(count)
            link_colors.append(rgba_str(r, g, b, 0.4))

    # Propagate final cluster colors backward through tree
    # For each tree node, find which final cluster gets the most points
    for i, sn in enumerate(tree_nodes):
        pts = sn["point_set"]
        if not pts:
            continue
        # Dominant final cluster
        final_counts = defaultdict(int)
        for p in pts:
            final_counts[int(final_labels[p])] += 1
        dominant = max(final_counts, key=final_counts.get)
        r, g, b = final_colors[dominant]
        node_colors[i] = rgba_str(r, g, b, 0.7)

    # Same for cut cluster nodes
    for cut_cid, sn_idx in cut_node_map.items():
        pts = np.where(cut_labels == cut_cid)[0]
        final_counts = defaultdict(int)
        for p in pts:
            final_counts[int(final_labels[p])] += 1
        dominant = max(final_counts, key=final_counts.get)
        r, g, b = final_colors[dominant]
        node_colors[sn_idx] = rgba_str(r, g, b, 0.7)

    # Also propagate colors to tree links
    for i, tl in enumerate(tree_links):
        target_sn = tree_nodes[tl["target"]]
        pts = target_sn["point_set"]
        if pts:
            final_counts = defaultdict(int)
            for p in pts:
                final_counts[int(final_labels[p])] += 1
            dominant = max(final_counts, key=final_counts.get)
            r, g, b = final_colors[dominant]
            link_colors[i] = rgba_str(r, g, b, 0.15)

    # Compute Y positions: position children near their parents to minimize
    # link crossings.  Process columns left-to-right; for each node, compute
    # a target Y from the weighted average Y of its incoming sources.  Then
    # spread nodes within each column to avoid overlap while preserving that
    # relative order.

    # Build incoming-link index: target_node -> [(source_node, weight), ...]
    incoming = defaultdict(list)
    for s, t, v in zip(link_sources, link_targets, link_values):
        incoming[t].append((s, v))

    # Group nodes by column (ordered by x position)
    col_nodes = defaultdict(list)
    for i in range(len(node_labels)):
        col_nodes[node_x[i]].append(i)

    sorted_cols = sorted(col_nodes.keys())

    for col_x in sorted_cols:
        indices = col_nodes[col_x]
        n = len(indices)
        if n == 1:
            node_y[indices[0]] = 0.5
            continue

        # Compute target Y for each node from parent positions
        target_y = {}
        for idx in indices:
            sources = incoming.get(idx, [])
            if sources:
                total_w = sum(w for _, w in sources)
                if total_w > 0:
                    target_y[idx] = sum(
                        node_y[s] * w for s, w in sources
                    ) / total_w
                else:
                    target_y[idx] = 0.5
            else:
                target_y[idx] = 0.5

        # Sort by target Y to preserve parent-relative ordering
        indices.sort(key=lambda idx: target_y[idx])

        # Distribute evenly within [0.001, 0.999] in that order
        for j, idx in enumerate(indices):
            node_y[idx] = 0.001 + (j / max(n - 1, 1)) * 0.998

    return (
        node_labels, node_colors, node_x, node_y,
        link_sources, link_targets, link_values, link_colors,
        actual_levels, n_cut, n_final,
    )


# ── Main ────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(
        description="DYF tree Sankey visualization")
    parser.add_argument("parquet_path", help="Path to embeddings parquet")
    parser.add_argument("--sample", type=int, default=8000)
    parser.add_argument("--n-clusters", type=int, default=25,
                        help="Target number of clusters")
    parser.add_argument("--dyf-depth", type=int, default=8,
                        help="DYF tree max depth")
    parser.add_argument("--dyf-bits", type=int, default=3,
                        help="DYF tree LSH bits per level")
    parser.add_argument("--min-display-size", type=int, default=30,
                        help="Min points for a node to be shown individually "
                        "at deep levels")
    parser.add_argument("--output", "-o", default=None,
                        help="Output HTML path (default: alongside input)")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    print(f"Loading embeddings from {args.parquet_path}...")
    embeddings, titles = load_embeddings(
        args.parquet_path, sample=args.sample, seed=args.seed)
    n = len(embeddings)
    print(f"  {n:,} points, dim={embeddings.shape[1]}")

    print(f"\nBuilding DYF tree (depth={args.dyf_depth}, bits={args.dyf_bits})...")
    tree = build_dyf_tree(
        embeddings, max_depth=args.dyf_depth,
        num_bits=args.dyf_bits, min_leaf_size=4, seed=args.seed,
    )

    print("Refining tree...")
    stats = refine_dyf_tree(tree, embeddings)
    print(f"  Refined {stats['n_refined']} leaves "
          f"(coherence {stats['coherence_before']:.3f} -> "
          f"{stats['coherence_after']:.3f}, "
          f"leaves {stats['n_leaves_before']} -> {stats['n_leaves_after']})")

    # Count actual tree levels
    nodes_by_level = collect_nodes_by_level(tree)
    max_level = max(nodes_by_level.keys())
    print(f"  Tree has {max_level + 1} levels, "
          f"nodes per level: {', '.join(f'{len(nodes_by_level[l])}' for l in range(max_level + 1))}")

    print(f"\nBuilding Sankey data...")
    (
        node_labels, node_colors, node_x, node_y,
        link_sources, link_targets, link_values, link_colors,
        actual_levels, n_cut, n_final,
    ) = build_sankey(
        tree, embeddings, args.n_clusters,
        n_levels=max_level + 1,
        min_display_size=args.min_display_size,
    )

    print(f"  Sankey: {len(node_labels)} nodes, {len(link_sources)} links")
    print(f"  Columns: {actual_levels} tree levels + cut ({n_cut}) + final ({n_final})")

    # Build Plotly figure
    fig = go.Figure(data=[go.Sankey(
        arrangement="snap",
        node=dict(
            pad=2,
            thickness=15,
            line=dict(color="rgba(50,50,50,0.5)", width=0.5),
            label=node_labels,
            color=node_colors,
            x=node_x,
            y=node_y,
        ),
        link=dict(
            source=link_sources,
            target=link_targets,
            value=link_values,
            color=link_colors,
        ),
    )])

    # Column annotations
    col_names = [f"Level {i}" for i in range(actual_levels)]
    col_names.append(f"Cut ({n_cut})")
    col_names.append(f"Final ({n_final})")

    n_columns = actual_levels + 2
    x_positions = [i / max(n_columns - 1, 1) for i in range(n_columns)]
    x_positions = [0.001 + x * 0.998 for x in x_positions]

    annotations = []
    for i, name in enumerate(col_names):
        annotations.append(dict(
            x=x_positions[i],
            y=1.05,
            xref="paper",
            yref="paper",
            text=f"<b>{name}</b>",
            showarrow=False,
            font=dict(size=12, color="#ccc"),
        ))

    fig.update_layout(
        title=dict(
            text=f"DYF Tree Pipeline — {n:,} points, "
                 f"{actual_levels} levels → {n_cut} cut → {n_final} final",
            font=dict(size=16, color="#ddd"),
        ),
        font=dict(size=10, color="#ccc"),
        paper_bgcolor="#1a1a2e",
        plot_bgcolor="#1a1a2e",
        width=2400,
        height=1200,
        dragmode="pan",
        annotations=annotations,
        margin=dict(l=20, r=20, t=80, b=20),
    )

    # Output path
    if args.output:
        out_path = Path(args.output)
    else:
        out_path = Path(args.parquet_path).parent / "dyf_tree_sankey.html"

    fig.write_html(
        str(out_path),
        config={"scrollZoom": True},
    )
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
