"""
GUDID Medical Device Embeddings Demo

Loads FDA GUDID (Global Unique Device Identification Database) records,
generates embeddings, and analyzes structure with dyf.

The GUDID data contains ~2.7M medical devices. For demo purposes, we sample
a subset and analyze the semantic structure of device descriptions.

Requirements:
    pip install sentence-transformers polars pyarrow umap-learn plotly scikit-learn

Usage:
    # Generate embeddings (run once)
    python gudid_embeddings.py --embed --n-samples 50000 --output gudid_50k.parquet

    # Analyze with dyf
    python gudid_embeddings.py --analyze gudid_50k.parquet

    # Visualize with UMAP and edge bundling
    python gudid_embeddings.py --visualize gudid_50k.parquet

    # All in one step
    python gudid_embeddings.py --embed --analyze --visualize --n-samples 50000
"""

import argparse
import json
from pathlib import Path

import numpy as np
import polars as pl


# Path to GUDID data (from semantic-proprioception project)
GUDID_DATA_PATH = Path("/Users/jdonaldson/Projects/semantic-proprioception/data/full_gudid_devices.json")


def load_gudid_sample(n_samples: int, seed: int = 42) -> pl.DataFrame:
    """Load a random sample of GUDID devices."""
    print(f"Loading GUDID data from {GUDID_DATA_PATH}...")

    with open(GUDID_DATA_PATH) as f:
        devices = json.load(f)

    print(f"  Total devices: {len(devices):,}")

    # Convert to polars for efficient sampling
    df = pl.DataFrame(devices)

    # Sample
    rng = np.random.default_rng(seed)
    indices = rng.choice(len(df), min(n_samples, len(df)), replace=False)
    df = df[indices.tolist()]

    print(f"  Sampled: {len(df):,} devices")

    # Show distribution
    print(f"\n  Top companies:")
    company_counts = df.group_by("company").len().sort("len", descending=True)
    for row in company_counts.head(10).iter_rows(named=True):
        print(f"    {row['company'][:50]}: {row['len']:,}")

    return df


def embed_devices(
    df: pl.DataFrame,
    model_name: str = "nomic-ai/nomic-embed-text-v1.5",
    batch_size: int = 64,
    diagnose: bool = False,
) -> pl.DataFrame:
    """Generate embeddings for device descriptions.

    If diagnose=True, runs a two-pass pipeline:
    1. Embed with baseline text
    2. Discover categorical columns, diagnose axis purity
    3. Re-embed with structured text promoting under-served axes
    """
    from sentence_transformers import SentenceTransformer

    print(f"\nLoading embedding model: {model_name}")
    model = SentenceTransformer(model_name, trust_remote_code=True)

    # Build text from GMDN terms + description (no company/product codes)
    def build_text(row: dict) -> str:
        parts = []
        gmdn = row.get("gmdn_terms", [])
        if gmdn:
            parts.extend(gmdn)  # all GMDN terms first
        if row.get("description"):
            parts.append(row["description"])
        return " | ".join(parts) if parts else "Unknown device"

    texts = [build_text(row) for row in df.iter_rows(named=True)]

    # Nomic requires "search_document: " prefix for encoding
    prefixed_texts = [f"search_document: {t}" for t in texts]

    print(f"Generating embeddings for {len(texts):,} devices...")
    embeddings = model.encode(
        prefixed_texts,
        batch_size=batch_size,
        show_progress_bar=True,
        normalize_embeddings=True,
    )

    print(f"  Embedding dimension: {embeddings.shape[1]}")

    final_texts = texts

    if diagnose:
        from dyf.categorical import discover_categorical_columns, embed_with_diagnostics

        # Discover categorical columns from the dataframe
        label_cols = discover_categorical_columns(df, text_col="text")
        if label_cols:
            print(f"\n  Discovered {len(label_cols)} categorical axes: "
                  f"{', '.join(label_cols.keys())}")

            # Two-pass embed with diagnostics
            embeddings, before, after, final_texts = embed_with_diagnostics(
                embeddings, texts, label_cols,
                embed_fn=lambda t: model.encode(
                    t, batch_size=batch_size,
                    show_progress_bar=True, normalize_embeddings=True,
                ),
                lift_threshold=3.0,
                prefix="search_document: ",
            )

            # Print report
            promoted = [d.name for d in before if d.lift < 3.0]
            print(f"\n  Axis diagnostics (before → after):")
            for b in before:
                a = next((x for x in after if x.name == b.name), b)
                marker = " → PROMOTED" if b.name in promoted else ""
                print(f"    {b.name}: {b.lift:.1f}x → {a.lift:.1f}x{marker}")

            if not promoted:
                print("  All axes well-served — no re-embedding needed.")
        else:
            print("  No categorical columns discovered for diagnostics.")

    # Add embeddings and text to dataframe
    df = df.with_columns([
        pl.Series("text", final_texts),
        pl.Series("embedding", [e.tolist() for e in embeddings]),
    ])

    return df


def analyze_gudid(parquet_path: Path):
    """Analyze GUDID embeddings with dyf DensityClassifier."""
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
    from dyf import DensityClassifierFull as DensityClassifier

    print(f"\nLoading embeddings from {parquet_path}...")
    df = pl.read_parquet(parquet_path)
    print(f"  Loaded {len(df):,} devices")

    # Extract GMDN category as the primary category
    def get_primary_gmdn(gmdn_list):
        if gmdn_list and len(gmdn_list) > 0:
            return gmdn_list[0]
        return "Unknown"

    categories = [get_primary_gmdn(row) for row in df["gmdn_terms"].to_list()]

    # Add category column for classifier
    df = df.with_columns(pl.Series("category", categories))

    # Run density classifier
    print("\nRunning DensityClassifier...")
    classifier = DensityClassifier.from_polars(
        df,
        embedding_col="embedding",
        text_col="text",
        category_col="category",
        num_bits=14,
    )

    # Print report
    print(classifier.report())

    # Get labels and analyze
    result = classifier.to_polars()

    # Add GMDN category for analysis
    result = result.with_columns(pl.Series("gmdn_category", categories))

    print("\n" + "=" * 70)
    print("BUCKET SIZE DISTRIBUTION")
    print("=" * 70)

    # Create bucket size bins
    bucket_dist = (
        result
        .with_columns(
            pl.when(pl.col("bucket_size") == 1).then(pl.lit("1 (singleton)"))
            .when(pl.col("bucket_size") <= 5).then(pl.lit("2-5"))
            .when(pl.col("bucket_size") <= 10).then(pl.lit("6-10"))
            .when(pl.col("bucket_size") <= 50).then(pl.lit("11-50"))
            .when(pl.col("bucket_size") <= 100).then(pl.lit("51-100"))
            .otherwise(pl.lit("100+"))
            .alias("bucket_bin")
        )
        .group_by("bucket_bin")
        .agg(pl.len().alias("count"))
        .sort("bucket_bin")
    )
    for row in bucket_dist.iter_rows(named=True):
        pct = row["count"] / len(result) * 100
        print(f"  {row['bucket_bin']:<15} {row['count']:>6,} ({pct:>5.1f}%)")

    # Show most isolated devices
    print("\n" + "=" * 70)
    print("MOST ISOLATED DEVICES (high isolation score)")
    print("=" * 70)
    isolated = result.sort("isolation_score", descending=True).head(10)
    for row in isolated.iter_rows(named=True):
        desc = row.get("description", row.get("text", ""))[:60]
        print(f"  [{row['isolation_score']:.3f}] {desc}...")

    # Show least stable devices
    print("\n" + "=" * 70)
    print("LEAST STABLE DEVICES (low stability score)")
    print("=" * 70)
    unstable = result.sort("stability_score").head(10)
    for row in unstable.iter_rows(named=True):
        desc = row.get("description", row.get("text", ""))[:60]
        print(f"  [{row['stability_score']:.3f}] {desc}...")

    # Show GMDN category breakdown
    print("\n" + "=" * 70)
    print("GMDN CATEGORY ANALYSIS")
    print("=" * 70)

    category_stats = (
        result
        .group_by("gmdn_category")
        .agg([
            pl.len().alias("count"),
            pl.col("bucket_size").mean().alias("avg_bucket_size"),
            pl.col("isolation_score").mean().alias("avg_isolation"),
            pl.col("stability_score").mean().alias("avg_stability"),
        ])
        .sort("count", descending=True)
    )

    print(f"\nTop 15 GMDN categories by count:")
    print(f"{'Category':<45} {'Count':>7} {'AvgBkt':>7} {'AvgIso':>7} {'AvgStab':>7}")
    print("-" * 80)
    for row in category_stats.head(15).iter_rows(named=True):
        cat = row["gmdn_category"][:44]
        print(f"{cat:<45} {row['count']:>7,} {row['avg_bucket_size']:>7.1f} {row['avg_isolation']:>7.3f} {row['avg_stability']:>7.3f}")

    # Find categories with high isolation (potential anomalies/unique devices)
    print(f"\nCategories with highest average isolation (min 10 devices):")
    high_iso = category_stats.filter(pl.col("count") >= 10).sort("avg_isolation", descending=True).head(10)
    for row in high_iso.iter_rows(named=True):
        cat = row["gmdn_category"][:50]
        print(f"  [{row['avg_isolation']:.3f}] {cat} (n={row['count']:,})")

    return result


def natmerge_clustering(embeddings, classifier, categories, sim_threshold=0.4, max_clusters=12):
    """
    NatMerge clustering: natural similarity-based clusters merged to max_clusters.

    Phase 1: Greedy clustering by similarity threshold
    Phase 2: Merge smallest clusters into most-similar neighbor until max_clusters

    Returns: (assignments, cluster_to_indices, cluster_names)
    """
    from collections import defaultdict, Counter

    bucket_ids = np.array(classifier.get_bucket_ids())

    bucket_to_indices = defaultdict(list)
    for i, bid in enumerate(bucket_ids):
        bucket_to_indices[bid].append(i)

    # Compute bucket centroids
    centroids = {b: embeddings[indices].mean(axis=0) for b, indices in bucket_to_indices.items()}

    # Sort buckets by size descending
    sorted_buckets = sorted(bucket_to_indices.items(), key=lambda x: -len(x[1]))

    # Phase 1: Natural clustering by similarity threshold
    clusters = []  # list of [centroid, size, bucket_list]
    bucket_to_cluster = {}

    for bid, indices in sorted_buckets:
        bucket_centroid = centroids[bid]
        bucket_size = len(indices)

        if not clusters:
            clusters.append([bucket_centroid.copy(), bucket_size, [bid]])
            bucket_to_cluster[bid] = 0
            continue

        sims = [np.dot(bucket_centroid, c[0]) for c in clusters]
        best_cluster = np.argmax(sims)
        best_sim = sims[best_cluster]

        if best_sim >= sim_threshold:
            bucket_to_cluster[bid] = best_cluster
            old_size = clusters[best_cluster][1]
            new_size = old_size + bucket_size
            clusters[best_cluster][0] = (
                clusters[best_cluster][0] * old_size + bucket_centroid * bucket_size
            ) / new_size
            clusters[best_cluster][1] = new_size
            clusters[best_cluster][2].append(bid)
        else:
            new_cluster_id = len(clusters)
            clusters.append([bucket_centroid.copy(), bucket_size, [bid]])
            bucket_to_cluster[bid] = new_cluster_id

    print(f"  Phase 1: {len(clusters)} natural clusters (threshold={sim_threshold})")

    # Phase 2: Merge smallest clusters until we hit max_clusters
    while len(clusters) > max_clusters:
        sizes = [c[1] for c in clusters]
        smallest_idx = np.argmin(sizes)
        smallest = clusters[smallest_idx]

        sims = [np.dot(smallest[0], c[0]) if i != smallest_idx else -1
                for i, c in enumerate(clusters)]
        merge_target = np.argmax(sims)

        target = clusters[merge_target]
        old_size = target[1]
        new_size = old_size + smallest[1]
        target[0] = (target[0] * old_size + smallest[0] * smallest[1]) / new_size
        target[1] = new_size
        target[2].extend(smallest[2])

        for bid in smallest[2]:
            bucket_to_cluster[bid] = merge_target

        clusters.pop(smallest_idx)

        for i, c in enumerate(clusters):
            for bid in c[2]:
                bucket_to_cluster[bid] = i

    print(f"  Phase 2: Merged to {len(clusters)} clusters")

    # Build assignments array
    assignments = np.array([bucket_to_cluster[bid] for bid in bucket_ids], dtype=np.int32)

    # Build cluster_to_indices
    cluster_to_indices = defaultdict(list)
    for idx, cluster_id in enumerate(assignments):
        cluster_to_indices[cluster_id].append(idx)

    # Label clusters by most common category
    cluster_names = {}
    for cluster_id, indices in cluster_to_indices.items():
        cats = [categories[i] for i in indices]
        top_cat = Counter(cats).most_common(1)[0][0]
        cluster_names[cluster_id] = top_cat[:30] if len(top_cat) > 30 else top_cat

    return assignments, dict(cluster_to_indices), cluster_names


def visualize_gudid(parquet_path: Path, output_path: Path, n_clusters: int = 12, seed: int = 42):
    """Create UMAP visualization with edge bundling for GUDID embeddings."""
    from collections import defaultdict, Counter
    from umap import UMAP
    import plotly.graph_objects as go
    # Use Rust classifier for bridge analysis
    from dyf import DensityClassifier, BridgeAnalysis

    print(f"\nLoading embeddings from {parquet_path}...")
    df = pl.read_parquet(parquet_path)
    print(f"  Loaded {len(df):,} devices")

    embeddings = np.array(df['embedding'].to_list(), dtype=np.float32)
    texts = df['text'].to_list() if 'text' in df.columns else df['description'].to_list()

    # Extract GMDN categories
    def get_primary_gmdn(gmdn_list):
        if gmdn_list and len(gmdn_list) > 0:
            return gmdn_list[0]
        return "Unknown"

    categories = [get_primary_gmdn(row) for row in df["gmdn_terms"].to_list()]

    # Normalize embeddings
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    embeddings = embeddings / np.where(norms > 0, norms, 1)

    # Run Rust density classifier
    print("Running density classification...")
    classifier = DensityClassifier(embedding_dim=embeddings.shape[1], num_bits=12, seed=seed)
    classifier.fit(embeddings)
    print(f"  {classifier.report()}")

    # Analyze bridges
    print("Analyzing bridges...")
    bridge_analysis = classifier.analyze_bridges(embeddings)
    print(f"  Found {len(bridge_analysis.bridge_indices)} bridge points")

    # Project to 2D with UMAP
    print("Projecting to 2D with UMAP...")
    reducer = UMAP(
        n_components=2,
        random_state=seed,
        n_neighbors=30,      # more neighbors for better global structure
        min_dist=0.25,       # more spread
        spread=1.5,          # additional spread
        metric='cosine'      # better for normalized embeddings
    )
    coords_2d = reducer.fit_transform(embeddings)

    # Cluster using NatMerge: natural similarity clusters merged to n_clusters
    print(f"Clustering via NatMerge (similarity threshold + merge to {n_clusters})...")
    cluster_assignments, cluster_to_indices, cluster_names = natmerge_clustering(
        embeddings, classifier, categories, sim_threshold=0.4, max_clusters=n_clusters
    )

    # Compute edge bundles using force-directed bundling
    print("Computing edge bundles...")
    bucket_ids = classifier.get_bucket_ids()
    bridge_set = set(bridge_analysis.bridge_indices)

    bucket_to_indices = defaultdict(list)
    for idx, bid in enumerate(bucket_ids):
        bucket_to_indices[bid].append(idx)

    bucket_centroids = {}
    for bid, indices in bucket_to_indices.items():
        bucket_centroids[bid] = coords_2d[indices].mean(axis=0)

    # Get top connected bucket pairs
    top_pairs = bridge_analysis.top_connected_pairs(200)

    # Collect all edges first for bundling
    edges = []
    for bucket1, bucket2, count in top_pairs:
        if bucket1 not in bucket_centroids or bucket2 not in bucket_centroids:
            continue

        c1 = bucket_centroids[bucket1]
        c2 = bucket_centroids[bucket2]

        bridges = bridge_analysis.bridges_between(bucket1, bucket2)
        if len(bridges) == 0:
            # Use midpoint with slight curve
            mid = (c1 + c2) / 2
            edges.append((c1, mid, c2, count, bridges))
        else:
            # Use centroid of bridge points as control point
            bridge_centroid = coords_2d[bridges].mean(axis=0)
            edges.append((c1, bridge_centroid, c2, count, bridges))

    # Simple force-directed bundling: attract edges that are close and parallel
    # For each edge, compute a curved path using quadratic bezier
    def quadratic_bezier(p0, p1, p2, n_points=30):
        """Generate points along a quadratic bezier curve."""
        t = np.linspace(0, 1, n_points)
        x = (1-t)**2 * p0[0] + 2*(1-t)*t * p1[0] + t**2 * p2[0]
        y = (1-t)**2 * p0[1] + 2*(1-t)*t * p1[1] + t**2 * p2[1]
        return x, y

    medium_x, medium_y = [], []
    heavy_x, heavy_y = [], []

    for c1, ctrl, c2, count, bridges in edges:
        # Add some perpendicular offset based on edge density for bundling effect
        direction = c2 - c1
        length = np.linalg.norm(direction)
        if length > 0:
            perp = np.array([-direction[1], direction[0]]) / length
            # Offset control point slightly for visual separation
            offset = 0.05 * length * (hash((tuple(c1), tuple(c2))) % 10 - 5) / 5
            ctrl_adjusted = ctrl + perp * offset
        else:
            ctrl_adjusted = ctrl

        x_curve, y_curve = quadratic_bezier(c1, ctrl_adjusted, c2, n_points=40)

        if count >= 3:
            heavy_x.extend(list(x_curve) + [None])
            heavy_y.extend(list(y_curve) + [None])
        else:
            medium_x.extend(list(x_curve) + [None])
            medium_y.extend(list(y_curve) + [None])

    print(f"  Medium edges: {sum(1 for x in medium_x if x is None)} paths")
    print(f"  Heavy edges: {sum(1 for x in heavy_x if x is None)} paths")

    # Create visualization
    print("Creating visualization...")
    colors = [
        '#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7', '#DDA0DD',
        '#98D8C8', '#F7DC6F', '#BB8FCE', '#85C1E9', '#F8B500', '#00CED1',
        '#E74C3C', '#3498DB', '#2ECC71', '#9B59B6', '#F39C12', '#1ABC9C',
    ]
    bg_color = "#1a1a2e"

    traces = []

    # Medium density edges
    traces.append(go.Scattergl(
        x=medium_x, y=medium_y,
        mode='lines',
        line=dict(color='rgba(120,160,200,0.08)', width=0.8),
        hoverinfo='none',
        name='Medium density bridges'
    ))

    # Heavy density edges
    traces.append(go.Scattergl(
        x=heavy_x, y=heavy_y,
        mode='lines',
        line=dict(color='rgba(255,255,255,0.5)', width=2),
        hoverinfo='none',
        name='High density bridges'
    ))

    # Cluster scatter traces
    annotations = []
    for cluster_id in sorted(cluster_to_indices.keys()):
        indices = cluster_to_indices[cluster_id]
        color = colors[cluster_id % len(colors)]
        name = cluster_names.get(cluster_id, f"Cluster {cluster_id}")

        traces.append(go.Scattergl(
            x=coords_2d[indices, 0],
            y=coords_2d[indices, 1],
            mode='markers',
            marker=dict(color=color, size=4, opacity=0.8),
            name=f"{name} ({len(indices):,})",
            text=[texts[i][:100] for i in indices],
            hovertemplate='%{text}<extra></extra>'
        ))

        cx = coords_2d[indices, 0].mean()
        cy = coords_2d[indices, 1].mean()
        annotations.append(dict(
            x=cx, y=cy,
            text=f"<b>{name}</b>",
            showarrow=False,
            font=dict(size=9, color='white', family='Arial'),
            bgcolor='rgba(30,30,30,0.8)',
            bordercolor=color,
            borderwidth=1,
            borderpad=3
        ))

    fig = go.Figure(data=traces)

    fig.update_layout(
        title=dict(
            text=f'<b>FDA GUDID Medical Device Semantic Map</b><br><sup>{len(texts):,} devices | Bright lines = density bridges connecting device clusters</sup>',
            font=dict(size=20, color='white', family='Arial'),
            x=0.5, xanchor='center'
        ),
        showlegend=True,
        legend=dict(
            bgcolor='rgba(20,20,20,0.8)',
            bordercolor='rgba(255,255,255,0.2)',
            borderwidth=1,
            font=dict(color='white', size=9),
            title=dict(text='Device Categories', font=dict(color='#aaa', size=9))
        ),
        hovermode='closest',
        dragmode='pan',
        annotations=annotations,
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False, showline=False, scaleanchor="y", scaleratio=1),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False, showline=False),
        plot_bgcolor=bg_color,
        paper_bgcolor=bg_color,
        width=1600, height=1000,
        margin=dict(l=20, r=20, t=80, b=20)
    )

    fig.write_html(output_path, config={'scrollZoom': True, 'displayModeBar': True})
    print(f"Saved visualization to {output_path}")

    return output_path


def main():
    parser = argparse.ArgumentParser(description="GUDID medical device embedding analysis")
    parser.add_argument("--embed", action="store_true", help="Generate embeddings")
    parser.add_argument("--analyze", nargs="?", const="gudid_50k.parquet", help="Analyze parquet file")
    parser.add_argument("--visualize", nargs="?", const="gudid_50k.parquet", help="Visualize with UMAP + edge bundling")
    parser.add_argument("--n-samples", type=int, default=50000, help="Number of devices to sample")
    parser.add_argument("--n-clusters", type=int, default=12, help="Number of clusters for visualization")
    parser.add_argument("--output", "-o", default="gudid_50k.parquet", help="Output parquet file")
    parser.add_argument("--diagnose", action="store_true",
                        help="Run two-pass embedding with axis diagnostics")
    parser.add_argument("--model", default="nomic-ai/nomic-embed-text-v1.5", help="Embedding model")
    parser.add_argument("--batch-size", type=int, default=64, help="Embedding batch size")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--open", action="store_true", help="Open visualization in browser")
    args = parser.parse_args()

    output_path = Path(args.output)

    if args.embed:
        # Load and embed
        df = load_gudid_sample(args.n_samples, seed=args.seed)
        df = embed_devices(df, model_name=args.model, batch_size=args.batch_size,
                           diagnose=args.diagnose)

        # Save
        df.write_parquet(output_path)
        print(f"\nSaved to {output_path} ({output_path.stat().st_size / 1e6:.1f} MB)")

    if args.analyze is not None:
        # If --analyze provided without value, use output_path; otherwise use provided path
        if args.analyze == "gudid_50k.parquet" and args.embed:
            analyze_path = output_path
        else:
            analyze_path = Path(args.analyze)
        if not analyze_path.exists():
            print(f"ERROR: {analyze_path} not found. Run with --embed first.")
            return
        analyze_gudid(analyze_path)

    if args.visualize is not None:
        # Determine input path
        if args.visualize == "gudid_50k.parquet" and args.embed:
            viz_input_path = output_path
        else:
            viz_input_path = Path(args.visualize)
        if not viz_input_path.exists():
            print(f"ERROR: {viz_input_path} not found. Run with --embed first.")
            return

        # Output HTML path
        viz_output_path = viz_input_path.with_suffix('.html')

        html_path = visualize_gudid(
            viz_input_path,
            viz_output_path,
            n_clusters=args.n_clusters,
            seed=args.seed
        )

        if args.open:
            import webbrowser
            webbrowser.open(f"file://{html_path.absolute()}")


if __name__ == "__main__":
    main()
