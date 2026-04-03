"""Level 2 → 3: Bridge edges + narration enrichment."""

import json
import logging
import re
from collections import defaultdict

import numpy as np

from dyf.lazy_index import LazyIndex, rewrite_lazy_index
from dyf.provenance import create_provenance, provenance_to_dict

from ._narration import _generate_narration

logger = logging.getLogger(__name__)


def compute_bridge_edges(coords, embeddings, labels, n_clusters):
    """Compute cross-cluster bridge edges using ROG ontology."""
    import dyf

    logger.info("  Building ROG ontology for bridge detection...")
    result = dyf.build_rog_ontology(
        embeddings, initial_threshold=0.55, min_threshold=0.35,
        target_coverage=0.95, verbose=False)

    ont = result.ontology
    pair_counts = defaultdict(int)
    for parent, children_list in ont.children.items():
        for child, sim, div_gap in children_list:
            c1, c2 = int(labels[parent]), int(labels[child])
            if c1 != c2:
                pair = (min(c1, c2), max(c1, c2))
                pair_counts[pair] += 1

    cross = sum(pair_counts.values())
    logger.info(f"  {len(pair_counts)} cluster pairs, {cross:,} cross-cluster edges")

    centroids = np.zeros((n_clusters, coords.shape[1]), dtype=np.float32)
    for c in range(n_clusters):
        mask = labels == c
        if mask.any():
            centroids[c] = coords[mask].mean(axis=0)

    edge_list = sorted(pair_counts.keys(), key=lambda p: -pair_counts[p])
    if not edge_list:
        return [], {}

    edge_pairs = [[int(c1), int(c2), int(pair_counts[(c1, c2)])]
                  for c1, c2 in edge_list]

    import pandas as pd
    from datashader.bundling import hammer_bundle

    centroids_2d = centroids[:, :2]
    nodes_df = pd.DataFrame({
        "x": centroids_2d[:, 0].astype(float),
        "y": centroids_2d[:, 1].astype(float),
    })
    edges_df = pd.DataFrame({
        "source": [e[0] for e in edge_list],
        "target": [e[1] for e in edge_list],
    })
    bundled_df = hammer_bundle(nodes_df, edges_df)

    edge_paths_2d = []
    current_path = []
    for _, row in bundled_df.iterrows():
        if pd.isna(row["x"]) or pd.isna(row["y"]):
            if current_path:
                edge_paths_2d.append(current_path)
                current_path = []
        else:
            current_path.append([round(row["x"], 4), round(row["y"], 4)])
    if current_path:
        edge_paths_2d.append(current_path)

    logger.info(f"  Bundled {len(edge_paths_2d)} 2D edge paths")
    return edge_pairs, edge_paths_2d


def enrich_viz(dyf_path, cluster_level=None, model="gpt-oss:20b",
               title=None, output_path=None, force=False, domain=None):
    """Add bridge edges and tour narration (Level 2 → 3)."""
    logger.info("=== Level 3: Viz Enrichment ===")
    logger.info(f"  Input: {dyf_path}")

    with LazyIndex(dyf_path) as idx:
        level = idx.detect_enrichment_level()
        if level < 2:
            logger.warning(f"  Need level 2 (clusters), got level {level}. "
                          f"Run 'cluster' first.")
            return
        if level >= 3 and not force:
            logger.info(f"  Already at level {level} (viz-ready), skipping. "
                       f"Use --force to re-run.")
            return

    with LazyIndex(dyf_path) as idx:
        data = idx.extract_all_fields()
    n = len(data['embeddings'])

    coords = np.column_stack([
        data['fields']['umap_x'],
        data['fields']['umap_y'],
        data['fields']['umap_z'],
    ])
    embeddings = data['embeddings']

    if domain is None:
        domain = data['metadata'].get('domain')
    if domain:
        logger.info(f"  Domain: {domain}")

    # Resolve cluster labels and names
    use_louvain = ('community_id' in data['fields']
                   and 'louvain_dendrogram' in data['metadata'])

    if use_louvain:
        labels = np.asarray(data['fields']['community_id'], dtype=np.int32)
        n_clusters = len(set(labels.tolist()))
        dendro = json.loads(data['metadata']['louvain_dendrogram'])
        cluster_names = {int(k): v
                         for k, v in dendro['community_names'].items()}
        logger.info(f"  Using community_id ({n_clusters} communities)")
    else:
        if cluster_level is None:
            available = [
                f for f in data['fields']
                if re.match(r'cluster_\d+_2d$', f)
            ]
            if available:
                cluster_level = min(
                    int(re.match(r'cluster_(\d+)', f).group(1))
                    for f in available)
                logger.info(f"  Auto-detected cluster_level={cluster_level}")
            else:
                available_bare = [
                    f for f in data['fields']
                    if re.match(r'cluster_\d+$', f)
                ]
                if available_bare:
                    cluster_level = min(
                        int(re.match(r'cluster_(\d+)', f).group(1))
                        for f in available_bare)
                    logger.info(f"  Auto-detected cluster_level={cluster_level} "
                               f"(bare)")
                else:
                    logger.warning("  No cluster fields found in .dyf file.")
                    return

        cluster_field_2d = f'cluster_{cluster_level}_2d'
        cluster_field_bare = f'cluster_{cluster_level}'
        if cluster_field_2d in data['fields']:
            cluster_field = cluster_field_2d
            names_key = f'cluster_names_{cluster_level}_2d'
        elif cluster_field_bare in data['fields']:
            cluster_field = cluster_field_bare
            names_key = f'cluster_names_{cluster_level}'
        else:
            available = [f for f in data['fields']
                         if f.startswith('cluster_')]
            logger.warning(f"  cluster_{cluster_level} not found. "
                          f"Available: {available}")
            return
        labels = data['fields'][cluster_field]
        n_clusters = len(set(labels.tolist()))
        names_json = data['metadata'].get(names_key, '{}')
        cluster_names = {int(k): v
                         for k, v in json.loads(names_json).items()}

    # Bridge edges
    logger.info(f"  Computing bridge edges for {n_clusters} clusters...")
    edge_pairs, edge_paths_2d = compute_bridge_edges(
        coords, embeddings, labels, n_clusters)

    # Generate narration
    titles = data['fields'].get('title')
    if titles is None:
        titles = [f"Item {i}" for i in range(n)]
    narration = _generate_narration(
        cluster_names, titles, labels, coords, model=model, title=title,
        domain=domain)

    new_meta = {
        'edge_pairs': json.dumps(edge_pairs),
        'edge_paths_2d': json.dumps(edge_paths_2d),
        'tour_narration': json.dumps(
            {str(k): v for k, v in narration.items()}),
    }

    new_meta['_provenance_level_3'] = json.dumps(provenance_to_dict(
        create_provenance(
            artifact_type="dyf",
            n_items=n,
            source_paths=[str(dyf_path)],
            params={"cluster_level": cluster_level, "model": model},
        )
    ))

    out = output_path or dyf_path
    logger.info(f"  Writing enriched file: {out}")
    rewrite_lazy_index(dyf_path, new_metadata=new_meta, output_path=out)
    logger.info("  Done. Level 2 → 3")
