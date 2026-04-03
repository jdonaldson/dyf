"""Re-run glyph annotations on existing cluster names."""

import json

import numpy as np

from dyf.lazy_index import LazyIndex, rewrite_lazy_index

from ._labeling import annotate_cluster_names, transfer_labels_majority_vote


def reannotate(dyf_path, output_path=None):
    """Re-run glyph annotations on existing cluster names without re-clustering."""
    print("\n=== Reannotate Cluster Glyphs ===")
    print(f"  Input: {dyf_path}")

    with LazyIndex(dyf_path) as idx:
        level = idx.detect_enrichment_level()
        if level < 2:
            print(f"  ERROR: Need level 2 (clusters), got level {level}.")
            return
        data = idx.extract_all_fields()

    label_cache_data = json.loads(data['metadata'].get('_label_cache', '{}'))
    if not label_cache_data:
        print("  ERROR: No _label_cache in metadata. Run 'cluster' first.")
        return

    embeddings = data['embeddings']

    # Discover cluster levels from stored fields
    cluster_ks = sorted({
        parts[1]
        for sf_name in data['fields']
        if sf_name.startswith('cluster_')
        for parts in [sf_name.split('_')]
        if len(parts) == 3 and parts[2] in ('2d', '3d')
    })

    new_meta = {}
    for target_k in cluster_ks:
        cache_key_2d = f"cluster_{target_k}_2d"
        raw_2d = label_cache_data.get(cache_key_2d)
        if raw_2d is None:
            raw_2d = label_cache_data.get(f"cluster_{target_k}")
        if raw_2d is None:
            print(f"  Skipping k={target_k}: no raw names in cache")
            continue
        raw_2d = {int(k): v for k, v in raw_2d.items()}

        sf_2d = f'cluster_{target_k}_2d'
        if sf_2d in data['fields']:
            labels_2d = data['fields'][sf_2d].astype(np.int32)
            print(f"  Reannotating {sf_2d}...")
            ann_2d, gly_2d = annotate_cluster_names(
                raw_2d, labels_2d, embeddings)
            new_meta[f'cluster_names_{target_k}_2d'] = json.dumps(
                {str(k): v for k, v in ann_2d.items()})
            new_meta[f'cluster_glyphs_{target_k}_2d'] = json.dumps(
                {str(k): v for k, v in gly_2d.items()})

        sf_3d = f'cluster_{target_k}_3d'
        if sf_3d in data['fields'] and sf_2d in data['fields']:
            labels_3d = data['fields'][sf_3d].astype(np.int32)
            raw_3d = transfer_labels_majority_vote(
                labels_2d, raw_2d, labels_3d)
            print(f"  Reannotating {sf_3d}...")
            ann_3d, gly_3d = annotate_cluster_names(
                raw_3d, labels_3d, embeddings)
            new_meta[f'cluster_names_{target_k}_3d'] = json.dumps(
                {str(k): v for k, v in ann_3d.items()})
            new_meta[f'cluster_glyphs_{target_k}_3d'] = json.dumps(
                {str(k): v for k, v in gly_3d.items()})

    out = output_path or dyf_path
    print(f"\n  Writing: {out}")
    rewrite_lazy_index(dyf_path, new_metadata=new_meta, output_path=out)
    print("  Done.")
