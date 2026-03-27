"""Level 0 → 1: UMAP projection enrichment."""

import json
import time
from pathlib import Path

import numpy as np

from dyf.lazy_index import LazyIndex, rewrite_lazy_index
from dyf.provenance import create_provenance, provenance_to_dict


def suggest_n_neighbors(embeddings, num_bits=12, min_k=15, max_k=100):
    """Use DYF LSH bucket density to suggest UMAP n_neighbors."""
    from dyf_rs import DensityClassifier

    clf = DensityClassifier(
        embedding_dim=embeddings.shape[1], num_bits=num_bits, seed=42)
    clf.fit(embeddings)
    bucket_sizes = clf.get_bucket_sizes()
    mean_size = bucket_sizes.mean()
    suggested = int(np.clip(mean_size, min_k, max_k))
    n_buckets = len(set(clf.get_bucket_ids()))
    print(f"  DYF: {n_buckets} buckets, mean_size={mean_size:.0f}, "
          f"suggested n_neighbors={suggested}")
    return suggested


def run_umap(embeddings, n_neighbors=15, n_components=3, densmap=False):
    """Run UMAP and return median-centered, MAD-scaled coords."""
    import umap
    from sklearn.neighbors import NearestNeighbors

    label = "densMAP" if densmap else "UMAP"
    print(f"  Running {label} (n_neighbors={n_neighbors}, {n_components}D)...")
    t0 = time.time()
    reducer = umap.UMAP(
        n_components=n_components,
        n_neighbors=n_neighbors,
        min_dist=0.1,
        densmap=densmap,
        n_jobs=-1,
        verbose=False,
        random_state=42,
    )
    coords = np.asarray(reducer.fit_transform(embeddings))

    nan_mask = np.isnan(coords).any(axis=1)
    if nan_mask.any():
        print(f"    Replacing {nan_mask.sum()} NaN coords")
        nn = NearestNeighbors(n_neighbors=1, metric='cosine')
        nn.fit(embeddings[~nan_mask])
        _, idx = nn.kneighbors(embeddings[nan_mask])
        coords[nan_mask] = coords[~nan_mask][idx.ravel()]

    median = np.nanmedian(coords, axis=0)
    mad = np.nanmedian(np.abs(coords - median), axis=0)
    scale = float(np.fmax(np.nanmax(mad), 1e-8))
    coords = (coords - median) / scale
    print(f"    Done in {time.time() - t0:.1f}s")
    return coords


def orient_landscape(coords):
    """Rotate XY plane so the widest spread aligns with the X axis."""
    xy = coords[:, :2]
    cov = np.cov(xy, rowvar=False)
    theta = 0.5 * np.arctan2(2 * cov[0, 1], cov[0, 0] - cov[1, 1])
    c, s = np.cos(-theta), np.sin(-theta)
    rot = xy @ np.array([[c, s], [-s, c]])
    if np.ptp(rot[:, 1]) > np.ptp(rot[:, 0]):
        c2, s2 = np.cos(np.pi / 2), np.sin(np.pi / 2)
        rot = rot @ np.array([[c2, s2], [-s2, c2]])
    out = coords.copy()
    out[:, :2] = rot
    xr = np.ptp(out[:, 0])
    yr = np.ptp(out[:, 1])
    print(f"    Landscape orient: rotated {np.degrees(theta):.1f}°, "
          f"spread X={xr:.2f} Y={yr:.2f} (ratio {xr / yr:.2f})")
    return out


def enrich_project(dyf_path, n_components=3, densmap=False, output_path=None,
                   fisher_col=None, fisher_parquet=None,
                   diagnose_parquet=None):
    """Add UMAP coordinates to a .dyf file (Level 0 → 1)."""
    print(f"\n=== Level 1: UMAP Projection ===")
    print(f"  Input: {dyf_path}")

    with LazyIndex(dyf_path) as idx:
        level = idx.detect_enrichment_level()
        if level >= 1:
            print(f"  Already at level {level} (has UMAP coords), skipping.")
            return
        n = idx.total_items
        print(f"  {n:,} items, dim={idx.embedding_dim}")

    # Extract embeddings
    with LazyIndex(dyf_path) as idx:
        data = idx.extract_all_fields()
    embeddings = data['embeddings']

    # Optional Fisher dimension weighting
    fisher_weights = None
    if fisher_col:
        from dyf.fisher import compute_fisher_weights, apply_fisher_weights
        from dyf.categorical import coarsen
        import polars as pl

        if fisher_parquet:
            df = pl.read_parquet(fisher_parquet)
            if fisher_col in df.columns:
                raw_vals = df[fisher_col].to_list()
            else:
                print(f"  WARNING: column '{fisher_col}' not in {fisher_parquet}, "
                      f"skipping Fisher weighting")
                raw_vals = None
        elif fisher_col in data.get('fields', {}):
            raw_vals = data['fields'][fisher_col]
        else:
            print(f"  WARNING: fisher_col='{fisher_col}' not found, "
                  f"skipping Fisher weighting")
            raw_vals = None

        if raw_vals is not None:
            fisher_labels = coarsen(raw_vals)
            fisher_weights = compute_fisher_weights(embeddings, fisher_labels)
            embeddings = apply_fisher_weights(embeddings, fisher_weights)
            print(f"  Fisher weighting applied ({fisher_col}): "
                  f"top-5 dims {np.argsort(fisher_weights)[-5:][::-1]}")

    # Optional axis diagnostics sanity check
    if diagnose_parquet:
        import polars as pl
        from dyf.categorical import discover_categorical_columns, diagnose_axes

        diag_path = Path(diagnose_parquet)
        if diag_path.exists():
            diag_df = pl.read_parquet(diag_path)
            label_cols = discover_categorical_columns(diag_df, text_col="text")
            if label_cols:
                diags = diagnose_axes(embeddings, label_cols)
                print(f"  Axis diagnostics ({len(diags)} axes):")
                for d in diags:
                    flag = " ⚠ UNDER-SERVED" if d.lift < 3.0 else ""
                    print(f"    {d.name}: lift={d.lift:.1f}x  "
                          f"purity={d.knn_purity:.3f}{flag}")
                under = [d for d in diags if d.lift < 3.0]
                if under:
                    print(f"  WARNING: {len(under)} axis(es) under-served. "
                          f"Consider re-embedding with --diagnose in gudid_embeddings.py")
        else:
            print(f"  WARNING: --diagnose-parquet={diag_path} not found, skipping")

    # Compute UMAP
    dyf_k = suggest_n_neighbors(embeddings)
    coords = run_umap(embeddings, n_neighbors=dyf_k,
                       n_components=n_components, densmap=densmap)
    coords = orient_landscape(coords)

    # Write back
    new_sf = {
        'umap_x': coords[:, 0].astype(np.float32),
        'umap_y': coords[:, 1].astype(np.float32),
        'umap_z': (coords[:, 2].astype(np.float32) if n_components >= 3
                   else np.zeros(len(coords), dtype=np.float32)),
    }
    new_meta = {
        'umap_n_neighbors': str(dyf_k),
        'umap_n_components': str(n_components),
        'umap_densmap': str(densmap).lower(),
    }
    if fisher_weights is not None:
        new_meta['fisher_col'] = fisher_col
        new_meta['fisher_weights'] = json.dumps(fisher_weights.tolist())
        from dyf.categorical import CategoryGraph, store_category_graph
        graph = CategoryGraph.from_single_level(fisher_labels)
        new_meta.update(store_category_graph(graph, fisher_col))

    # Stamp provenance for Level 1
    new_meta['_provenance_level_1'] = json.dumps(provenance_to_dict(
        create_provenance(
            artifact_type="dyf",
            n_items=len(embeddings),
            source_paths=[str(dyf_path)],
            params={"n_components": n_components, "densmap": densmap,
                    "fisher_col": fisher_col},
        )
    ))

    out = output_path or dyf_path
    print(f"  Writing enriched file: {out}")
    rewrite_lazy_index(dyf_path, new_stored_fields=new_sf,
                       new_metadata=new_meta, output_path=out)
    print(f"  Done. Level 0 → 1")
