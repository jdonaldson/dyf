"""Split keyword enrichment for the DYF tree."""

import json

import numpy as np

from dyf.lazy_index import LazyIndex, rewrite_lazy_index


def enrich_splits(
    dyf_path,
    max_depth=3,
    bigram_check=False,
    output_path=None,
    domain_threshold=0.10,
    min_child_items=50,
    use_embeddings=True,
):
    """Compute tree split keywords and store in .dyf metadata."""
    from dyf.splits import (
        build_tree_maps,
        compute_domain_stopwords,
        compute_embedding_keywords,
        compute_split_keywords,
    )

    print("\n=== Split Keywords ===")
    print(f"  Input: {dyf_path}")

    hyperplanes = None
    with LazyIndex(dyf_path) as idx:
        tree, children_map, leaf_batches = build_tree_maps(idx)
        if use_embeddings:
            hyperplanes = idx.get_split_hyperplanes()

    with LazyIndex(dyf_path) as idx:
        data = idx.extract_all_fields()
    titles = data["fields"].get("title")
    if titles is None:
        titles = [f"Item {i}" for i in range(len(data["embeddings"]))]
    if isinstance(titles, np.ndarray):
        titles = titles.tolist()
    embeddings = data["embeddings"]

    n = len(titles)
    print(f"  {n:,} items, tree has {len(tree)} nodes")

    domain_sw = compute_domain_stopwords(titles, threshold=domain_threshold)
    print(f"  {len(domain_sw)} domain stop words (e.g. {sorted(domain_sw)[:5]})")

    if use_embeddings and hyperplanes:
        print(f"  Using embedding-space projection ({len(hyperplanes)} nodes with hyperplanes)")
        result = compute_embedding_keywords(
            titles,
            embeddings,
            tree,
            leaf_batches,
            children_map,
            hyperplanes,
            max_depth_from_root=max_depth,
            min_child_items=min_child_items,
            domain_stopwords=domain_sw,
        )
    else:
        if use_embeddings:
            print("  No hyperplanes found, falling back to TF-IDF")
        print("  Using TF-IDF keyword extraction")
        result = compute_split_keywords(
            titles,
            tree,
            leaf_batches,
            children_map,
            max_depth_from_root=max_depth,
            min_child_items=min_child_items,
            domain_stopwords=domain_sw,
            bigram_check=bigram_check,
        )

    n_splits = len(result["splits"])
    print(f"  Computed keywords for {n_splits} splits (depth 0-{max_depth - 1})")

    if bigram_check:
        needed = sum(1 for s in result["splits"].values() if s.get("bigram_needed"))
        print(f"  Bigram needed: {needed}/{n_splits} splits")

    serializable = {
        "domain_stopwords": result["domain_stopwords"],
        "splits": {},
    }
    for nid, split in result["splits"].items():
        s = {
            "depth": split["depth"],
            "children": {},
        }
        if "bigram_needed" in split:
            s["bigram_needed"] = split["bigram_needed"]
        for cid, cinfo in split["children"].items():
            entry = {
                "count": cinfo["count"],
                "unigrams": cinfo["unigrams"],
            }
            if "bigrams" in cinfo:
                entry["bigrams"] = cinfo["bigrams"]
            s["children"][str(cid)] = entry
        serializable["splits"][str(nid)] = s

    new_meta = {
        "split_keywords": json.dumps(serializable),
    }

    out = output_path or dyf_path
    print(f"  Writing enriched file: {out}")
    rewrite_lazy_index(dyf_path, new_metadata=new_meta, output_path=out)
    print("  Done.")
