"""Bottom-up tree labeling via LLM."""

import json
import logging
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from dyf.lazy_index import LazyIndex, rewrite_lazy_index

from ._ollama import _call_ollama

logger = logging.getLogger(__name__)


@dataclass
class _LabelingConfig:
    """Configuration for tree branch labeling."""
    model: str
    rng: np.random.Generator
    samples_per_child: int
    min_child_size: int


def _collect_descendant_indices(node_id, children_of, leaf_batches):
    """Recursively collect all item indices under a node."""
    if node_id in leaf_batches:
        return leaf_batches[node_id]
    kids = children_of.get(node_id, [])
    if not kids:
        return np.array([], dtype=int)
    return np.concatenate([
        _collect_descendant_indices(k, children_of, leaf_batches)
        for k in kids
    ])


def _label_branch(node, children_of, by_id, leaf_batches, titles,
                   config: _LabelingConfig):
    """Label a single tree branch by sampling child groups and querying the LLM.

    Returns (node_id, branch_label, kid_labels_dict).
    """
    nid = node['node_id']
    kids = children_of[nid]
    kids_sorted = sorted(kids, key=lambda k: -by_id[k]['num_items'])

    child_samples = {}
    for kid_id in kids_sorted:
        kn = by_id[kid_id]
        if kn['num_items'] < config.min_child_size:
            continue
        all_idx = _collect_descendant_indices(
            kid_id, children_of, leaf_batches)
        if len(all_idx) < 3:
            continue

        n_sample = min(config.samples_per_child, len(all_idx))
        sample_idx = config.rng.choice(all_idx, size=n_sample, replace=False)
        sample_titles = [titles[j][:120] for j in sample_idx]

        seen = set()
        unique = []
        for t in sample_titles:
            if t not in seen:
                seen.add(t)
                unique.append(t)
        child_samples[kid_id] = (kn['num_items'], unique)

    if not child_samples:
        return nid, f"Group {nid}", {}

    prompt_parts = [
        "You are labeling groups in an embedding-space tree. Below are "
        "sample titles from each child group within a branch.\n"
    ]
    child_ids_ordered = list(child_samples.keys())
    for i, kid_id in enumerate(child_ids_ordered):
        count, samples = child_samples[kid_id]
        prompt_parts.append(f"--- Group {i+1} ({count} items) ---")
        for t in samples:
            prompt_parts.append(f"  - {t}")
        prompt_parts.append("")

    prompt_parts.append(
        "For each group, give a SHORT (2-5 word) descriptive label that "
        "captures what makes it distinctive FROM THE OTHER GROUPS.\n"
        "Then give ONE summary label (2-5 words) for the entire branch.\n\n"
        "BAD labels (too vague): 'General Items', 'Miscellaneous', "
        "'Various Topics'\n"
        "GOOD labels (specific): 'Marine Biology', 'European Monarchs', "
        "'Spinal Fixation Screws', 'Video Game Consoles'\n\n"
        "Reply in this EXACT format (one line per group, then summary):\n"
    )
    for i in range(len(child_ids_ordered)):
        prompt_parts.append(f"Group {i+1}: <label>")
    prompt_parts.append("Branch: <summary label>")

    prompt = "\n".join(prompt_parts)
    response = _call_ollama(config.model, prompt)

    kid_labels = {}
    branch_label = f"Branch {nid}"
    for line in response.split('\n'):
        line = line.strip()
        if not line:
            continue
        if line.lower().startswith('branch:'):
            branch_label = line.split(':', 1)[1].strip().strip('"\'')[:50]
        else:
            for i, kid_id in enumerate(child_ids_ordered):
                prefix = f"Group {i+1}:"
                if line.startswith(prefix) or line.lower().startswith(
                        prefix.lower()):
                    label = line.split(':', 1)[1].strip().strip('"\'')[:50]
                    kid_labels[kid_id] = label
                    break

    for kid_id in child_ids_ordered:
        if kid_id not in kid_labels:
            kid_labels[kid_id] = f"Subgroup {kid_id}"

    return nid, branch_label, kid_labels


def label_tree_bottomup(idx, titles, model="gpt-oss:20b", target_depth=3,
                        samples_per_child=8, min_child_size=20,
                        cache_file=None, cache_data=None):
    """Label tree nodes bottom-up using the DYF tree hierarchy."""

    tree = idx.get_tree_structure()
    by_id = {n['node_id']: n for n in tree}

    children_of = defaultdict(list)
    for n in tree:
        if n['parent_id'] is not None:
            children_of[n['parent_id']].append(n['node_id'])

    leaf_batches = {}
    for n in tree:
        if n['is_leaf'] and n['batch_index'] >= 0:
            batch = idx.get_leaf(n['batch_index'])
            leaf_batches[n['node_id']] = batch.column('item_index').to_numpy()

    target_nodes = [n for n in tree if n['depth'] == target_depth]
    target_nodes.sort(key=lambda n: -n['num_items'])

    logger.info(f"  Tree labeling: {len(target_nodes)} branches at depth {target_depth}")

    # Check cache
    _cache_key = f"tree_depth_{target_depth}"
    if cache_data is not None:
        cached = cache_data.get(_cache_key, {})
        if cached.get("branch_labels") and cached.get("child_labels"):
            logger.info("  Loaded tree labels from cache")
            return cached
    elif cache_file:
        cache_path = Path(cache_file)
        if cache_path.exists():
            file_cache = json.loads(cache_path.read_text())
            cached = file_cache.get(_cache_key, {})
            if cached.get("branch_labels") and cached.get("child_labels"):
                logger.info("  Loaded tree labels from cache")
                return cached

    config = _LabelingConfig(
        model=model,
        rng=np.random.default_rng(42),
        samples_per_child=samples_per_child,
        min_child_size=min_child_size,
    )
    branch_labels = {}
    child_labels = {}
    hierarchy = {}

    for i, node in enumerate(target_nodes):
        nid, b_label, k_labels = _label_branch(
            node, children_of, by_id, leaf_batches, titles, config)
        branch_labels[nid] = b_label
        child_labels.update(k_labels)
        hierarchy[nid] = sorted(k_labels.keys())

        n_kids = len(k_labels)
        logger.info(f"    [{i+1}/{len(target_nodes)}] {b_label:<35s} "
                    f"({node['num_items']:>6,} items, {n_kids} children)")
        for kid_id, kid_label in sorted(k_labels.items(),
                                         key=lambda x: -by_id[x[0]]['num_items']):
            kn = by_id[kid_id]
            logger.debug(f"      └─ {kid_label:<30s} ({kn['num_items']:>5,} items)")

    result = {
        'branch_labels': {str(k): v for k, v in branch_labels.items()},
        'child_labels': {str(k): v for k, v in child_labels.items()},
        'hierarchy': {str(k): v for k, v in hierarchy.items()},
    }

    if cache_file and cache_data is None:
        cache_path = Path(cache_file)
        file_cache = {}
        if cache_path.exists():
            file_cache = json.loads(cache_path.read_text())
        file_cache[_cache_key] = result
        cache_path.write_text(json.dumps(file_cache, indent=2))
        logger.info(f"  Saved tree labels to cache ({cache_file})")

    return result


def enrich_tree(dyf_path, model="gpt-oss:20b", target_depth=3,
                samples_per_child=8, output_path=None):
    """Add tree-based hierarchical labels to a .dyf file."""
    logger.info(f"\n=== Tree Labeling (depth={target_depth}) ===")
    logger.info(f"  Input: {dyf_path}")

    with LazyIndex(dyf_path) as idx:
        sf_names = idx.stored_field_names
        if 'title' not in sf_names:
            logger.warning("  No 'title' stored field found.")
            return

        data = idx.extract_all_fields()
        titles = data['fields']['title']

        label_cache_data = json.loads(data['metadata'].get('_label_cache', '{}'))
        if label_cache_data:
            logger.info(f"  Loaded {len(label_cache_data)} label cache entries from .dyf")

        result = label_tree_bottomup(
            idx, titles, model=model, target_depth=target_depth,
            samples_per_child=samples_per_child,
            cache_data=label_cache_data)
        label_cache_data[f"tree_depth_{target_depth}"] = result

    new_meta = {
        f'tree_labels_depth_{target_depth}': json.dumps(result),
        '_label_cache': json.dumps(label_cache_data),
    }

    out = output_path or dyf_path
    logger.info(f"\n  Writing labels to: {out}")
    rewrite_lazy_index(dyf_path, new_metadata=new_meta, output_path=out)
    logger.info(f"  Done. Tree labels at depth {target_depth} stored.")
