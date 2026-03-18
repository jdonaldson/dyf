"""Level 3 → 3+: Pre-compute LLM scaffold from existing enrichment data.

Reads louvain_dendrogram, tour_narration, and embedding centroids to produce
a structured scaffold suitable for small LLMs. No LLM calls needed — this is
pure computation over existing metadata.

Stored as metadata key 'llm_scaffold'.
"""

import json

import numpy as np


def _group_label_from_names(member_names, member_sizes=None):
    """Infer a domain label from community names using keyword matching.

    If member_sizes is provided, weight keyword matches by community size
    to avoid a single small member dominating the label.
    """
    domain_keywords = {
        "spine": ["spine", "spinal", "pedicle", "interbody"],
        "dental": ["dental", "orthodontic", "tooth", "jaw"],
        "orthopedic": ["orthopedic", "orthopaedic", "bone", "joint"],
        "lower extremity": ["limb", "ankle", "foot", "orthoses", "afo"],
        "sensory/protective": ["eye", "hearing", "ear", "protection"],
        "cardiac/perfusion": ["perfusion", "cardiac", "heart"],
        "compression": ["compression", "support", "garment"],
    }
    brand_words = {"cardinal", "medline", "razek", "health", "baxter"}

    if member_sizes is None:
        member_sizes = [1] * len(member_names)

    total_size = sum(member_sizes) or 1

    # Score each domain by the fraction of items it covers
    domain_scores = {}
    for domain, kws in domain_keywords.items():
        score = 0
        for name, size in zip(member_names, member_sizes):
            if any(kw in name.lower() for kw in kws):
                score += size
        if score > 0:
            domain_scores[domain] = score / total_size

    # Check for brand/supplier
    brand_score = 0
    for name, size in zip(member_names, member_sizes):
        if any(b in name.lower() for b in brand_words):
            brand_score += size
    has_brand = brand_score / total_size > 0.15

    # Detect trial/sizing
    has_trial = any("trial" in n.lower() for n in member_names)

    if has_trial:
        return "sizing + final product"

    # Filter to domains covering >20% of items
    significant = {d: s for d, s in domain_scores.items() if s > 0.2}

    if has_brand and not significant:
        return "supplier brands"
    if has_brand and significant:
        top = max(significant, key=significant.get)
        return f"{top} (supplier)"
    if len(significant) == 1:
        return list(significant.keys())[0]
    if len(significant) >= 2:
        top2 = sorted(significant, key=significant.get, reverse=True)[:2]
        return " + ".join(top2)

    # Fallback: check if mostly "surgical" or "instrument" keywords
    surgical_score = sum(
        size for name, size in zip(member_names, member_sizes)
        if any(w in name.lower() for w in ["surgical", "instrument", "cannulated"])
    ) / total_size
    if surgical_score > 0.5:
        return "general surgical"

    return None


def _build_dendrogram_groups(Z, cids, names, sizes, max_group_size=4):  # noqa: N803
    """Walk the agglomerative dendrogram and emit labeled groups."""
    n = len(cids)
    clusters = {i: [cids[i]] for i in range(n)}
    groups = []

    for i, row in enumerate(Z):
        a, b = int(row[0]), int(row[1])
        new_id = n + i
        ma = clusters.get(a, [])
        mb = clusters.get(b, [])
        members = ma + mb
        clusters[new_id] = members

        if len(members) > max_group_size:
            continue

        member_names = [names[c] for c in members]
        member_sizes = [sizes[c] for c in members]
        label = _group_label_from_names(member_names, member_sizes)
        groups.append({
            "members": members,
            "label": label,
            "merge_distance": float(row[2]),
        })

    return groups


def _build_use_case_bundles(Z, cids, names, sizes):
    """Build larger super-groups from dendrogram as pre-aggregated bundles.

    These are the multi-hop aggregations that small models struggle with:
    "everything for spine surgery" = [10] + [14] + [15] + [20] etc.

    Strategy: walk the full dendrogram and emit every merge of 3-8 members.
    Label each bundle by the largest member's name (most recognizable).
    Deduplicate by keeping only the largest bundle per branch.
    """
    n = len(cids)
    clusters = {i: [cids[i]] for i in range(n)}
    bundles = []

    for i, row in enumerate(Z):
        a, b = int(row[0]), int(row[1])
        new_id = n + i
        ma = clusters.get(a, [])
        mb = clusters.get(b, [])
        members = ma + mb
        clusters[new_id] = members

        # Emit bundles for groups of 2-8 members
        if len(members) < 2 or len(members) > 8:
            continue

        total_items = sum(sizes[c] for c in members)
        # Label by domain keywords, fall back to largest member name
        member_names = [names[c] for c in members]
        member_sizes = [sizes[c] for c in members]
        label = _group_label_from_names(member_names, member_sizes)
        if not label:
            biggest = max(members, key=lambda c: sizes[c])
            label = names[biggest].split("(")[0].strip()

        bundles.append({
            "members": members,
            "label": label,
            "total_items": total_items,
            "merge_distance": float(row[2]),
            "member_details": [
                {"id": c, "name": names[c], "size": sizes[c]}
                for c in sorted(members, key=lambda c: sizes[c], reverse=True)
            ],
        })

    # Keep only the largest (most members) bundle per branch.
    # A bundle is subsumed if another bundle fully contains it.
    final = []
    for b in bundles:
        member_set = set(b["members"])
        subsumed = any(
            set(other["members"]) > member_set
            for other in bundles if other is not b
        )
        if not subsumed:
            final.append(b)

    # Sort by total items descending
    final.sort(key=lambda b: b["total_items"], reverse=True)
    return final


def _compute_similarity_signals(centroids, cids):
    """Compute pairwise similarity, top pairs, and outliers."""
    vecs = {}
    for c in cids:
        v = np.array(centroids[c])
        norm = np.linalg.norm(v)
        vecs[c] = v / norm if norm > 0 else v

    # All pairwise similarities
    pairs = []
    for i, a in enumerate(cids):
        for b in cids[i + 1:]:
            sim = float(np.dot(vecs[a], vecs[b]))
            pairs.append({"a": a, "b": b, "sim": round(sim, 3)})
    pairs.sort(key=lambda p: p["sim"], reverse=True)

    # Average similarity per community (for outlier detection)
    avg_sim = {}
    for c in cids:
        sims = [float(np.dot(vecs[c], vecs[o])) for o in cids if o != c]
        avg_sim[c] = round(float(np.mean(sims)), 3)

    outliers = sorted(cids, key=lambda c: avg_sim[c])

    return {
        "top_pairs": pairs[:8],
        "bottom_pairs": pairs[-3:],
        "avg_similarity": avg_sim,
        "most_distinct": outliers[:3],
        "most_central": outliers[-3:],
    }


def _find_structural_analogies(groups):
    """Find pairs of groups with similar merge distances in different branches."""
    analogies = []
    for i, g1 in enumerate(groups):
        if len(g1["members"]) != 2:
            continue
        for g2 in groups[i + 1:]:
            if len(g2["members"]) != 2:
                continue
            # No overlapping members
            if set(g1["members"]) & set(g2["members"]):
                continue
            # Similar merge distance = similar structural role
            d1, d2 = g1["merge_distance"], g2["merge_distance"]
            if abs(d1 - d2) < 0.06:
                analogies.append({
                    "pair_a": g1["members"],
                    "pair_b": g2["members"],
                    "label_a": g1["label"],
                    "label_b": g2["label"],
                    "distance_a": round(d1, 3),
                    "distance_b": round(d2, 3),
                })
    return analogies


def compute_scaffold(dyf_path):
    """Compute the full scaffold data structure from a .dyf file.

    Returns a dict that can be JSON-serialized and stored as metadata.
    """
    from dyf.lazy_index import LazyIndex

    idx = LazyIndex(dyf_path)
    fields = idx.extract_all_fields()
    meta = fields["metadata"]
    ld = json.loads(meta["louvain_dendrogram"])

    names = ld["community_names"]
    sizes = ld["community_sizes"]
    centroids = ld["community_embedding_centroids"]
    Z = ld["Z"]

    cids = sorted(names.keys(), key=int)
    n_items = sum(sizes[c] for c in cids)

    # Size ranking
    by_size = sorted(cids, key=lambda c: sizes[c], reverse=True)

    # Groups from dendrogram
    groups = _build_dendrogram_groups(Z, cids, names, sizes)

    # Use-case bundles (pre-aggregated multi-hop recommendations)
    bundles = _build_use_case_bundles(Z, cids, names, sizes)

    # Similarity signals
    sim_signals = _compute_similarity_signals(centroids, cids)

    # Structural analogies
    analogies = _find_structural_analogies(groups)

    # Narration (if available)
    narr = json.loads(meta.get("tour_narration", "{}"))
    descriptions = {}
    for c in cids:
        if c in narr:
            first_sentence = narr[c].split(".")[0] + "."
            # Skip trivial "Name." descriptions
            if first_sentence.strip().rstrip(".") != names[c].strip():
                if len(first_sentence) > 30:
                    descriptions[c] = first_sentence[:150]

    return {
        "n_items": n_items,
        "n_communities": len(cids),
        "communities": {c: {"name": names[c], "size": sizes[c]} for c in cids},
        "size_ranking": by_size,
        "largest": by_size[0],
        "smallest": by_size[-1],
        "groups": [
            {"members": g["members"], "label": g["label"]}
            for g in groups if len(g["members"]) >= 2
        ],
        "similarity": sim_signals,
        "analogies": analogies,
        "bundles": bundles,
        "descriptions": descriptions,
    }


def render_scaffold(scaffold_data):
    """Render a scaffold dict into a compact text string for LLM consumption."""
    d = scaffold_data
    names = {c: d["communities"][c]["name"] for c in d["communities"]}
    sizes = {c: d["communities"][c]["size"] for c in d["communities"]}

    lines = [f"{d['n_items']:,} items, {d['n_communities']} communities"]

    # Explicit extremes
    lg, sm = d["largest"], d["smallest"]
    lines.append("")
    lines.append(f"Largest: [{lg}] {names[lg]} ({sizes[lg]:,})")
    lines.append(f"Smallest: [{sm}] {names[sm]} ({sizes[sm]:,})")

    # All communities sorted by size
    lines.append("")
    lines.append("Communities (largest first):")
    for c in d["size_ranking"]:
        lines.append(f"  [{c}] {names[c]} ({sizes[c]:,})")

    # Groups
    groups = d["groups"]
    if groups:
        lines.append("")
        lines.append("Groups:")
        for g in groups:
            ids = ", ".join(f"[{c}]" for c in g["members"])
            tag = f" — {g['label']}" if g["label"] else ""
            lines.append(f"  {ids}{tag}")

    # Similarity
    sim = d["similarity"]
    if sim["top_pairs"]:
        lines.append("")
        lines.append("Most similar:")
        for p in sim["top_pairs"][:6]:
            lines.append(
                f"  [{p['a']}] {names[p['a']]} <-> "
                f"[{p['b']}] {names[p['b']]} ({p['sim']:.2f})")

    if sim["most_distinct"]:
        lines.append("")
        lines.append("Most distinct:")
        for c in sim["most_distinct"]:
            lines.append(
                f"  [{c}] {names[c]} (avg sim {sim['avg_similarity'][c]:.2f})")

    # Use-case bundles (pre-aggregated recommendations)
    if d.get("bundles"):
        lines.append("")
        lines.append("Use-case bundles (pre-built recommendations):")
        for b in d["bundles"]:
            ids = ", ".join(f"[{m['id']}]" for m in b["member_details"])
            lines.append(f"  {b['label']}: {ids} ({b['total_items']:,} items total)")

    # Structural analogies
    if d["analogies"]:
        lines.append("")
        lines.append("Structural analogies (pairs that play similar roles):")
        for a in d["analogies"][:5]:
            pa = ", ".join(f"[{c}]" for c in a["pair_a"])
            pb = ", ".join(f"[{c}]" for c in a["pair_b"])
            la = a["label_a"] or "related"
            lb = a["label_b"] or "related"
            lines.append(f"  {pa} ({la}) ~ {pb} ({lb})")

    # Descriptions
    if d["descriptions"]:
        lines.append("")
        lines.append("Descriptions:")
        for c in d["size_ranking"]:
            if c in d["descriptions"]:
                lines.append(f"  [{c}]: {d['descriptions'][c][:120]}")

    return "\n".join(lines)


def enrich_scaffold(dyf_path, output_path=None):
    """Compute and store the LLM scaffold in a .dyf file."""
    from dyf.lazy_index import LazyIndex, rewrite_lazy_index

    print(f"Computing LLM scaffold for {dyf_path}...")

    scaffold_data = compute_scaffold(dyf_path)
    scaffold_text = render_scaffold(scaffold_data)

    print(f"  Scaffold: {len(scaffold_data['communities'])} communities, "
          f"{len(scaffold_data['groups'])} groups, "
          f"{len(scaffold_data['analogies'])} analogies")
    print(f"  Rendered: {len(scaffold_text)} chars (~{len(scaffold_text)//4} tokens)")

    # Store both structured data and rendered text
    out = output_path or dyf_path
    rewrite_lazy_index(
        dyf_path,
        new_metadata={
            "llm_scaffold": json.dumps(scaffold_data),
            "llm_scaffold_text": scaffold_text,
        },
        output_path=out,
    )
    print(f"  Written to {out}")
