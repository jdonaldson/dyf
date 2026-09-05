"""`dyf info` — describe a .dyf artifact without loading it.

The cheap index over an expensive body. `LazyIndex` opens in ~5 ms without touching
Arrow data, and already knows everything a caller needs in order to decide what to do
next: how many items, what dimensionality, which stored fields exist, and how far the
enrichment pipeline has been run. Before this command the only way to ask those
questions was to write Python, which is a poor deal for anything driving dyf as a tool.

Output is deliberately available in two shapes:

* human — aligned key/value lines, for reading
* ``--json`` — a ``schema_version`` envelope, for parsing

**The JSON schema is version 0 and carries no compatibility guarantee before v1.**
That is the whole point of stamping the version: callers get something parseable now,
and the project stays free to change the shape while the underlying values are still
being validated. See AGENT_LEGIBILITY_TODO.md.

Fields whose values are known to be wrong are omitted rather than serialized. Emitting a
field an agent will trust, whose value is meaningless, is worse than emitting nothing.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
from typing import Any

logger = logging.getLogger(__name__)

SCHEMA_VERSION = 0

ENRICHMENT_LEVELS = {
    0: "base (embeddings + tree only)",
    1: "projected (umap_x/y/z)",
    2: "clustered (community_id / cluster_*)",
    3: "viz-ready (edge_pairs / tour_narration)",
}

# Metadata keys that are structural rather than descriptive. They are reported as
# presence/shape instead of dumped verbatim: several hold large JSON blobs (dendrograms,
# edge lists, narration) that would bury the summary this command exists to give.
_BULK_METADATA_KEYS = {
    "stored_fields",
    "louvain_dendrogram",
    "edge_pairs",
    "edge_paths_2d",
    "tour_narration",
}


def collect_info(path: str) -> dict[str, Any]:
    """Gather a summary of a .dyf file. Returns the payload used by both output modes.

    Opens the index lazily — no embedding or stored-field data is read.
    """
    from .lazy_index import LazyIndex

    info: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "path": os.path.abspath(path),
        "file_size_bytes": os.path.getsize(path),
    }

    with LazyIndex(path) as idx:
        summary = idx.tree_summary
        info["format_version"] = idx.format_version
        info["embedding_dim"] = summary.get("embedding_dim")
        info["total_items"] = summary.get("total_items")
        info["num_leaves"] = summary.get("num_leaves")
        info["num_nodes"] = summary.get("num_nodes")
        info["build_params"] = summary.get("build_params")
        if "pq" in summary:
            info["pq"] = summary["pq"]

        info["stored_fields"] = sorted(idx.stored_field_names)

        level = idx.detect_enrichment_level()
        info["enrichment_level"] = level
        info["enrichment_label"] = ENRICHMENT_LEVELS.get(level, "unknown")

        meta = dict(idx._get_metadata())
        info["domain"] = meta.get("domain")

        # Provenance is stamped per enrichment stage as `_provenance_level_N`. Report
        # which stages left a record; the records themselves are nested JSON strings.
        provenance = {}
        for key in sorted(k for k in meta if k.startswith("_provenance_level_")):
            stage = key.rsplit("_", 1)[-1]
            try:
                provenance[stage] = json.loads(meta[key])
            except (ValueError, TypeError):
                provenance[stage] = {"unparseable": True}
        info["provenance"] = provenance

        info["metadata_keys"] = sorted(meta)
        info["metadata"] = {
            k: v
            for k, v in sorted(meta.items())
            if k not in _BULK_METADATA_KEYS and not k.startswith("_provenance_level_")
        }

    return info


def _format_human(info: dict[str, Any]) -> str:
    """Render the payload as aligned key/value lines."""
    lines: list[str] = []
    size_mb = info["file_size_bytes"] / 1_048_576

    lines.append(info["path"])
    lines.append("")
    lines.append(f"  items            {info['total_items']:,}")
    lines.append(f"  dim              {info['embedding_dim']}")
    lines.append(f"  leaves / nodes   {info['num_leaves']:,} / {info['num_nodes']:,}")
    lines.append(f"  file size        {size_mb:,.1f} MB")
    lines.append(f"  format version   {info['format_version']}")
    if info.get("domain"):
        lines.append(f"  domain           {info['domain']}")

    lines.append("")
    lines.append(f"  enrichment       level {info['enrichment_level']} — {info['enrichment_label']}")
    if info["provenance"]:
        stages = ", ".join(sorted(info["provenance"]))
        lines.append(f"  provenance       stages {stages}")
    else:
        lines.append("  provenance       none recorded")

    bp = info.get("build_params")
    if bp:
        lines.append("")
        lines.append("  build params")
        for key in ("max_depth", "num_bits", "min_leaf_size", "seed", "quantization", "compression"):
            if bp.get(key) is not None:
                lines.append(f"    {key:<16} {bp[key]}")

    if info.get("pq"):
        lines.append("")
        lines.append("  product quantization")
        for key, value in info["pq"].items():
            lines.append(f"    {key:<16} {value}")

    lines.append("")
    if info["stored_fields"]:
        lines.append(f"  stored fields ({len(info['stored_fields'])})")
        for name in info["stored_fields"]:
            lines.append(f"    {name}")
    else:
        lines.append("  stored fields    none")

    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="dyf info",
        description="Describe a .dyf file without loading its data.",
    )
    parser.add_argument("path", help="Path to a .dyf file")
    parser.add_argument(
        "--json",
        action="store_true",
        dest="as_json",
        help="Emit JSON (schema_version 0 — unstable before v1)",
    )
    args = parser.parse_args(argv)

    if not os.path.exists(args.path):
        logger.error("no such file: %s", args.path)
        return 2
    if os.path.isdir(args.path):
        logger.error("%s is a directory — pass a .dyf file", args.path)
        return 2

    try:
        info = collect_info(args.path)
    except ImportError as exc:
        # Reading a .dyf needs the [lazy] extra. Say so, rather than raising a
        # ModuleNotFoundError traceback the caller has to interpret.
        logger.error("%s", exc)
        logger.error("reading .dyf files requires: pip install 'dyf[lazy]'")
        return 3
    except Exception as exc:
        logger.error("could not read %s as a .dyf index: %s", args.path, exc)
        return 1

    if args.as_json:
        print(json.dumps(info, indent=2, default=str))
    else:
        logger.info("%s", _format_human(info))
    return 0
