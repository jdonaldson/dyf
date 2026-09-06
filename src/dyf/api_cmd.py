"""`dyf api` — print the grouped map of dyf's public Python API.

The map itself lives in `_api_map.py`; this exposes it to anyone driving dyf as a tool
rather than importing it. Without this you have to write Python to find out what writing
Python against dyf would give you, which is a poor trade for a caller deciding whether
the package is worth importing at all.

Same `schema_version` contract as `dyf info`: version 0, unstable before v1.
"""

from __future__ import annotations

import argparse
import json
import logging

from ._api_map import API_GROUPS, NOT_REEXPORTED, overview

logger = logging.getLogger(__name__)

SCHEMA_VERSION = 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="dyf api",
        description="Print the grouped map of dyf's public Python API.",
    )
    parser.add_argument(
        "group",
        nargs="?",
        help=f"Show only this group. One of: {', '.join(sorted(API_GROUPS))}",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        dest="as_json",
        help="Emit JSON (schema_version 0 — unstable before v1)",
    )
    args = parser.parse_args(argv)

    if args.group is not None and args.group not in API_GROUPS:
        logger.error("unknown group: %s", args.group)
        logger.error("  known groups: %s", ", ".join(sorted(API_GROUPS)))
        return 2

    if args.as_json:
        payload = {
            "schema_version": SCHEMA_VERSION,
            "groups": overview(args.group, as_dict=True),
            "not_reexported": NOT_REEXPORTED,
        }
        print(json.dumps(payload, indent=2))
        return 0

    logger.info("%s", overview(args.group))
    return 0
