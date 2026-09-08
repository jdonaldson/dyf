"""The shared tail of every `dyf index-*` command, and its shared CLI flags.

The three ingest modules differ entirely in their *front* — tree-sitter parsing, vision
embedding, scene detection — and were identical in their *back*: normalize, build a tree
with the same parameters, write a `.dyf` with the same compression and quantization. That
tail was copy-pasted three times, along with three separately-maintained argparse blocks
carrying the same flags.

The copy-paste had already cost something concrete rather than merely being untidy:
**`--dedup` existed only for source code.** Near-identical video keyframes are the textbook
case for collapsing duplicates — the README's own table measures 88.3% duplicates on
adjacent-frame data — and images from a burst shoot are nearly as good a case. Neither
could use it, because the flag lived in one of the three copies. Unifying the tail gives it
to all three at once, which is the point: the duplication was not a style problem, it was
a capability that could not spread.

What stays per-module is what actually differs: how to produce embeddings and stored
fields from a source, and the `metadata` that describes the medium.
"""

from __future__ import annotations

import json
import logging
import time
from collections.abc import Sequence
from pathlib import Path

import numpy as np

from .dyf_tree import build_dyf_tree
from .lazy_index import write_lazy_index
from .provenance import create_provenance, provenance_to_dict

logger = logging.getLogger(__name__)

#: Tree parameters shared by every ingest path. `itq` fitting is deliberate and was
#: identical in all three copies; it produces tighter partitions than raw PCA.
DEFAULT_MAX_DEPTH = 4
DEFAULT_NUM_BITS = 4
DEFAULT_MIN_LEAF_SIZE = 5
DEFAULT_SEED = 42
FIT_METHOD = "itq"


def add_common_index_args(parser, *, default_model: str) -> None:
    """Add the flags every `index-*` command shares.

    Defined once so the three commands cannot drift apart — which they already had, since
    `--dedup` reached only one of them.
    """
    parser.add_argument("-o", "--output", type=Path, default=None, help="Output .dyf path")
    parser.add_argument("--model", default=default_model, help=f"Embedding model (default: {default_model})")
    parser.add_argument("--max-depth", type=int, default=DEFAULT_MAX_DEPTH)
    parser.add_argument("--num-bits", type=int, default=DEFAULT_NUM_BITS)
    parser.add_argument("--min-leaf-size", type=int, default=DEFAULT_MIN_LEAF_SIZE)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument(
        "--dedup",
        type=float,
        nargs="?",
        const=0.99,
        default=None,
        metavar="COSINE",
        help=(
            "Collapse near-duplicates above this cosine before indexing (0.99 with no value). "
            "Measure first — duplicate rates are wildly corpus-dependent."
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Report what would be indexed and stop. Embeds nothing, writes nothing.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        dest="as_json",
        help="With --dry-run, emit JSON (schema_version 0 — unstable before v1)",
    )


def ingest_params(
    *,
    max_depth: int = DEFAULT_MAX_DEPTH,
    num_bits: int = DEFAULT_NUM_BITS,
    min_leaf_size: int = DEFAULT_MIN_LEAF_SIZE,
    seed: int = DEFAULT_SEED,
    dedup: float | None = None,
    embedding_model: str | None = None,
    domain: str | None = None,
) -> dict:
    """The exact dict `finalize_index` hashes into the provenance record.

    `Pipeline` decides "fresh" by comparing `params_hash(stage.params)` to the stamped
    hash byte for byte, so a stage that wraps an `index-*` command must declare the
    same dict — build it with this rather than guessing the keys. Only inputs go in:
    tree parameters, the dedup threshold, the fixed storage format, and the embedding
    model (which lives in the medium's metadata and would otherwise not invalidate).
    Outcomes such as the pre-dedup count do not, since a caller cannot know them ahead
    of the run and they would make "fresh" unreachable.
    """
    return {
        "max_depth": max_depth,
        "num_bits": num_bits,
        "min_leaf_size": min_leaf_size,
        "seed": seed,
        "fit_method": FIT_METHOD,
        "quantization": "float16",
        "compression": "none",
        "dedup": dedup,
        "embedding_model": embedding_model,
        "domain": domain,
    }


def finalize_index(
    embeddings: np.ndarray,
    output: Path,
    *,
    stored_fields: dict,
    metadata: dict,
    max_depth: int = DEFAULT_MAX_DEPTH,
    num_bits: int = DEFAULT_NUM_BITS,
    min_leaf_size: int = DEFAULT_MIN_LEAF_SIZE,
    seed: int = DEFAULT_SEED,
    dedup: float | None = None,
    source_paths: Sequence[str | Path] = (),
) -> None:
    """Dedup, normalize, build the tree, and write the `.dyf`, stamped with provenance.

    Everything after "I have embeddings and stored fields". Callers supply the medium's
    own `metadata`; the rest is identical by construction rather than by three people
    remembering to keep it identical.

    Args:
        embeddings: (n, dim) array. Normalized in place of the caller having to.
        output: Where to write.
        stored_fields: Parallel to `embeddings`; subset in lockstep when dedup runs, which
            is why dedup lives here rather than in each caller — subsetting one without
            the other silently mislabels every row.
        metadata: Medium-specific description, e.g. ``{"domain": "images"}``.
        dedup: Cosine threshold, or None to skip.
        source_paths: The files this index was built from. Hashed (size, mtime, first
            64 KB each) into the provenance record's ``source_hash`` so `Pipeline` can
            tell whether the inputs changed. Empty means "unknown inputs": the record is
            still written, but its source hash is a constant and cannot detect change.

    Provenance: the written file carries ``_provenance_level_0`` — level 0 of the same
    ladder the downstream `dyfviz` stages continue at 1, 2 and 3 — holding the item
    count after dedup, the source hash, and every parameter that shaped the output.
    Before 2026-09-07 nothing in dyf wrote provenance at all, so `Pipeline` reported
    every `.dyf` as ``stale (no provenance)`` and `dyf info` said ``none recorded``.
    """
    n_original = len(embeddings)

    if dedup is not None:
        from .dedup import dedup_for_index

        t0 = time.time()
        result = dedup_for_index(embeddings, stored_fields, threshold=dedup)
        embeddings, stored_fields = result.embeddings, result.stored_fields
        logger.info(
            f"Dedup at cosine > {dedup}: {n_original} -> {len(embeddings)} items "
            f"({result.removed_fraction:.1%} removed) in {time.time() - t0:.1f}s"
        )
        if not result.bookkeeping_added:
            logger.info("  No duplicates: omitted the orig_index/dup_members fields.")

    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    embeddings = embeddings / np.where(norms > 0, norms, 1)

    logger.info("Building DYF tree...")
    t0 = time.time()
    tree = build_dyf_tree(
        embeddings,
        max_depth=max_depth,
        num_bits=num_bits,
        min_leaf_size=min_leaf_size,
        seed=seed,
        fit_method=FIT_METHOD,
    )
    logger.info(f"  Tree built in {time.time() - t0:.1f}s")

    build_params = {
        "max_depth": max_depth,
        "num_bits": num_bits,
        "min_leaf_size": min_leaf_size,
        "seed": seed,
    }
    provenance = create_provenance(
        artifact_type="dyf",
        n_items=len(embeddings),
        source_paths=list(source_paths),
        params=ingest_params(
            max_depth=max_depth,
            num_bits=num_bits,
            min_leaf_size=min_leaf_size,
            seed=seed,
            dedup=dedup,
            embedding_model=metadata.get("embedding_model"),
            domain=metadata.get("domain"),
        ),
    )
    metadata = {**metadata, "_provenance_level_0": json.dumps(provenance_to_dict(provenance))}

    logger.info("Writing .dyf...")
    t0 = time.time()
    write_lazy_index(
        tree,
        embeddings,
        str(output),
        compression="none",
        quantization="float16",
        metadata=metadata,
        build_params=build_params,
        stored_fields=stored_fields,
    )
    size_mb = output.stat().st_size / 1_048_576
    logger.info(f"  Wrote {output} ({size_mb:.1f} MB) in {time.time() - t0:.1f}s")
    logger.info(f"Done: {len(embeddings)} items indexed.")
