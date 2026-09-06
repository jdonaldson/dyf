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

import logging
import time
from pathlib import Path

import numpy as np

from .dyf_tree import build_dyf_tree
from .lazy_index import write_lazy_index

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
) -> None:
    """Dedup, normalize, build the tree, and write the `.dyf`.

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

    logger.info("Writing .dyf...")
    t0 = time.time()
    write_lazy_index(
        tree,
        embeddings,
        str(output),
        compression="none",
        quantization="float16",
        metadata=metadata,
        build_params={
            "max_depth": max_depth,
            "num_bits": num_bits,
            "min_leaf_size": min_leaf_size,
            "seed": seed,
        },
        stored_fields=stored_fields,
    )
    size_mb = output.stat().st_size / 1_048_576
    logger.info(f"  Wrote {output} ({size_mb:.1f} MB) in {time.time() - t0:.1f}s")
    logger.info(f"Done: {len(embeddings)} items indexed.")
