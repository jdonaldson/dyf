"""
DYF CLI — entry point for `dyf` command.

Usage:
    dyf concepts build       # build concept graph
    dyf concepts query ...   # query concepts
    dyf concepts check       # check staleness
    dyf concepts list        # list all nodes
    dyf index-source dir/    # index source code into .dyf
    dyf index-images dir/    # index images into .dyf (vision embeddings)
    dyf index-video file.mp4 # index video keyframes into .dyf
    dyf enrich project f.dyf # UMAP projection (Level 0→1)
    dyf enrich cluster f.dyf # Louvain clustering (Level 1→2)
    dyf enrich viz f.dyf     # Bridge edges + narration (Level 2→3)
    dyf enrich all f.dyf     # Run all enrichment levels
    dyf tour f.dyf           # Launch browser viewer with tour
"""

import logging
import sys


def _configure_cli_logging() -> None:
    """Give the package logger a console handler when running as a CLI.

    `__init__.py` installs a NullHandler on the `dyf` logger — correct for a library,
    since it suppresses Python's handler of last resort. But nothing else in the
    package configures logging, so as a CLI every subcommand that reports through
    `logger` emitted nothing at all: `dyf concepts list` printed 0 bytes on a graph
    with 100+ nodes. See AGENT_LEGIBILITY_TODO.md P0.

    Scoped to the `dyf` logger rather than `basicConfig`, which configures the *root*
    logger and would turn on INFO for every third-party library too (httpx dumping
    every HuggingFace request during `concepts build` is the case that caught this).

    Results go to **stdout**, problems to **stderr**. `logging.StreamHandler()` defaults
    to stderr, but for these subcommands `logger.info` carries the actual answer — the
    node list, the file summary — not a diagnostic. Left on stderr,
    `dyf concepts list > out.txt` writes an empty file, which breaks any caller that
    redirects or pipes. Splitting at WARNING keeps both audiences correct.
    """
    pkg_logger = logging.getLogger("dyf")
    if any(not isinstance(h, logging.NullHandler) for h in pkg_logger.handlers):
        return  # already configured — don't double up

    fmt = logging.Formatter("%(message)s")

    out = logging.StreamHandler(sys.stdout)
    out.setFormatter(fmt)
    out.setLevel(logging.INFO)
    out.addFilter(lambda record: record.levelno < logging.WARNING)

    err = logging.StreamHandler(sys.stderr)
    err.setFormatter(fmt)
    err.setLevel(logging.WARNING)

    pkg_logger.addHandler(out)
    pkg_logger.addHandler(err)
    pkg_logger.setLevel(logging.INFO)


def main():
    _configure_cli_logging()

    if len(sys.argv) > 1:
        cmd = sys.argv[1]
        if cmd == "concepts":
            from .concept_graph import main as concepts_main

            sys.exit(concepts_main(sys.argv[2:]))
        elif cmd == "index-source":
            from .index_source import main as index_main

            sys.exit(index_main(sys.argv[2:]))
        elif cmd == "enrich":
            from .enrich import main as enrich_main

            enrich_main(sys.argv[2:])
            sys.exit(0)
        elif cmd == "index-images":
            from .index_images import main as index_images_main

            sys.exit(index_images_main(sys.argv[2:]))
        elif cmd == "index-video":
            from .index_video import main as index_video_main

            sys.exit(index_video_main(sys.argv[2:]))
        elif cmd == "tour":
            from .tour import main as tour_main

            tour_main(sys.argv[2:])
            sys.exit(0)

    print("Usage: dyf <command>")
    print()
    print("Commands:")
    print("  concepts      Build and query concept graphs from markdown files")
    print("  index-source  Index source code into a .dyf file (Python, JS, TS, Rust, Go, Java, C, C++, OCaml)")
    print("  index-images  Index images into a .dyf file (vision embeddings + thumbnails)")
    print("  index-video   Index video keyframes into a .dyf file (scene detection + vision)")
    print("  enrich        Enrich a .dyf file (UMAP, clustering, narration)")
    print("  tour          Launch browser viewer with tour autoplay")
    sys.exit(1)


if __name__ == "__main__":
    main()
