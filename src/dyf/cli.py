"""
DYF CLI — entry point for `dyf` command.

Usage:
    dyf info f.dyf           # describe a .dyf file (add --json to parse it)
    dyf api                  # map of the public Python API, grouped
    dyf concepts build       # build concept graph
    dyf concepts query ...   # query concepts
    dyf concepts check       # check staleness
    dyf concepts list        # list all nodes
    dyf index-source dir/    # index source code into .dyf
    dyf index-images dir/    # index images into .dyf (vision embeddings)
    dyf index-video file.mp4 # index video keyframes into .dyf

Enrichment (UMAP projection, clustering, narration, audio) and the browser tour live
downstream in `dyfviz`, split out 2026-09-05: `dyfviz enrich all f.dyf`, `dyfviz tour`.
"""

import logging
import sys

# A missing or unknown top-level command is a malformed request, which the ingest
# exit-code contract already numbers. Reused here so the whole CLI speaks one language
# rather than growing a second, parallel set of codes.
from ._ingest_errors import EXIT_BAD_REQUEST


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


# Which pyproject extra supplies each subcommand's optional dependencies. Used to turn a
# ModuleNotFoundError into a line the caller can act on.
COMMAND_EXTRAS = {
    "info": "lazy",
    "api": "",  # pure metadata, no optional deps
    "concepts": "concepts",
    "index-source": "source",
    "index-images": "vision",
    "index-video": "video",
}

# Subcommands that moved to the downstream `dyfviz` package on 2026-09-05. Kept here so
# an old command line gets a redirect instead of a bare usage dump — sec10quant's
# Makefile and anyone's muscle memory still say `dyf enrich`.
MOVED_COMMANDS = {
    "enrich": "dyfviz enrich",
    "tour": "dyfviz tour",
}


def _dispatch(cmd: str, argv: list[str]) -> int | None:
    """Route to a subcommand, returning its exit code.

    Returns None if `cmd` is not a known subcommand, so the caller can print usage.
    Lets ImportError propagate: `main` turns it into an actionable message.
    """
    if cmd == "info":
        from .info import main as info_main

        return info_main(argv)
    if cmd == "api":
        from .api_cmd import main as api_main

        return api_main(argv)
    if cmd == "concepts":
        from .concept_graph import main as concepts_main

        return concepts_main(argv)
    if cmd == "index-source":
        from .index_source import main as index_main

        return index_main(argv)
    if cmd == "index-images":
        from .index_images import main as index_images_main

        return index_images_main(argv)
    if cmd == "index-video":
        from .index_video import main as index_video_main

        return index_video_main(argv)
    return None


HELP_FLAGS = {"-h", "--help", "help"}


def _print_usage(stream) -> None:
    lines = [
        "Usage: dyf <command>",
        "",
        "Commands:",
        "  info          Describe a .dyf file (items, fields, enrichment level; --json)",
        "  api           Map of the public Python API, grouped with entry points (--json)",
        "  concepts      Build and query concept graphs from markdown files",
        "  index-source  Index source code into a .dyf file (Python, JS, TS, Rust, Go, Java, C, C++, OCaml)",
        "  index-images  Index images into a .dyf file (vision embeddings + thumbnails)",
        "  index-video   Index video keyframes into a .dyf file (scene detection + vision)",
        "",
        "Run `dyf <command> --help` for a command's own options.",
        "Enrichment and the browser tour moved to dyfviz: dyfviz enrich | dyfviz tour",
    ]
    print("\n".join(lines), file=stream)


def main():
    _configure_cli_logging()

    if len(sys.argv) > 1:
        cmd = sys.argv[1]

        # Asking for help is a successful request, so it exits 0 and answers on stdout.
        # Every other path through this block is a usage *error* and exits 2.
        #
        # These four cases — `--help`, an unknown command, a moved command, and no
        # command at all — used to share one usage dump and one exit code (1), because
        # `_dispatch` returns None for anything it does not recognise and the fallthrough
        # could not tell those apart. A caller that ran `dyf --help` got rc 1; a caller
        # that typoed a command got rc 1 with nothing saying which word was wrong.
        #
        # `benchmarks/audit_cli_surface.py` documents why it skips `--help` for
        # subcommands: argparse answers those itself without touching the logger, so a
        # help smoke test is blind to a mute execution path. True — but that reasoning
        # only holds where argparse is doing the parsing. Up here there is none, and
        # generalising it to "do not test --help" is what let this ship in 0.13.0.
        if cmd in HELP_FLAGS:
            _print_usage(sys.stdout)
            sys.exit(0)

        if cmd in MOVED_COMMANDS:
            logger = logging.getLogger("dyf")
            logger.error("`dyf %s` moved to the dyfviz package on 2026-09-05.", cmd)
            logger.error("  Use: %s %s", MOVED_COMMANDS[cmd], " ".join(sys.argv[2:]))
            logger.error("  Install with: pip install dyfviz[all]")
            sys.exit(4)

        try:
            code = _dispatch(cmd, sys.argv[2:])
        except ImportError as exc:
            # Optional dependencies are the most common way these commands fail, and a
            # ModuleNotFoundError traceback tells the caller nothing about the fix.
            # Modules that already raise a helpful ImportError keep their own wording.
            logger = logging.getLogger("dyf")
            logger.error("%s", exc)
            extra = COMMAND_EXTRAS.get(cmd)
            if extra and f"dyf[{extra}]" not in str(exc):
                logger.error("`dyf %s` needs: pip install 'dyf[%s]'", cmd, extra)
            sys.exit(3)
        if code is not None:
            sys.exit(code)

        # `_dispatch` returned None: `cmd` is not a subcommand. Name the offending word
        # rather than dumping usage and leaving the caller to diff it against what they
        # typed.
        logger = logging.getLogger("dyf")
        logger.error("unknown command: %s", cmd)
        _print_usage(sys.stderr)
        sys.exit(EXIT_BAD_REQUEST)

    # No command at all. Usage goes to stderr because stdout is reserved for answers,
    # and this is an error, not an answer.
    _print_usage(sys.stderr)
    sys.exit(EXIT_BAD_REQUEST)


if __name__ == "__main__":
    main()
