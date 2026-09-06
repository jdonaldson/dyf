"""
Concept Graph — discover and navigate related concepts across markdown files.

Chunks markdown files at ## headers, embeds them, and builds a cosine-similarity
neighbor graph for pre-edit lookup. Two search modes:

1. Fuzzy header match (instant, <3ms, no model needed)
2. Semantic search (~4s, uses sentence-transformers via EmbedderConfig)

Usage as CLI (via `dyf concepts`):
    dyf concepts build              # build/rebuild the graph
    dyf concepts query "Debrief"    # fuzzy header match
    dyf concepts query --semantic "saving things"  # embedding search
    dyf concepts check              # check if graph is stale
    dyf concepts list               # list all nodes

Exit codes — these are a contract, not incidental. A caller that cannot tell "no such
concept" from "the tool is broken" will report the wrong thing:

    0  success; for `check`, the graph is current
    1  a normal negative answer: nothing matched, or `check` found the graph stale
       (grep's convention — a query finding nothing is not an error)
    2  the request itself was wrong: bad or missing --config
    3  a dependency is missing; the message names the extra to install

Usage as library:
    >>> from dyf.concept_graph import chunk_markdown, build_concept_graph
    >>> chunks = chunk_markdown(text, "myfile.md")
    >>> graph, embeddings = build_concept_graph(chunks)
"""

from __future__ import annotations

import argparse
import glob as glob_mod
import json
import logging
import os
import re
import time
from dataclasses import asdict, dataclass, field, fields
from difflib import SequenceMatcher
from pathlib import Path
from typing import TYPE_CHECKING

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    import numpy as np


class ConfigError(Exception):
    """A config file was requested but cannot be used.

    Carries a message meant to be shown to the caller verbatim — it should name the file
    and say what to do about it, not just what went wrong.
    """


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class MarkdownChunk:
    """A section of a markdown file, identified by its ## header."""

    id: str
    header: str
    text: str
    source: str
    line: int
    metadata: dict = field(default_factory=dict)


@dataclass
class ConceptNode:
    """A node in the concept graph with its neighbors."""

    header: str
    source: str
    line: int
    metadata: dict = field(default_factory=dict)
    neighbors: list[dict] = field(default_factory=list)


@dataclass
class ConceptGraphConfig:
    """Configuration for building and querying the concept graph."""

    sources: list[str] = field(
        default_factory=lambda: [
            "~/.claude/CLAUDE.md",
            "~/.claude/projects/*/memory/MEMORY.md",
            "~/Projects/CLAUDE.md",
            "/tmp/learnings_*.md",
        ]
    )
    output_path: str = "~/.dyf/concept_graph.json"
    embeddings_cache_path: str = "~/.dyf/concept_embeddings.npz"
    embedder: str = "low"
    top_k: int = 5
    similarity_threshold: float = 0.2

    @classmethod
    def load(cls, path: str | None = None) -> ConceptGraphConfig:
        """Load config from JSON file, falling back to defaults.

        A config the caller *asked for* is treated differently from the default one:

        * ``path=None`` — look in ``~/.config/dyf/concept_graph.json`` and silently use
          defaults if it is not there. Having no config file is the normal case.
        * an explicit ``path`` — missing, malformed or containing unknown keys is an
          error. Falling back to defaults there is the worst outcome: the caller believes
          their settings are in effect, the tool writes somewhere else, and nothing says
          so. That silence was the original behaviour.

        Raises:
            ConfigError: with a message naming the file and the problem.
        """
        explicit = path is not None
        if path is None:
            path = "~/.config/dyf/concept_graph.json"
        path = os.path.expanduser(path)

        if not os.path.exists(path):
            if explicit:
                raise ConfigError(f"config file not found: {path}")
            return cls()

        try:
            with open(path) as f:
                data = json.load(f)
        except json.JSONDecodeError as exc:
            raise ConfigError(f"{path} is not valid JSON: {exc}") from exc
        except OSError as exc:
            raise ConfigError(f"could not read {path}: {exc}") from exc

        if not isinstance(data, dict):
            raise ConfigError(f"{path} must contain a JSON object, got {type(data).__name__}")

        known = {f.name for f in fields(cls)}
        unknown = sorted(set(data) - known)
        if unknown:
            raise ConfigError(f"{path} has unknown setting(s): {', '.join(unknown)}. Valid: {', '.join(sorted(known))}")

        return cls(**data)

    def expand_sources(self) -> list[Path]:
        """Expand glob patterns and tilde in source paths."""
        paths = []
        for pattern in self.sources:
            expanded = os.path.expanduser(pattern)
            matches = sorted(glob_mod.glob(expanded))
            for m in matches:
                p = Path(m)
                if p.is_file():
                    paths.append(p)
        return paths

    def expand_path(self, attr: str) -> str:
        """Expand tilde in a path attribute."""
        return os.path.expanduser(getattr(self, attr))


# ---------------------------------------------------------------------------
# Markdown chunking
# ---------------------------------------------------------------------------


def slugify(text: str) -> str:
    """Convert header text to a URL-safe slug."""
    text = text.lower().strip()
    text = re.sub(r"[^\w\s-]", "", text)
    text = re.sub(r"[\s]+", "-", text)
    return text[:50]


def chunk_markdown(
    text: str,
    source: str,
    min_length: int = 20,
    header_level: int = 2,
) -> list[MarkdownChunk]:
    """Split markdown into chunks at the given header level.

    Args:
        text: Markdown content.
        source: Source file path (used in chunk IDs).
        min_length: Minimum body length to keep a chunk.
        header_level: Header level to split on (2 = ##, 3 = ###, etc.).

    Returns:
        List of MarkdownChunk objects.
    """
    prefix = "#" * header_level + " "
    # Source prefix for deduplication across files
    source_slug = slugify(Path(source).stem)

    chunks = []
    current_header = None
    current_lines: list[str] = []
    current_start_line = 1
    line_num = 0

    for line in text.split("\n"):
        line_num += 1
        if line.startswith(prefix) and (len(line) <= len(prefix) or line[len(prefix)] != "#"):
            # Save previous chunk
            if current_header is not None:
                body = "\n".join(current_lines).strip()
                if len(body) >= min_length:
                    chunk_id = f"{source_slug}/{slugify(current_header)}"
                    chunks.append(
                        MarkdownChunk(
                            id=chunk_id,
                            header=current_header,
                            text=f"{current_header}: {body}",
                            source=source,
                            line=current_start_line,
                        )
                    )
            current_header = line[len(prefix) :].strip()
            current_lines = []
            current_start_line = line_num
        else:
            current_lines.append(line)

    # Last chunk
    if current_header is not None:
        body = "\n".join(current_lines).strip()
        if len(body) >= min_length:
            chunk_id = f"{source_slug}/{slugify(current_header)}"
            chunks.append(
                MarkdownChunk(
                    id=chunk_id,
                    header=current_header,
                    text=f"{current_header}: {body}",
                    source=source,
                    line=current_start_line,
                )
            )

    return chunks


# ---------------------------------------------------------------------------
# Graph construction
# ---------------------------------------------------------------------------


def build_header_only_graph(chunks: list[MarkdownChunk]) -> dict[str, ConceptNode]:
    """Build the graph without embeddings — every node, no neighbors.

    The index half of the concept graph (header -> source:line) needs no model at all,
    and `fuzzy_match` is pure `SequenceMatcher`. Only *neighbors* require embedding. So
    when sentence-transformers is unavailable, a header-only graph still supports
    `concepts list`, `concepts check`, and `concepts query <header>` — which is the
    common path.

    This matters because the graph is consumed by agents told to run `dyf concepts`
    before editing notes: a tool that cannot build at all without a multi-GB torch stack
    is a tool they cannot use.
    """
    return {
        chunk.id: ConceptNode(
            header=chunk.header,
            source=chunk.source,
            line=chunk.line,
            metadata=chunk.metadata,
            neighbors=[],
        )
        for chunk in chunks
    }


def build_concept_graph(
    chunks: list[MarkdownChunk],
    top_k: int = 5,
    threshold: float = 0.2,
    embedder_name: str = "low",
) -> tuple[dict[str, ConceptNode], np.ndarray]:
    """Embed chunks and build a cosine-similarity neighbor graph.

    Lazy-imports EmbedderConfig so that fuzzy_match stays dependency-free. Raises
    ImportError if the embedding stack is absent — callers wanting a graph regardless
    should fall back to `build_header_only_graph`.

    Returns:
        (graph dict mapping chunk_id -> ConceptNode, embeddings array)
    """
    import numpy as np

    from dyf import EmbedderConfig

    embedder = getattr(EmbedderConfig, embedder_name.upper())
    texts = [c.text for c in chunks]
    embeddings = embedder.embed(texts)

    # Cosine similarity matrix
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    cos_matrix = (embeddings @ embeddings.T) / (norms @ norms.T + 1e-10)

    graph: dict[str, ConceptNode] = {}
    for i, chunk in enumerate(chunks):
        sims = cos_matrix[i]
        ranked = np.argsort(-sims)
        neighbors = []
        for idx in ranked:
            if idx == i:
                continue
            if sims[idx] < threshold:
                break
            neighbors.append(
                {
                    "id": chunks[idx].id,
                    "header": chunks[idx].header,
                    "similarity": round(float(sims[idx]), 3),
                    "source": chunks[idx].source,
                    "line": chunks[idx].line,
                }
            )
            if len(neighbors) >= top_k:
                break

        graph[chunk.id] = ConceptNode(
            header=chunk.header,
            source=chunk.source,
            line=chunk.line,
            metadata=chunk.metadata,
            neighbors=neighbors,
        )

    return graph, embeddings


# ---------------------------------------------------------------------------
# Serialization
# ---------------------------------------------------------------------------


# Reserved top-level key in the saved graph. Node ids come from markdown headers and are
# slugified, so they cannot collide with a leading underscore.
GRAPH_META_KEY = "_meta"


def save_graph(graph: dict[str, ConceptNode], path: str, *, has_embeddings: bool = True) -> None:
    """Save concept graph to JSON.

    `has_embeddings` records whether neighbors were actually computed, so a reader can
    distinguish "no neighbors above threshold" from "this graph was built without a
    model". Graphs written before this key existed simply lack it.
    """
    path = os.path.expanduser(path)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    data: dict = {GRAPH_META_KEY: {"has_embeddings": has_embeddings}}
    for node_id, node in graph.items():
        data[node_id] = asdict(node)
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


def load_graph(path: str) -> dict[str, ConceptNode]:
    """Load concept graph from JSON."""
    path = os.path.expanduser(path)
    with open(path) as f:
        data = json.load(f)
    graph = {}
    for node_id, node_data in data.items():
        if node_id == GRAPH_META_KEY:
            continue
        graph[node_id] = ConceptNode(**node_data)
    return graph


def load_graph_meta(path: str) -> dict:
    """Read the saved graph's metadata. Returns {} for graphs written before it existed."""
    path = os.path.expanduser(path)
    try:
        with open(path) as f:
            data = json.load(f)
    except (OSError, ValueError):
        return {}
    meta = data.get(GRAPH_META_KEY)
    return meta if isinstance(meta, dict) else {}


# ---------------------------------------------------------------------------
# Search
# ---------------------------------------------------------------------------


def fuzzy_match(
    query: str,
    graph: dict[str, ConceptNode],
    threshold: float = 0.4,
) -> tuple[str | None, float]:
    """Match query to node headers using string similarity.

    No model or heavy dependencies needed — uses SequenceMatcher.

    Returns:
        (best_node_id, best_score) or (None, best_score) if below threshold.
    """
    query_lower = query.lower().strip()
    best_score = 0.0
    best_id = None

    for node_id, node in graph.items():
        header = node.header.lower()
        scores = [
            SequenceMatcher(None, query_lower, header).ratio(),
            SequenceMatcher(None, query_lower, node_id).ratio(),
        ]
        # Bonus for substring containment
        if query_lower in header or header in query_lower:
            scores.append(0.9)
        if query_lower in node_id or node_id in query_lower:
            scores.append(0.85)

        score = max(scores)
        if score > best_score:
            best_score = score
            best_id = node_id

    if best_score >= threshold:
        return best_id, best_score
    return None, best_score


def semantic_search(
    query: str,
    graph: dict[str, ConceptNode],
    embeddings_cache_path: str = "~/.dyf/concept_embeddings.npz",
    embedder_name: str = "low",
    top_k: int = 5,
) -> list[tuple[str, float]]:
    """Search by embedding similarity. Returns list of (node_id, score).

    Lazy-imports numpy and EmbedderConfig.
    """
    import numpy as np

    from dyf import EmbedderConfig

    cache_path = os.path.expanduser(embeddings_cache_path)

    # Load or build cached embeddings
    if os.path.exists(cache_path):
        data = np.load(cache_path, allow_pickle=True)
        node_embeddings = data["embeddings"]
        node_ids = list(data["ids"])
    else:
        embedder = getattr(EmbedderConfig, embedder_name.upper())
        node_ids = list(graph.keys())
        headers = [graph[nid].header for nid in node_ids]
        node_embeddings = embedder.embed(headers)
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        np.savez(cache_path, embeddings=node_embeddings, ids=np.array(node_ids))

    # Embed query
    embedder = getattr(EmbedderConfig, embedder_name.upper())
    q_emb = embedder.embed([query])

    # Cosine similarity
    sims = (node_embeddings @ q_emb.T).flatten()
    norms = np.linalg.norm(node_embeddings, axis=1) * np.linalg.norm(q_emb)
    cos_sims = sims / (norms + 1e-10)
    ranked = np.argsort(-cos_sims)[:top_k]

    return [(node_ids[idx], float(cos_sims[idx])) for idx in ranked]


# ---------------------------------------------------------------------------
# Staleness check
# ---------------------------------------------------------------------------


def check_staleness(config: ConceptGraphConfig) -> bool:
    """Check if graph needs rebuilding based on source file mtimes.

    Returns True if stale (graph missing or older than any source).
    """
    graph_path = config.expand_path("output_path")
    if not os.path.exists(graph_path):
        return True

    graph_mtime = os.path.getmtime(graph_path)
    for source_path in config.expand_sources():
        if source_path.exists() and os.path.getmtime(source_path) > graph_mtime:
            return True
    return False


# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------


def _format_node(node: ConceptNode, header_only: bool = False) -> str:
    """Format a single node with its neighbors for display.

    `header_only` distinguishes the two reasons a node can show no neighbors: none scored
    above the similarity threshold, or the graph was built without a model and *cannot*
    have any. Reporting the first when the second is true sends a reader looking for a
    threshold to tune.
    """
    lines = [
        f"## {node.header}",
        f"   {node.source}:{node.line}",
    ]
    if node.neighbors:
        lines.append("\n   Also check:")
        for n in node.neighbors:
            lines.append(f"   {n['similarity']:.3f}  {n['header']}")
            lines.append(f"          {n['source']}:{n['line']}")
    elif header_only:
        lines.append("   (header-only graph — rebuild with 'dyf[concepts]' for neighbors)")
    else:
        lines.append("   (no neighbors above threshold)")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    """CLI entry point for `dyf concepts`."""
    parser = argparse.ArgumentParser(
        prog="dyf concepts",
        description="Build and query concept graphs from markdown files.",
    )
    sub = parser.add_subparsers(dest="command")

    # build
    build_p = sub.add_parser("build", help="Build or rebuild the concept graph")
    build_p.add_argument("--config", help="Path to config JSON")
    build_p.add_argument(
        "--no-embeddings",
        action="store_true",
        help="Build a header-only graph: no model needed, no semantic neighbors",
    )
    build_p.add_argument("extra_sources", nargs="*", help="Additional source files")

    # query
    query_p = sub.add_parser("query", help="Look up a concept")
    query_p.add_argument("text", nargs="+", help="Query text")
    query_p.add_argument("--semantic", action="store_true", help="Force embedding-based search")
    query_p.add_argument("--top-k", type=int, default=5)
    query_p.add_argument("--config", help="Path to config JSON")

    # check
    check_p = sub.add_parser("check", help="Check if graph needs rebuilding")
    check_p.add_argument("--config", help="Path to config JSON")

    # list
    list_p = sub.add_parser("list", help="List all concept nodes")
    list_p.add_argument("--verbose", "-v", action="store_true", help="Show neighbors")
    list_p.add_argument("--config", help="Path to config JSON")

    args = parser.parse_args(argv)

    if args.command is None:
        parser.print_help()
        return 1

    try:
        config = ConceptGraphConfig.load(getattr(args, "config", None))
    except ConfigError as exc:
        logger.error("%s", exc)
        return 2

    if args.command == "build":
        return _cmd_build(
            config,
            getattr(args, "extra_sources", []),
            no_embeddings=getattr(args, "no_embeddings", False),
        )
    elif args.command == "check":
        return _cmd_check(config)
    elif args.command == "query":
        query_text = " ".join(args.text)
        return _cmd_query(config, query_text, args.semantic, args.top_k)
    elif args.command == "list":
        return _cmd_list(config, args.verbose)

    return 1


def _cmd_build(config: ConceptGraphConfig, extra_sources: list[str], no_embeddings: bool = False) -> int:
    """Build the concept graph from configured sources."""
    all_chunks: list[MarkdownChunk] = []

    source_paths = config.expand_sources()
    for extra in extra_sources:
        p = Path(os.path.expanduser(extra))
        if p.is_file():
            source_paths.append(p)

    for source_path in source_paths:
        logger.info("Reading %s", source_path)
        text = source_path.read_text()
        chunks = chunk_markdown(text, str(source_path))
        logger.debug("  -> %d chunks", len(chunks))
        all_chunks.extend(chunks)

    if not all_chunks:
        logger.warning("No chunks found!")
        return 1

    # Deduplicate by id
    seen: set = set()
    deduped = []
    for c in all_chunks:
        if c.id not in seen:
            seen.add(c.id)
            deduped.append(c)
        else:
            logger.debug("  Skipping duplicate: %s", c.id)
    all_chunks = deduped

    logger.info("Total: %d unique concept chunks", len(all_chunks))

    embeddings = None
    if no_embeddings:
        graph = build_header_only_graph(all_chunks)
    else:
        try:
            graph, embeddings = build_concept_graph(
                all_chunks,
                top_k=config.top_k,
                threshold=config.similarity_threshold,
                embedder_name=config.embedder,
            )
        except ImportError as exc:
            # Degrade rather than die. `check` tells the user to run `build`, so a build
            # that cannot run without a multi-GB torch stack makes the tool's own advice
            # unfollowable.
            #
            # But refuse to *silently downgrade* a graph that already has neighbors.
            # dyf is installed globally as a lightweight tool while the project venv
            # carries the model, so the same `dyf concepts build` means different things
            # depending on which one is on PATH — and the lightweight one would otherwise
            # quietly delete every edge the other computed.
            existing_path = config.expand_path("output_path")
            if os.path.exists(existing_path) and load_graph_meta(existing_path).get("has_embeddings") is not False:
                logger.error("Embedding model unavailable (%s)", exc)
                logger.error("  Refusing to overwrite the existing graph at %s,", existing_path)
                logger.error("  which has semantic neighbors this build cannot reproduce.")
                logger.error("  Install the model:  pip install 'dyf[concepts]'")
                logger.error("  Or discard neighbors deliberately:  dyf concepts build --no-embeddings")
                return 1

            logger.warning("Embedding model unavailable (%s)", exc)
            logger.warning("  Building a header-only graph: no semantic neighbors.")
            logger.warning("  For neighbors and `query --semantic`: pip install 'dyf[concepts]'")
            graph = build_header_only_graph(all_chunks)

    output_path = config.expand_path("output_path")
    save_graph(graph, output_path, has_embeddings=embeddings is not None)
    total_edges = sum(len(n.neighbors) for n in graph.values())
    logger.info("Graph saved to %s", output_path)
    if embeddings is None:
        logger.info("  %d nodes, header-only (no neighbors)", len(graph))
        return 0
    logger.info("  %d nodes, %d edges", len(graph), total_edges)

    # Save embeddings cache
    import numpy as np

    cache_path = config.expand_path("embeddings_cache_path")
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    node_ids = list(graph.keys())
    np.savez(cache_path, embeddings=embeddings, ids=np.array(node_ids))
    logger.debug("  Embeddings cached to %s", cache_path)

    return 0


def _cmd_check(config: ConceptGraphConfig) -> int:
    """Check if graph is stale."""
    if check_staleness(config):
        logger.warning("STALE - run `dyf concepts build`")
        return 1
    else:
        logger.info("OK - graph is current")
        return 0


def _cmd_query(
    config: ConceptGraphConfig,
    query_text: str,
    force_semantic: bool,
    top_k: int,
) -> int:
    """Query the concept graph."""
    graph_path = config.expand_path("output_path")
    if not os.path.exists(graph_path):
        logger.warning("No graph found. Run: dyf concepts build")
        return 1

    t0 = time.time()
    graph = load_graph(graph_path)
    header_only = load_graph_meta(graph_path).get("has_embeddings") is False

    if force_semantic and header_only:
        logger.error("This graph was built without embeddings, so semantic search is unavailable.")
        logger.error("  Rebuild with: pip install 'dyf[concepts]' && dyf concepts build")
        return 1

    if force_semantic:
        results = semantic_search(
            query_text,
            graph,
            embeddings_cache_path=config.expand_path("embeddings_cache_path"),
            embedder_name=config.embedder,
            top_k=top_k,
        )
        elapsed = time.time() - t0
        logger.info('Semantic search: "%s"', query_text)
        for node_id, score in results:
            node = graph[node_id]
            logger.info("   %0.3f  %s", score, node.header)
            logger.debug("          %s:%d", node.source, node.line)
            if node.neighbors:
                neighbor_headers = ", ".join(n["header"] for n in node.neighbors[:3])
                logger.info("          -> neighbors: %s", neighbor_headers)
        logger.debug("   (%.3fs)", elapsed)
        return 0

    # Fuzzy match first
    node_id, score = fuzzy_match(query_text, graph)
    elapsed = time.time() - t0

    if node_id:
        node = graph[node_id]
        logger.info(_format_node(node, header_only=header_only))
        logger.debug("   (matched '%s' score=%.2f, %.1fms)", node_id, score, elapsed * 1000)
        return 0

    # No header match. Exit 1 when nothing is found, following grep: a caller — human or
    # agent — needs to distinguish "no such concept" from "the tool worked".
    logger.info('No header match for "%s" (best=%.2f)', query_text, score)

    if header_only:
        logger.warning("  This graph has no embeddings, so there is no semantic fallback.")
        logger.warning("  For fuzzy-to-semantic search: pip install 'dyf[concepts]' && dyf concepts build")
        return 1

    logger.info("Falling back to semantic search...")
    try:
        results = semantic_search(
            query_text,
            graph,
            embeddings_cache_path=config.expand_path("embeddings_cache_path"),
            embedder_name=config.embedder,
            top_k=top_k,
        )
    except ImportError as exc:
        logger.error("  Semantic fallback unavailable (%s)", exc)
        logger.error("  Install with: pip install 'dyf[concepts]'")
        return 3

    elapsed = time.time() - t0
    if not results:
        logger.info("  No semantic matches either.")
        return 1

    for node_id, sim_score in results:
        node = graph[node_id]
        logger.info("   %0.3f  %s", sim_score, node.header)
        logger.debug("          %s:%d", node.source, node.line)
    logger.debug("   (%.3fs)", elapsed)
    return 0


def _cmd_list(config: ConceptGraphConfig, verbose: bool) -> int:
    """List all nodes in the graph."""
    graph_path = config.expand_path("output_path")
    if not os.path.exists(graph_path):
        logger.warning("No graph found. Run: dyf concepts build")
        return 1

    graph = load_graph(graph_path)
    for node_id, node in graph.items():
        logger.info("  %s  [%s]", node_id, node.header)
        logger.debug("    %s:%d", node.source, node.line)
        if verbose and node.neighbors:
            for n in node.neighbors:
                logger.info("      %0.3f  %s", n["similarity"], n["header"])
    if load_graph_meta(graph_path).get("has_embeddings") is False:
        logger.info("%d nodes total (header-only graph — no neighbors)", len(graph))
        return 0
    logger.info("%d nodes total", len(graph))
    return 0
