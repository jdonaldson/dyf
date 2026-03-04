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

Usage as library:
    >>> from dyf.concept_graph import chunk_markdown, build_concept_graph
    >>> chunks = chunk_markdown(text, "myfile.md")
    >>> graph, embeddings = build_concept_graph(chunks)
"""

from __future__ import annotations

import argparse
import glob as glob_mod
import json
import os
import re
import time
from dataclasses import dataclass, field, asdict
from difflib import SequenceMatcher
from pathlib import Path
from typing import Dict, List, Optional, Tuple, TYPE_CHECKING

if TYPE_CHECKING:
    import numpy as np


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
    metadata: Dict = field(default_factory=dict)


@dataclass
class ConceptNode:
    """A node in the concept graph with its neighbors."""
    header: str
    source: str
    line: int
    metadata: Dict = field(default_factory=dict)
    neighbors: List[Dict] = field(default_factory=list)


@dataclass
class ConceptGraphConfig:
    """Configuration for building and querying the concept graph."""
    sources: List[str] = field(default_factory=lambda: [
        "~/.claude/CLAUDE.md",
        "~/.claude/projects/*/memory/MEMORY.md",
        "~/Projects/CLAUDE.md",
        "/tmp/learnings_*.md",
    ])
    output_path: str = "~/.dyf/concept_graph.json"
    embeddings_cache_path: str = "~/.dyf/concept_embeddings.npz"
    embedder: str = "low"
    top_k: int = 5
    similarity_threshold: float = 0.2

    @classmethod
    def load(cls, path: Optional[str] = None) -> "ConceptGraphConfig":
        """Load config from JSON file, falling back to defaults."""
        if path is None:
            path = os.path.expanduser("~/.config/dyf/concept_graph.json")
        path = os.path.expanduser(path)
        if os.path.exists(path):
            with open(path) as f:
                data = json.load(f)
            return cls(**data)
        return cls()

    def expand_sources(self) -> List[Path]:
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
) -> List[MarkdownChunk]:
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
    current_lines: List[str] = []
    current_start_line = 1
    line_num = 0

    for line in text.split("\n"):
        line_num += 1
        if line.startswith(prefix) and (
            len(line) <= len(prefix) or line[len(prefix)] != "#"
        ):
            # Save previous chunk
            if current_header is not None:
                body = "\n".join(current_lines).strip()
                if len(body) >= min_length:
                    chunk_id = f"{source_slug}/{slugify(current_header)}"
                    chunks.append(MarkdownChunk(
                        id=chunk_id,
                        header=current_header,
                        text=f"{current_header}: {body}",
                        source=source,
                        line=current_start_line,
                    ))
            current_header = line[len(prefix):].strip()
            current_lines = []
            current_start_line = line_num
        else:
            current_lines.append(line)

    # Last chunk
    if current_header is not None:
        body = "\n".join(current_lines).strip()
        if len(body) >= min_length:
            chunk_id = f"{source_slug}/{slugify(current_header)}"
            chunks.append(MarkdownChunk(
                id=chunk_id,
                header=current_header,
                text=f"{current_header}: {body}",
                source=source,
                line=current_start_line,
            ))

    return chunks


# ---------------------------------------------------------------------------
# Graph construction
# ---------------------------------------------------------------------------

def build_concept_graph(
    chunks: List[MarkdownChunk],
    top_k: int = 5,
    threshold: float = 0.2,
    embedder_name: str = "low",
) -> Tuple[Dict[str, ConceptNode], np.ndarray]:
    """Embed chunks and build a cosine-similarity neighbor graph.

    Lazy-imports EmbedderConfig so that fuzzy_match stays dependency-free.

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

    graph: Dict[str, ConceptNode] = {}
    for i, chunk in enumerate(chunks):
        sims = cos_matrix[i]
        ranked = np.argsort(-sims)
        neighbors = []
        for idx in ranked:
            if idx == i:
                continue
            if sims[idx] < threshold:
                break
            neighbors.append({
                "id": chunks[idx].id,
                "header": chunks[idx].header,
                "similarity": round(float(sims[idx]), 3),
                "source": chunks[idx].source,
                "line": chunks[idx].line,
            })
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

def save_graph(graph: Dict[str, ConceptNode], path: str) -> None:
    """Save concept graph to JSON."""
    path = os.path.expanduser(path)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    data = {}
    for node_id, node in graph.items():
        data[node_id] = asdict(node)
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


def load_graph(path: str) -> Dict[str, ConceptNode]:
    """Load concept graph from JSON."""
    path = os.path.expanduser(path)
    with open(path) as f:
        data = json.load(f)
    graph = {}
    for node_id, node_data in data.items():
        graph[node_id] = ConceptNode(**node_data)
    return graph


# ---------------------------------------------------------------------------
# Search
# ---------------------------------------------------------------------------

def fuzzy_match(
    query: str,
    graph: Dict[str, ConceptNode],
    threshold: float = 0.4,
) -> Tuple[Optional[str], float]:
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
    graph: Dict[str, ConceptNode],
    embeddings_cache_path: str = "~/.dyf/concept_embeddings.npz",
    embedder_name: str = "low",
    top_k: int = 5,
) -> List[Tuple[str, float]]:
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

def _format_node(node: ConceptNode) -> str:
    """Format a single node with its neighbors for display."""
    lines = [
        f"## {node.header}",
        f"   {node.source}:{node.line}",
    ]
    if node.neighbors:
        lines.append("\n   Also check:")
        for n in node.neighbors:
            lines.append(f"   {n['similarity']:.3f}  {n['header']}")
            lines.append(f"          {n['source']}:{n['line']}")
    else:
        lines.append("   (no neighbors above threshold)")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main(argv: Optional[List[str]] = None) -> int:
    """CLI entry point for `dyf concepts`."""
    parser = argparse.ArgumentParser(
        prog="dyf concepts",
        description="Build and query concept graphs from markdown files.",
    )
    sub = parser.add_subparsers(dest="command")

    # build
    build_p = sub.add_parser("build", help="Build or rebuild the concept graph")
    build_p.add_argument("--config", help="Path to config JSON")
    build_p.add_argument("extra_sources", nargs="*", help="Additional source files")

    # query
    query_p = sub.add_parser("query", help="Look up a concept")
    query_p.add_argument("text", nargs="+", help="Query text")
    query_p.add_argument("--semantic", action="store_true",
                         help="Force embedding-based search")
    query_p.add_argument("--top-k", type=int, default=5)
    query_p.add_argument("--config", help="Path to config JSON")

    # check
    check_p = sub.add_parser("check", help="Check if graph needs rebuilding")
    check_p.add_argument("--config", help="Path to config JSON")

    # list
    list_p = sub.add_parser("list", help="List all concept nodes")
    list_p.add_argument("--verbose", "-v", action="store_true",
                        help="Show neighbors")
    list_p.add_argument("--config", help="Path to config JSON")

    args = parser.parse_args(argv)

    if args.command is None:
        parser.print_help()
        return 1

    config = ConceptGraphConfig.load(
        getattr(args, "config", None)
    )

    if args.command == "build":
        return _cmd_build(config, getattr(args, "extra_sources", []))
    elif args.command == "check":
        return _cmd_check(config)
    elif args.command == "query":
        query_text = " ".join(args.text)
        return _cmd_query(config, query_text, args.semantic, args.top_k)
    elif args.command == "list":
        return _cmd_list(config, args.verbose)

    return 1


def _cmd_build(config: ConceptGraphConfig, extra_sources: List[str]) -> int:
    """Build the concept graph from configured sources."""
    all_chunks: List[MarkdownChunk] = []

    source_paths = config.expand_sources()
    for extra in extra_sources:
        p = Path(os.path.expanduser(extra))
        if p.is_file():
            source_paths.append(p)

    for source_path in source_paths:
        print(f"Reading {source_path}")
        text = source_path.read_text()
        chunks = chunk_markdown(text, str(source_path))
        print(f"  -> {len(chunks)} chunks")
        all_chunks.extend(chunks)

    if not all_chunks:
        print("No chunks found!")
        return 1

    # Deduplicate by id
    seen: set = set()
    deduped = []
    for c in all_chunks:
        if c.id not in seen:
            seen.add(c.id)
            deduped.append(c)
        else:
            print(f"  Skipping duplicate: {c.id}")
    all_chunks = deduped

    print(f"\nTotal: {len(all_chunks)} unique concept chunks")

    graph, embeddings = build_concept_graph(
        all_chunks,
        top_k=config.top_k,
        threshold=config.similarity_threshold,
        embedder_name=config.embedder,
    )

    # Save graph
    output_path = config.expand_path("output_path")
    save_graph(graph, output_path)
    total_edges = sum(len(n.neighbors) for n in graph.values())
    print(f"\nGraph saved to {output_path}")
    print(f"  {len(graph)} nodes, {total_edges} edges")

    # Save embeddings cache
    import numpy as np
    cache_path = config.expand_path("embeddings_cache_path")
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    node_ids = list(graph.keys())
    np.savez(cache_path, embeddings=embeddings, ids=np.array(node_ids))
    print(f"  Embeddings cached to {cache_path}")

    return 0


def _cmd_check(config: ConceptGraphConfig) -> int:
    """Check if graph is stale."""
    if check_staleness(config):
        print("STALE - run `dyf concepts build`")
        return 1
    else:
        print("OK - graph is current")
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
        print("No graph found. Run: dyf concepts build")
        return 1

    t0 = time.time()
    graph = load_graph(graph_path)

    if force_semantic:
        results = semantic_search(
            query_text, graph,
            embeddings_cache_path=config.expand_path("embeddings_cache_path"),
            embedder_name=config.embedder,
            top_k=top_k,
        )
        elapsed = time.time() - t0
        print(f'## Semantic search: "{query_text}"\n')
        for node_id, score in results:
            node = graph[node_id]
            print(f"   {score:.3f}  {node.header}")
            print(f"          {node.source}:{node.line}")
            if node.neighbors:
                neighbor_headers = ", ".join(
                    n["header"] for n in node.neighbors[:3]
                )
                print(f"          -> neighbors: {neighbor_headers}")
            print()
        print(f"   ({elapsed:.3f}s)")
        return 0

    # Fuzzy match first
    node_id, score = fuzzy_match(query_text, graph)
    elapsed = time.time() - t0

    if node_id:
        node = graph[node_id]
        print(_format_node(node))
        print(f"\n   (matched '{node_id}' score={score:.2f}, {elapsed*1000:.1f}ms)")
        return 0

    # Fall back to semantic
    print(f'No header match for "{query_text}" (best={score:.2f})')
    print("Falling back to semantic search...\n")
    results = semantic_search(
        query_text, graph,
        embeddings_cache_path=config.expand_path("embeddings_cache_path"),
        embedder_name=config.embedder,
        top_k=top_k,
    )
    elapsed = time.time() - t0
    for node_id, sim_score in results:
        node = graph[node_id]
        print(f"   {sim_score:.3f}  {node.header}")
        print(f"          {node.source}:{node.line}")
    print(f"\n   ({elapsed:.3f}s)")
    return 0


def _cmd_list(config: ConceptGraphConfig, verbose: bool) -> int:
    """List all nodes in the graph."""
    graph_path = config.expand_path("output_path")
    if not os.path.exists(graph_path):
        print("No graph found. Run: dyf concepts build")
        return 1

    graph = load_graph(graph_path)
    for node_id, node in graph.items():
        print(f"  {node_id}  [{node.header}]")
        print(f"    {node.source}:{node.line}")
        if verbose and node.neighbors:
            for n in node.neighbors:
                print(f"      {n['similarity']:.3f}  {n['header']}")
        if verbose:
            print()

    print(f"\n{len(graph)} nodes total")
    return 0
