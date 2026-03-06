"""
Index source code into a .dyf file.

Chunks at function/class boundaries using tree-sitter, embeds via Ollama,
builds a DYF tree, and writes a .dyf index.

Supports Python, JavaScript, TypeScript, Rust, Go, Java, C, C++, and OCaml.

Usage (via CLI):
    dyf index-source src/mypackage/ -o mypackage.dyf
    dyf index-source . -o project.dyf --model nomic-embed-text

Requires: pip install "dyf[source]"
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import requests

from .dyf_tree import build_dyf_tree
from .lazy_index import write_lazy_index


DEFAULT_OLLAMA_URL = "http://localhost:11434/api/embed"
DEFAULT_MODEL = "nomic-embed-text"

# Per-language config: file extensions and AST node types to extract as chunks.
LANG_CONFIG = {
    "python": {
        "extensions": [".py"],
        "chunk_types": ["function_definition", "class_definition"],
    },
    "javascript": {
        "extensions": [".js", ".jsx", ".mjs"],
        "chunk_types": ["function_declaration", "class_declaration", "method_definition"],
    },
    "typescript": {
        "extensions": [".ts", ".tsx"],
        "chunk_types": ["function_declaration", "class_declaration", "method_definition"],
    },
    "rust": {
        "extensions": [".rs"],
        "chunk_types": ["function_item", "struct_item", "impl_item", "trait_item"],
    },
    "go": {
        "extensions": [".go"],
        "chunk_types": ["function_declaration", "method_declaration", "type_declaration"],
    },
    "java": {
        "extensions": [".java"],
        "chunk_types": ["class_declaration", "method_declaration", "constructor_declaration"],
    },
    "c": {
        "extensions": [".c", ".h"],
        "chunk_types": ["function_definition"],
    },
    "cpp": {
        "extensions": [".cpp", ".cc", ".cxx", ".hpp"],
        "chunk_types": ["function_definition", "class_specifier"],
    },
    "ocaml": {
        "extensions": [".ml", ".mli"],
        "chunk_types": ["value_definition", "type_definition", "module_definition"],
    },
}

# Build reverse lookup: extension → (language_name, config)
_EXT_TO_LANG: dict[str, tuple[str, dict]] = {}
for _lang, _cfg in LANG_CONFIG.items():
    for _ext in _cfg["extensions"]:
        _EXT_TO_LANG[_ext] = (_lang, _cfg)

# All supported extensions for globbing
SUPPORTED_EXTENSIONS: set[str] = set(_EXT_TO_LANG.keys())

# Parent-like node types (class/struct containers) across languages
_PARENT_TYPES = {
    "class_definition",      # python
    "class_declaration",     # js/ts/java
    "class_specifier",       # cpp
    "impl_item",             # rust
    "trait_item",            # rust
    "module_definition",     # ocaml
}


def _get_node_name(node) -> str | None:
    """Extract the name from a tree-sitter node.

    Tries `name` field first, then `type` field (for Rust impl),
    then `declarator.declarator` (for C/C++), then falls back
    to first identifier child.
    """
    # Most languages: name field
    name_node = node.child_by_field_name("name")
    if name_node:
        return name_node.text.decode("utf-8")

    # Rust impl_item: type field (only for impl_item to avoid matching
    # C/C++ function return types which also use a "type" field)
    if node.type == "impl_item":
        type_node = node.child_by_field_name("type")
        if type_node:
            return type_node.text.decode("utf-8")

    # C/C++ function_definition: declarator -> declarator (identifier)
    decl = node.child_by_field_name("declarator")
    if decl:
        inner = decl.child_by_field_name("declarator")
        if inner:
            return inner.text.decode("utf-8")

    # Go type_declaration: first type_spec child → first type_identifier child
    for child in node.children:
        if child.type == "type_spec":
            for gc in child.children:
                if gc.type == "type_identifier":
                    return gc.text.decode("utf-8")

    # OCaml: names live inside *_binding children (let_binding, type_binding, etc.)
    for child in node.children:
        if child.type.endswith("_binding"):
            for gc in child.children:
                if gc.type in ("value_name", "type_constructor", "module_name"):
                    return gc.text.decode("utf-8")

    # Last resort: first identifier child
    for child in node.children:
        if child.type == "identifier" or child.type == "type_identifier":
            return child.text.decode("utf-8")

    return None


def _find_parent_name(node) -> str | None:
    """Walk up from node to find an enclosing class/struct parent name."""
    current = node.parent
    while current:
        if current.type in _PARENT_TYPES:
            return _get_node_name(current)
        current = current.parent
    return None


def _node_kind(node_type: str) -> str:
    """Map tree-sitter node type to a human-readable kind string."""
    if "class" in node_type or node_type == "class_specifier":
        return "class"
    if "constructor" in node_type:
        return "constructor"
    if "struct" in node_type:
        return "struct"
    if "trait" in node_type:
        return "trait"
    if "impl" in node_type:
        return "impl"
    if "type" in node_type:
        return "type"
    if "method" in node_type:
        return "method"
    if node_type == "value_definition":
        return "function"
    if node_type == "module_definition":
        return "module"
    return "function"


def chunk_source_file(path: Path) -> list[dict]:
    """Extract function/class/struct chunks from a source file using tree-sitter."""
    try:
        from tree_sitter_language_pack import get_parser
    except ImportError:
        raise ImportError(
            "tree-sitter-language-pack is required for source indexing.\n"
            "Install it with: pip install \"dyf[source]\""
        )

    ext = path.suffix.lower()
    if ext not in _EXT_TO_LANG:
        return []

    language, config = _EXT_TO_LANG[ext]
    chunk_types = set(config["chunk_types"])

    source_bytes = path.read_bytes()
    parser = get_parser(language)
    tree = parser.parse(source_bytes)

    source_text = source_bytes.decode("utf-8", errors="replace")
    lines = source_text.splitlines()
    module_name = path.stem

    # Walk tree, collect matching nodes
    matches = []

    def walk(node):
        if node.type in chunk_types:
            matches.append(node)
        for child in node.children:
            walk(child)

    walk(tree.root_node)

    # Sort by line number
    matches.sort(key=lambda n: n.start_point[0])

    chunks = []
    for node in matches:
        start_line = node.start_point[0]
        end_line = node.end_point[0] + 1
        chunk_text = "\n".join(lines[start_line:end_line])

        name = _get_node_name(node)
        if not name:
            name = f"anonymous_{start_line + 1}"

        kind = _node_kind(node.type)

        parent_name = _find_parent_name(node)
        if parent_name:
            title = f"{module_name}.{parent_name}.{name}"
        else:
            title = f"{module_name}.{name}"

        embed_text = f"search_document: {language} {kind} {title}\n{chunk_text[:2000]}"

        chunks.append({
            "title": title,
            "file": path.name,
            "kind": kind,
            "line": start_line + 1,
            "language": language,
            "text": embed_text,
        })

    return chunks


def embed_batch(
    texts: list[str],
    ollama_url: str = DEFAULT_OLLAMA_URL,
    model: str = DEFAULT_MODEL,
    batch_size: int = 64,
) -> np.ndarray:
    """Embed texts using Ollama."""
    all_embeddings = []

    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        resp = requests.post(ollama_url, json={
            "model": model,
            "input": batch,
        })
        resp.raise_for_status()
        embs = resp.json()["embeddings"]
        all_embeddings.extend(embs)

        if (i + batch_size) % 128 == 0 or i + batch_size >= len(texts):
            print(f"  Embedded {min(i + batch_size, len(texts))}/{len(texts)}")

    return np.array(all_embeddings, dtype=np.float32)


def index_source(
    source_dir: Path,
    output: Path,
    model: str = DEFAULT_MODEL,
    ollama_url: str = DEFAULT_OLLAMA_URL,
    max_depth: int = 4,
    num_bits: int = 4,
    min_leaf_size: int = 5,
    seed: int = 42,
) -> None:
    """Index source code into a .dyf file."""
    print(f"Indexing source code")
    print(f"  Source: {source_dir}")
    print(f"  Output: {output}")
    print(f"  Model:  {model}")

    # Chunk all supported source files
    t0 = time.time()
    all_chunks = []
    langs_seen: set[str] = set()

    for ext in sorted(SUPPORTED_EXTENSIONS):
        for src_file in sorted(source_dir.rglob(f"*{ext}")):
            chunks = chunk_source_file(src_file)
            all_chunks.extend(chunks)
            if chunks:
                langs_seen.add(chunks[0]["language"])
                print(f"  {src_file.relative_to(source_dir)}: {len(chunks)} chunks")

    if not all_chunks:
        print("No source chunks found.")
        sys.exit(1)

    langs_str = ", ".join(sorted(langs_seen))
    print(f"\nTotal: {len(all_chunks)} chunks ({langs_str}) in {time.time()-t0:.1f}s")

    # Embed
    print(f"\nEmbedding with {model}...")
    t0 = time.time()
    texts = [c["text"] for c in all_chunks]
    embeddings = embed_batch(texts, ollama_url=ollama_url, model=model)
    print(f"  {embeddings.shape} in {time.time()-t0:.1f}s")

    # Normalize
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    embeddings = embeddings / np.where(norms > 0, norms, 1)

    # Build DYF tree
    print("\nBuilding DYF tree...")
    t0 = time.time()
    tree = build_dyf_tree(
        embeddings,
        max_depth=max_depth,
        num_bits=num_bits,
        min_leaf_size=min_leaf_size,
        seed=seed,
        fit_method="itq",
    )
    print(f"  Tree built in {time.time()-t0:.1f}s")

    # Write .dyf
    print("\nWriting .dyf...")
    t0 = time.time()
    titles = [c["title"] for c in all_chunks]
    files = [c["file"] for c in all_chunks]
    kinds = [c["kind"] for c in all_chunks]
    line_nums = [str(c["line"]) for c in all_chunks]
    languages = [c["language"] for c in all_chunks]

    write_lazy_index(
        tree,
        embeddings,
        str(output),
        compression="none",
        quantization="float16",
        metadata={
            "embedding_model": model,
            "domain": "source code",
            "chunk_method": "tree_sitter",
            "languages": langs_str,
        },
        build_params={
            "max_depth": max_depth,
            "num_bits": num_bits,
            "min_leaf_size": min_leaf_size,
            "seed": seed,
        },
        stored_fields={
            "title": titles,
            "file": files,
            "kind": kinds,
            "line": line_nums,
            "language": languages,
        },
    )
    size_mb = output.stat().st_size / (1024 * 1024)
    print(f"  Written {output.name} ({size_mb:.1f} MB) in {time.time()-t0:.1f}s")
    print(f"\nDone. {len(all_chunks)} chunks indexed.")


def main(argv: list[str] | None = None) -> int:
    """CLI entry point for `dyf index-source`."""
    parser = argparse.ArgumentParser(
        prog="dyf index-source",
        description="Index source code into a .dyf file (Python, JS, TS, Rust, Go, Java, C, C++, OCaml)",
    )
    parser.add_argument(
        "source_dir",
        type=Path,
        help="Directory containing source files",
    )
    parser.add_argument(
        "-o", "--output",
        type=Path,
        default=None,
        help="Output .dyf file path (default: <dirname>.dyf)",
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help=f"Ollama embedding model (default: {DEFAULT_MODEL})",
    )
    parser.add_argument(
        "--ollama-url",
        default=DEFAULT_OLLAMA_URL,
        help=f"Ollama API URL (default: {DEFAULT_OLLAMA_URL})",
    )
    parser.add_argument(
        "--max-depth", type=int, default=4,
        help="DYF tree max depth (default: 4)",
    )
    parser.add_argument(
        "--num-bits", type=int, default=4,
        help="LSH bits per level (default: 4)",
    )
    parser.add_argument(
        "--min-leaf-size", type=int, default=5,
        help="Minimum leaf size (default: 5)",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed (default: 42)",
    )

    args = parser.parse_args(argv)

    source_dir = args.source_dir.resolve()
    if not source_dir.is_dir():
        print(f"Error: {source_dir} is not a directory")
        return 1

    output = args.output
    if output is None:
        output = Path(f"{source_dir.name}.dyf")

    index_source(
        source_dir=source_dir,
        output=output.resolve(),
        model=args.model,
        ollama_url=args.ollama_url,
        max_depth=args.max_depth,
        num_bits=args.num_bits,
        min_leaf_size=args.min_leaf_size,
        seed=args.seed,
    )
    return 0
