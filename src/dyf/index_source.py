"""
Index Python source code into a .dyf file.

Chunks at function/class boundaries using AST, embeds via Ollama,
builds a DYF tree, and writes a .dyf index.

Usage (via CLI):
    dyf index-source src/mypackage/ -o mypackage.dyf
    dyf index-source . -o project.dyf --model nomic-embed-text
"""

import ast
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


def chunk_python_file(path: Path) -> list[dict]:
    """Extract function/class chunks from a Python file using AST."""
    source = path.read_text()
    lines = source.splitlines()

    try:
        tree = ast.parse(source)
    except SyntaxError:
        return []

    chunks = []
    module_name = path.stem

    defs = []
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            defs.append(node)

    defs.sort(key=lambda n: n.lineno)

    for node in defs:
        start = node.lineno - 1
        end = node.end_lineno if node.end_lineno else start + 1
        chunk_lines = lines[start:end]
        text = "\n".join(chunk_lines)

        kind = "class" if isinstance(node, ast.ClassDef) else "def"

        parent = _find_parent(tree, node)
        if parent:
            title = f"{module_name}.{parent.name}.{node.name}"
        else:
            title = f"{module_name}.{node.name}"

        embed_text = f"search_document: python {kind} {title}\n{text[:2000]}"

        chunks.append({
            "title": title,
            "file": path.name,
            "kind": kind,
            "line": node.lineno,
            "text": embed_text,
        })

    return chunks


def _find_parent(tree, target_node):
    """Find the parent class/function of a node, if any."""
    for node in ast.walk(tree):
        if isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            for child in ast.iter_child_nodes(node):
                if child is target_node:
                    return node
    return None


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
    """Index Python source into a .dyf file."""
    print(f"Indexing Python source")
    print(f"  Source: {source_dir}")
    print(f"  Output: {output}")
    print(f"  Model:  {model}")

    # Chunk all Python files
    t0 = time.time()
    all_chunks = []
    for py_file in sorted(source_dir.rglob("*.py")):
        chunks = chunk_python_file(py_file)
        all_chunks.extend(chunks)
        if chunks:
            print(f"  {py_file.relative_to(source_dir)}: {len(chunks)} chunks")

    if not all_chunks:
        print("No Python chunks found.")
        sys.exit(1)

    print(f"\nTotal: {len(all_chunks)} chunks in {time.time()-t0:.1f}s")

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

    write_lazy_index(
        tree,
        embeddings,
        str(output),
        compression="zstd",
        quantization="float16",
        metadata={
            "embedding_model": model,
            "domain": "python source code",
            "chunk_method": "ast_function_class",
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
        },
    )
    size_mb = output.stat().st_size / (1024 * 1024)
    print(f"  Written {output.name} ({size_mb:.1f} MB) in {time.time()-t0:.1f}s")
    print(f"\nDone. {len(all_chunks)} chunks indexed.")


def main(argv: list[str] | None = None) -> int:
    """CLI entry point for `dyf index-source`."""
    parser = argparse.ArgumentParser(
        prog="dyf index-source",
        description="Index Python source code into a .dyf file",
    )
    parser.add_argument(
        "source_dir",
        type=Path,
        help="Directory containing Python source files",
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
