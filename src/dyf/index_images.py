"""
Index images into a .dyf file.

Embeds images using a HuggingFace vision model (default: nomic-embed-vision-v1.5),
generates thumbnails as base64 WebP data URIs, builds a DYF tree, and writes a .dyf index.

Usage (via CLI):
    dyf index-images ~/photos/ -o photos.dyf
    dyf index-images . -o images.dyf --model nomic-ai/nomic-embed-vision-v1.5

Requires: pip install "dyf[vision]"

⚠ **The `title` stored field is the filename**, not a description of the image. The
embeddings are genuinely visual, but anything downstream that reads `title` — LLM cluster
labelling, TF-IDF keywording, the browser tour in `dyfviz` — is reading filenames and has
never seen a pixel. On a corpus of `IMG_4821.jpg` names that degrades quietly rather than
failing, which is the worst way for it to go wrong. Pass better titles by writing the
index yourself if the labels matter.

Exit codes (see `_ingest_errors`): 0 ok, 1 nothing to index, 2 bad request,
3 dependency or service unavailable.
"""

import argparse
import base64
import io
import logging
import time
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

from ._ingest_errors import BadIngestRequest, EmptyIngestError, IngestError
from ._preview import IngestPreview, batches_for
from .dyf_tree import build_dyf_tree
from .lazy_index import write_lazy_index

DEFAULT_MODEL = "nomic-ai/nomic-embed-vision-v1.5"
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".gif", ".bmp", ".tiff", ".tif"}


def load_vision_model(model_name: str = DEFAULT_MODEL, device: str | None = None):
    """Load a HuggingFace vision model and processor.

    Returns (processor, model, device_str).
    """
    try:
        import torch
        from transformers import AutoModel, AutoProcessor
    except ImportError:
        raise ImportError(
            'transformers and torch are required for image indexing.\nInstall them with: pip install "dyf[vision]"'
        )

    if device is None:
        if torch.backends.mps.is_available():
            device = "mps"
        elif torch.cuda.is_available():
            device = "cuda"
        else:
            device = "cpu"

    processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModel.from_pretrained(model_name, trust_remote_code=True).to(device)
    model.eval()
    return processor, model, device


def make_thumbnail(img, max_size: int = 128) -> str:
    """Create a base64 WebP data URI thumbnail from a PIL Image.

    Resizes to fit within max_size x max_size, preserving aspect ratio.
    """
    try:
        from PIL import Image
    except ImportError:
        raise ImportError('Pillow is required: pip install "dyf[vision]"')

    img_copy = img.copy()
    img_copy.thumbnail((max_size, max_size), Image.LANCZOS)
    if img_copy.mode == "RGBA":
        img_copy = img_copy.convert("RGB")

    buf = io.BytesIO()
    img_copy.save(buf, format="WEBP", quality=80)
    b64 = base64.b64encode(buf.getvalue()).decode("ascii")
    return f"data:image/webp;base64,{b64}"


def embed_images(
    images: list,
    processor,
    model,
    device: str,
    batch_size: int = 16,
) -> np.ndarray:
    """Embed a list of PIL Images using the vision model.

    Returns (N, D) float32 array.
    """
    import torch

    all_embeddings = []

    for i in range(0, len(images), batch_size):
        batch = images[i : i + batch_size]
        inputs = processor(images=batch, return_tensors="pt").to(device)

        with torch.no_grad():
            outputs = model(**inputs)

        # Use CLS token or pooled output
        if hasattr(outputs, "pooler_output") and outputs.pooler_output is not None:
            embs = outputs.pooler_output.cpu().numpy()
        else:
            embs = outputs.last_hidden_state[:, 0].cpu().numpy()

        all_embeddings.append(embs)

        done = min(i + batch_size, len(images))
        if done % 32 == 0 or done == len(images):
            logger.info(f"  Embedded {done}/{len(images)}")

    return np.concatenate(all_embeddings, axis=0).astype(np.float32)


def scan_images(source_dir: Path) -> list[Path]:
    """Recursively find image files in source_dir."""
    results = []
    for p in sorted(source_dir.rglob("*")):
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS:
            results.append(p)
    return results


def preview_images(
    source_dir: Path,
    output: Path,
    model: str = DEFAULT_MODEL,
    batch_size: int = 16,
) -> IngestPreview:
    """Report what `index_images` would do, without loading the model or embedding.

    Scanning is free; decoding every image to check it is valid is not, and neither is
    downloading a vision model. So the file count is exact and the *usable* count is not
    known until the real run — stated rather than guessed at.
    """
    paths = scan_images(source_dir)
    by_ext: dict[str, int] = {}
    for p in paths:
        by_ext[p.suffix.lower()] = by_ext.get(p.suffix.lower(), 0) + 1

    notes = []
    if paths:
        spread = ", ".join(f"{ext} {n}" for ext, n in sorted(by_ext.items()))
        notes.append(f"by extension: {spread}")
        notes.append("some files may fail to decode; the usable count is only known after a real run")
    total_mb = sum(p.stat().st_size for p in paths) / 1_048_576 if paths else 0.0

    return IngestPreview(
        command="index-images",
        source=str(source_dir),
        output=str(output),
        model=model,
        batch_size=batch_size,
        counts={"images": len(paths), "megabytes": int(total_mb)},
        batches=batches_for(len(paths), batch_size),
        notes=notes,
    )


def index_images(
    source_dir: Path,
    output: Path,
    model: str = DEFAULT_MODEL,
    max_depth: int = 4,
    num_bits: int = 4,
    min_leaf_size: int = 5,
    seed: int = 42,
    batch_size: int = 16,
) -> None:
    """Index images into a .dyf file."""
    from PIL import Image

    logger.info("Indexing images")
    logger.info(f"  Source: {source_dir}")
    logger.info(f"  Output: {output}")
    logger.info(f"  Model:  {model}")

    # Scan for images
    t0 = time.time()
    image_paths = scan_images(source_dir)
    if not image_paths:
        raise EmptyIngestError(
            f"no images found under {source_dir}.\n  Recognised extensions: {', '.join(sorted(IMAGE_EXTENSIONS))}"
        )
    logger.info(f"Found {len(image_paths)} images in {time.time() - t0:.1f}s")

    # Load vision model
    logger.info("Loading vision model...")
    t0 = time.time()
    processor, vision_model, device = load_vision_model(model)
    logger.info(f"  Model loaded on {device} in {time.time() - t0:.1f}s")

    # Load images, generate thumbnails, collect metadata
    logger.info("Loading images and generating thumbnails...")
    t0 = time.time()
    images = []
    titles = []
    thumbnails = []
    files = []
    widths = []
    heights = []
    valid_paths = []

    for path in image_paths:
        try:
            img = Image.open(path)
            img.load()  # Force load to catch corrupt files
            if img.mode not in ("RGB", "RGBA", "L"):
                img = img.convert("RGB")

            images.append(img)
            titles.append(path.name)
            thumbnails.append(make_thumbnail(img))
            files.append(str(path.relative_to(source_dir)))
            widths.append(img.width)
            heights.append(img.height)
            valid_paths.append(path)
        except Exception as e:
            logger.warning("Skipping %s: %s", path.name, e)

    if not images:
        # Distinct from "no images found": files matched, but every one failed to decode.
        raise EmptyIngestError(
            f"found {len(image_paths)} image file(s) under {source_dir}, but none could be "
            f"decoded.\n  Re-run with -v to see the per-file errors."
        )

    logger.info(f"  Loaded {len(images)} images in {time.time() - t0:.1f}s")

    # Embed
    logger.info("Embedding images...")
    t0 = time.time()
    embeddings = embed_images(images, processor, vision_model, device, batch_size)
    logger.info(f"  {embeddings.shape} in {time.time() - t0:.1f}s")

    # Normalize
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    embeddings = embeddings / np.where(norms > 0, norms, 1)

    # Build DYF tree
    logger.info("Building DYF tree...")
    t0 = time.time()
    tree = build_dyf_tree(
        embeddings,
        max_depth=max_depth,
        num_bits=num_bits,
        min_leaf_size=min_leaf_size,
        seed=seed,
        fit_method="itq",
    )
    logger.info(f"  Tree built in {time.time() - t0:.1f}s")

    # Write .dyf
    logger.info("Writing .dyf...")
    t0 = time.time()
    write_lazy_index(
        tree,
        embeddings,
        str(output),
        compression="none",
        quantization="float16",
        metadata={
            "embedding_model": model,
            "domain": "images",
            "thumbnail_size": "128x128",
            "thumbnail_format": "webp",
        },
        build_params={
            "max_depth": max_depth,
            "num_bits": num_bits,
            "min_leaf_size": min_leaf_size,
            "seed": seed,
        },
        stored_fields={
            "title": titles,
            "thumbnail": thumbnails,
            "file": files,
            "width": widths,
            "height": heights,
        },
    )
    size_mb = output.stat().st_size / (1024 * 1024)
    logger.info(f"  Written {output.name} ({size_mb:.1f} MB) in {time.time() - t0:.1f}s")
    logger.info(f"Done. {len(images)} images indexed.")


def main(argv: list[str] | None = None) -> int:
    """CLI entry point for `dyf index-images`."""
    parser = argparse.ArgumentParser(
        prog="dyf index-images",
        description="Index images into a .dyf file using vision embeddings",
    )
    parser.add_argument(
        "source_dir",
        type=Path,
        help="Directory containing images",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        help="Output .dyf file path (default: <dirname>.dyf)",
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help=f"HuggingFace vision model (default: {DEFAULT_MODEL})",
    )
    parser.add_argument(
        "--max-depth",
        type=int,
        default=4,
        help="DYF tree max depth (default: 4)",
    )
    parser.add_argument(
        "--num-bits",
        type=int,
        default=4,
        help="LSH bits per level (default: 4)",
    )
    parser.add_argument(
        "--min-leaf-size",
        type=int,
        default=5,
        help="Minimum leaf size (default: 5)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (default: 42)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help="Embedding batch size (default: 16)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Report what would be indexed and stop. Loads no model, writes nothing.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        dest="as_json",
        help="With --dry-run, emit JSON (schema_version 0 — unstable before v1)",
    )

    args = parser.parse_args(argv)

    source_dir = args.source_dir.resolve()
    if not source_dir.is_dir():
        logger.error("not a directory: %s", source_dir)
        return BadIngestRequest.exit_code

    output = args.output
    if output is None:
        output = Path(f"{source_dir.name}.dyf")

    if args.dry_run:
        return preview_images(
            source_dir=source_dir,
            output=output.resolve(),
            model=args.model,
            batch_size=args.batch_size,
        ).emit(args.as_json, logger)

    try:
        index_images(
            source_dir=source_dir,
            output=output.resolve(),
            model=args.model,
            max_depth=args.max_depth,
            num_bits=args.num_bits,
            min_leaf_size=args.min_leaf_size,
            seed=args.seed,
            batch_size=args.batch_size,
        )
    except IngestError as exc:
        for line in str(exc).splitlines():
            logger.error("%s", line)
        return exc.exit_code
    return 0
