"""
CIFAR-100 Image Embeddings via CLIP

Embeds CIFAR-100 images using CLIP (openai/clip-vit-base-patch32) for ROG testing.
CIFAR-100 has 100 fine classes grouped into 20 coarse classes — a good test for
hierarchical clustering and disambiguation.

Requirements:
    pip install torch torchvision transformers polars pyarrow

Usage:
    python demo/cifar_embeddings.py
    python demo/cifar_embeddings.py --split train --output demo/cifar100_train_embeddings.parquet
"""

import argparse
from pathlib import Path

import numpy as np
import polars as pl
import torch
from torchvision.datasets import CIFAR100
from transformers import CLIPModel, CLIPProcessor


# CIFAR-100 coarse class names (official order matching coarse label indices 0-19)
COARSE_LABELS = [
    "aquatic mammals", "fish", "flowers", "food containers",
    "fruit and vegetables", "household electrical devices",
    "household furniture", "insects", "large carnivores",
    "large man-made outdoor things", "large natural outdoor scenes",
    "large omnivores and herbivores", "medium-sized mammals",
    "non-insect invertebrates", "people", "reptiles",
    "small mammals", "trees", "vehicles 1", "vehicles 2",
]

# Mapping from fine label index (0-99) to coarse label index (0-19)
# Standard mapping from CIFAR-100 specification
FINE_TO_COARSE = np.array([
     4,  1, 14,  8,  0,  6,  7,  7, 18,  3,
     3, 14,  9, 18,  7, 11,  3,  9,  7, 11,
     6, 11,  5, 10,  7,  6, 13, 15,  3, 15,
     0, 11,  1, 10, 12, 14, 16,  9, 11,  5,
     5, 19,  8,  8, 15, 13, 14, 17, 18, 10,
    16,  4, 17,  4,  2,  0, 17,  4, 18, 17,
    10,  3,  2, 12, 12, 16, 12,  1,  9, 19,
     2, 10,  0,  1, 16, 12,  9, 13, 15, 13,
    16, 19,  2,  4,  6, 19,  5,  5,  8, 19,
    18,  1,  2, 15,  6,  0, 17,  8, 14, 13,
])


def main():
    parser = argparse.ArgumentParser(description="Embed CIFAR-100 images with CLIP")
    parser.add_argument("--split", default="test", choices=["train", "test"])
    parser.add_argument("--output", default="demo/cifar100_embeddings.parquet")
    parser.add_argument("--batch-size", type=int, default=128)
    args = parser.parse_args()

    is_train = args.split == "train"
    dataset = CIFAR100(root="~/.cache/torchvision", train=is_train, download=True)

    print(f"Loaded CIFAR-100 {args.split} split: {len(dataset)} images")
    print(f"  Fine classes: {len(dataset.classes)}")
    print(f"  Coarse classes: {len(COARSE_LABELS)}")

    # Load CLIP
    model_name = "openai/clip-vit-base-patch32"
    print(f"Loading CLIP model: {model_name}")
    model = CLIPModel.from_pretrained(model_name)
    processor = CLIPProcessor.from_pretrained(model_name)

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model = model.to(device).eval()
    print(f"  Device: {device}")

    fine_labels = dataset.classes

    # Embed in batches
    titles = []
    embeddings = []
    images_batch = []

    for idx in range(len(dataset)):
        img, fine_idx = dataset[idx]
        fine_name = fine_labels[fine_idx]
        coarse_idx = int(FINE_TO_COARSE[fine_idx])
        coarse_name = COARSE_LABELS[coarse_idx]
        title = f"{fine_name} ({coarse_name}) #{idx}"

        titles.append(title)
        images_batch.append(img)

        if len(images_batch) == args.batch_size or idx == len(dataset) - 1:
            inputs = processor(images=images_batch, return_tensors="pt", padding=True)
            inputs = {k: v.to(device) for k, v in inputs.items()}

            with torch.no_grad():
                emb = model.get_image_features(**inputs)
                emb = emb / emb.norm(dim=-1, keepdim=True)  # L2 normalize

            embeddings.append(emb.cpu().numpy())
            images_batch = []

            if (idx + 1) % 1000 == 0:
                print(f"  Embedded {idx + 1}/{len(dataset)}")

    all_embeddings = np.concatenate(embeddings, axis=0)
    print(f"  Final embedding shape: {all_embeddings.shape}")

    # Save parquet with title + embedding columns (matches ROG pipeline expectation)
    df = pl.DataFrame({
        "title": titles,
        "embedding": all_embeddings.tolist(),
    })

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.write_parquet(output_path)
    print(f"Saved {len(df)} images to {output_path}")
    print(f"  Embedding dim: {all_embeddings.shape[1]}")
    print(f"  File size: {output_path.stat().st_size / 1e6:.1f} MB")


if __name__ == "__main__":
    main()
