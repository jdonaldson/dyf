"""
PubMed Abstract Embeddings Generator

Downloads a cross-cutting sample of PubMed abstracts and generates embeddings
for visualization with dyf.

Requirements:
    pip install sentence-transformers polars pyarrow requests

Usage:
    python pubmed_embeddings.py --n-samples 50000 --output pubmed_50k.parquet
"""

import argparse
import time
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import polars as pl
import requests


def fetch_pmids_by_year(year: int, retmax: int = 10000) -> list[str]:
    """Fetch PMIDs for a given year using E-utilities."""
    base_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
    params = {
        "db": "pubmed",
        "term": f"{year}[pdat] AND hasabstract[text]",
        "retmax": retmax,
        "retmode": "json",
        "sort": "relevance",  # Mix of topics
    }
    resp = requests.get(base_url, params=params, timeout=30)
    resp.raise_for_status()
    data = resp.json()
    return data.get("esearchresult", {}).get("idlist", [])


def fetch_abstracts(pmids: list[str], batch_size: int = 200) -> list[dict]:
    """Fetch abstract details for a list of PMIDs."""
    base_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
    records = []
    total_batches = (len(pmids) + batch_size - 1) // batch_size

    for i in range(0, len(pmids), batch_size):
        batch_num = i // batch_size + 1
        if batch_num % 25 == 0 or batch_num == 1:
            print(f"    Batch {batch_num}/{total_batches} ({len(records):,} abstracts so far)")
        batch = pmids[i:i + batch_size]
        params = {
            "db": "pubmed",
            "id": ",".join(batch),
            "rettype": "xml",
            "retmode": "xml",
        }

        try:
            resp = requests.get(base_url, params=params, timeout=60)
            resp.raise_for_status()
            root = ET.fromstring(resp.content)

            for article in root.findall(".//PubmedArticle"):
                try:
                    # Get PMID
                    pmid_elem = article.find(".//PMID")
                    pmid = pmid_elem.text if pmid_elem is not None else ""

                    # Get title
                    title_elem = article.find(".//ArticleTitle")
                    title = title_elem.text if title_elem is not None else ""
                    if not title:
                        continue

                    # Get abstract
                    abstract_parts = []
                    for abs_text in article.findall(".//AbstractText"):
                        if abs_text.text:
                            label = abs_text.get("Label", "")
                            if label:
                                abstract_parts.append(f"{label}: {abs_text.text}")
                            else:
                                abstract_parts.append(abs_text.text)

                    abstract = " ".join(abstract_parts)
                    if len(abstract) < 100:
                        continue

                    # Get MeSH terms
                    mesh_terms = []
                    for mesh in article.findall(".//MeshHeading/DescriptorName"):
                        if mesh.text:
                            mesh_terms.append(mesh.text)

                    records.append({
                        "pmid": pmid,
                        "title": title,
                        "abstract": abstract[:2000],
                        "mesh_primary": mesh_terms[0] if mesh_terms else "Unknown",
                        "mesh_terms": "|".join(mesh_terms[:5]),
                    })

                except Exception:
                    continue

            # Rate limiting - NCBI allows 3 requests/second without API key
            time.sleep(0.4)

        except Exception as e:
            print(f"  Error fetching batch: {e}")
            time.sleep(1)

    return records


def load_pubmed_sample(n_samples: int, seed: int = 42) -> pl.DataFrame:
    """Load a cross-cutting sample of PubMed abstracts via E-utilities."""
    print(f"Fetching {n_samples:,} PubMed abstracts via E-utilities API...")

    rng = np.random.default_rng(seed)

    # Sample across multiple years for cross-cutting coverage
    years = list(range(2015, 2026))
    samples_per_year = n_samples // len(years) + 1

    all_pmids = []
    for year in years:
        print(f"  Fetching PMIDs for {year}...")
        pmids = fetch_pmids_by_year(year, retmax=samples_per_year * 2)
        # Random sample from each year
        if len(pmids) > samples_per_year:
            pmids = rng.choice(pmids, samples_per_year, replace=False).tolist()
        all_pmids.extend(pmids)
        time.sleep(0.4)

    # Shuffle and trim to target
    rng.shuffle(all_pmids)
    all_pmids = all_pmids[:n_samples]

    print(f"  Fetching abstracts for {len(all_pmids):,} PMIDs...")
    records = fetch_abstracts(all_pmids)

    print(f"  Collected {len(records):,} abstracts")
    return pl.DataFrame(records)


def embed_abstracts(
    df: pl.DataFrame,
    model_name: str = "NeuML/pubmedbert-base-embeddings",
    batch_size: int = 32,
) -> pl.DataFrame:
    """Generate embeddings for abstracts."""
    from sentence_transformers import SentenceTransformer

    print(f"Loading embedding model: {model_name}")
    model = SentenceTransformer(model_name)

    # Combine title + abstract for embedding
    texts = [
        f"{row['title']}. {row['abstract']}"
        for row in df.iter_rows(named=True)
    ]

    print(f"Generating embeddings for {len(texts):,} abstracts...")
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        normalize_embeddings=True,
    )

    print(f"  Embedding dimension: {embeddings.shape[1]}")

    # Add embeddings to dataframe
    df = df.with_columns(
        pl.Series("embedding", [e.tolist() for e in embeddings])
    )

    return df


def main():
    parser = argparse.ArgumentParser(description="Generate PubMed abstract embeddings")
    parser.add_argument("--n-samples", type=int, default=50000, help="Number of abstracts to sample")
    parser.add_argument("--output", "-o", default="pubmed_50k.parquet", help="Output parquet file")
    parser.add_argument("--model", default="NeuML/pubmedbert-base-embeddings", help="Embedding model")
    parser.add_argument("--batch-size", type=int, default=32, help="Embedding batch size")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    # Load sample
    df = load_pubmed_sample(args.n_samples, seed=args.seed)

    if len(df) == 0:
        print("ERROR: No abstracts collected. Check dataset access.")
        return

    # Show sample stats
    print(f"\nSample statistics:")
    print(f"  Total abstracts: {len(df):,}")
    mesh_counts = df.group_by("mesh_primary").len().sort("len", descending=True)
    print(f"  Unique primary MeSH terms: {len(mesh_counts):,}")
    print(f"  Top 10 MeSH terms:")
    for row in mesh_counts.head(10).iter_rows(named=True):
        print(f"    {row['mesh_primary']}: {row['len']:,}")

    # Generate embeddings
    df = embed_abstracts(df, model_name=args.model, batch_size=args.batch_size)

    # Save
    output_path = Path(args.output)
    df.write_parquet(output_path)
    print(f"\nSaved to {output_path} ({output_path.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
