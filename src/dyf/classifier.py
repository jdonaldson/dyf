"""
Density Classifier: Discover structure in embedding spaces

Returns raw density metrics per item - classification is up to you:
- bucket_id: LSH bucket assignment
- bucket_size: Number of items in the bucket
- centroid_similarity: Cosine similarity to bucket centroid (0-1)
- isolation_score: How isolated the item is (top_k_sim - median_sim)
- stability_score: How stable bucket assignment is across multiple seeds (0-1)

Example:
    >>> classifier = DensityClassifier(embedding_dim=384)
    >>> classifier.fit(embeddings)
    >>> print(classifier.report())

Configs:
    >>> from dyf import EmbedderConfig, LabelerConfig
    >>> embedder = EmbedderConfig.MEDIUM  # all-mpnet-base-v2
    >>> labeler = LabelerConfig.LOW       # phi3:mini
"""

import numpy as np
from typing import List, Dict, Optional, Tuple, TYPE_CHECKING
from enum import Enum

if TYPE_CHECKING:
    import polars as pl
from dataclasses import dataclass
from collections import Counter, defaultdict
from sklearn.decomposition import PCA


# =============================================================================
# Configuration Classes
# =============================================================================

@dataclass
class EmbedderConfig:
    """
    Configuration for text embedding models.

    Presets available as class attributes:
        - TFIDF: Built-in TF-IDF + SVD (no model download)
        - LOW: all-MiniLM-L6-v2 (80MB, 384d)
        - MEDIUM: all-mpnet-base-v2 (420MB, 768d)
        - MEDIUM_BGE: BAAI/bge-base-en-v1.5 (440MB, 768d)
        - HIGH: BAAI/bge-large-en-v1.5 (1.3GB, 1024d)
        - OPENAI: text-embedding-3-large (API, 3072d)

    Example:
        >>> config = EmbedderConfig.MEDIUM
        >>> embeddings = config.embed(texts)
    """
    name: str
    model_id: str
    dim: int
    size_mb: int
    provider: str  # 'tfidf', 'sentence-transformers', 'openai'
    description: str = ""

    def embed(self, texts: List[str], batch_size: int = 32, verbose: bool = True) -> np.ndarray:
        """Generate embeddings for texts using this config."""
        if self.provider == 'tfidf':
            return self._embed_tfidf(texts, verbose)
        elif self.provider == 'bm25':
            return self._embed_bm25(texts, verbose)
        elif self.provider == 'sentence-transformers':
            return self._embed_sentence_transformers(texts, batch_size, verbose)
        elif self.provider == 'openai':
            return self._embed_openai(texts, batch_size, verbose)
        else:
            raise ValueError(f"Unknown provider: {self.provider}")

    def _embed_tfidf(self, texts: List[str], verbose: bool) -> np.ndarray:
        """TF-IDF + SVD embeddings."""
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.decomposition import TruncatedSVD

        if verbose:
            print(f"Building TF-IDF embeddings ({len(texts):,} texts)...")

        vectorizer = TfidfVectorizer(
            max_features=10000, min_df=2, max_df=0.95,
            ngram_range=(1, 2), stop_words='english'
        )
        tfidf = vectorizer.fit_transform(texts)

        n_components = min(self.dim, tfidf.shape[1] - 1, len(texts) - 1)
        svd = TruncatedSVD(n_components=n_components, random_state=42)
        embeddings = svd.fit_transform(tfidf).astype(np.float32)

        if verbose:
            print(f"  Shape: {embeddings.shape}, variance: {svd.explained_variance_ratio_.sum():.1%}")

        return embeddings

    def _embed_bm25(self, texts: List[str], verbose: bool) -> np.ndarray:
        """BM25-weighted + SVD embeddings (saturated term frequencies)."""
        from sklearn.feature_extraction.text import CountVectorizer
        from sklearn.decomposition import TruncatedSVD
        from scipy import sparse

        if verbose:
            print(f"Building BM25 embeddings ({len(texts):,} texts)...")

        # BM25 parameters
        k1 = 1.5  # term frequency saturation
        b = 0.75  # length normalization

        # Get raw term counts
        count_vectorizer = CountVectorizer(
            max_features=10000, min_df=2, max_df=0.95,
            ngram_range=(1, 2), stop_words='english'
        )
        tf_matrix = count_vectorizer.fit_transform(texts)

        # Compute document lengths and average
        doc_lengths = np.array(tf_matrix.sum(axis=1)).flatten()
        avg_dl = doc_lengths.mean()

        if verbose:
            print(f"  Vocabulary: {len(count_vectorizer.vocabulary_):,}, avg doc len: {avg_dl:.1f}")

        # Apply BM25 saturation
        tf_saturated = sparse.lil_matrix(tf_matrix.shape, dtype=np.float32)

        for i in range(tf_matrix.shape[0]):
            row = tf_matrix.getrow(i)
            dl = doc_lengths[i]
            length_norm = 1 - b + b * (dl / avg_dl)

            for j in row.indices:
                tf = row[0, j]
                tf_sat = (tf * (k1 + 1)) / (tf + k1 * length_norm)
                tf_saturated[i, j] = tf_sat

        tf_saturated = tf_saturated.tocsr()

        # Apply IDF weighting
        n_docs = tf_matrix.shape[0]
        doc_freq = np.array((tf_matrix > 0).sum(axis=0)).flatten()
        idf = np.log((n_docs + 1) / (doc_freq + 1)) + 1

        bm25_matrix = tf_saturated.multiply(idf)

        # SVD reduction
        n_components = min(self.dim, bm25_matrix.shape[1] - 1, len(texts) - 1)
        svd = TruncatedSVD(n_components=n_components, random_state=42)
        embeddings = svd.fit_transform(bm25_matrix).astype(np.float32)

        if verbose:
            print(f"  Shape: {embeddings.shape}, variance: {svd.explained_variance_ratio_.sum():.1%}")

        return embeddings

    def _embed_sentence_transformers(self, texts: List[str], batch_size: int, verbose: bool) -> np.ndarray:
        """Sentence-transformers embeddings."""
        from sentence_transformers import SentenceTransformer

        if verbose:
            print(f"Loading {self.model_id}...")

        model = SentenceTransformer(self.model_id)
        embeddings = model.encode(
            texts, batch_size=batch_size,
            show_progress_bar=verbose,
            convert_to_numpy=True
        )
        return embeddings.astype(np.float32)

    def _embed_openai(self, texts: List[str], batch_size: int, verbose: bool) -> np.ndarray:
        """OpenAI API embeddings."""
        from openai import OpenAI
        import os

        client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
        all_embeddings = []

        if verbose:
            print(f"Calling OpenAI API ({len(texts):,} texts)...")

        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            response = client.embeddings.create(model=self.model_id, input=batch)
            batch_embeddings = [e.embedding for e in response.data]
            all_embeddings.extend(batch_embeddings)

            if verbose and (i + batch_size) % 1000 == 0:
                print(f"  {min(i + batch_size, len(texts)):,}/{len(texts):,}")

        return np.array(all_embeddings, dtype=np.float32)


# Embedder presets
EmbedderConfig.TFIDF = EmbedderConfig(
    name="tfidf", model_id="tfidf+svd", dim=128, size_mb=0,
    provider="tfidf", description="Built-in TF-IDF + SVD, no download"
)
EmbedderConfig.BM25 = EmbedderConfig(
    name="bm25", model_id="bm25+svd", dim=128, size_mb=0,
    provider="bm25", description="BM25 saturation + SVD, no download"
)
EmbedderConfig.LOW = EmbedderConfig(
    name="low", model_id="all-MiniLM-L6-v2", dim=384, size_mb=80,
    provider="sentence-transformers", description="Fast, good quality"
)
EmbedderConfig.MEDIUM = EmbedderConfig(
    name="medium", model_id="all-mpnet-base-v2", dim=768, size_mb=420,
    provider="sentence-transformers", description="Better semantic understanding"
)
EmbedderConfig.MEDIUM_BGE = EmbedderConfig(
    name="medium-bge", model_id="BAAI/bge-base-en-v1.5", dim=768, size_mb=440,
    provider="sentence-transformers", description="Strong retrieval performance"
)
EmbedderConfig.HIGH = EmbedderConfig(
    name="high", model_id="BAAI/bge-large-en-v1.5", dim=1024, size_mb=1300,
    provider="sentence-transformers", description="Best open-source"
)
EmbedderConfig.OPENAI = EmbedderConfig(
    name="openai", model_id="text-embedding-3-large", dim=3072, size_mb=0,
    provider="openai", description="OpenAI API, best overall"
)


@dataclass
class LabelerConfig:
    """
    Configuration for LLM-based bucket labeling.

    Presets available as class attributes:
        - KEYWORDS: Built-in TF-IDF keywords (no LLM)
        - LOW: phi3:mini / Phi-3-mini-4k (3.8B)
        - LOW_QWEN: qwen2.5:1.5b (1.5B, fastest)
        - MEDIUM: qwen2.5:7b (7B, good balance)
        - MEDIUM_LLAMA: llama3.1:8b (8B)
        - HIGH: qwen2.5:14b (14B, best local)

    Example:
        >>> config = LabelerConfig.MEDIUM
        >>> labels = classifier.label_buckets(**config.as_kwargs())
    """
    name: str
    model_id: str
    size_b: float  # billions of parameters
    provider: str  # 'keywords', 'ollama', 'mlx'
    ollama_name: str = ""
    mlx_name: str = ""
    base_url: str = "http://localhost:11434/v1"
    description: str = ""

    def as_kwargs(self, use_mlx: bool = False) -> Dict:
        """Get kwargs for label_buckets() method."""
        if self.provider == 'keywords':
            return {'_use_keywords': True}

        model = self.mlx_name if use_mlx else self.ollama_name
        url = "http://localhost:8080/v1" if use_mlx else self.base_url

        return {
            'base_url': url,
            'model': model
        }

    def install_cmd(self, use_mlx: bool = False) -> str:
        """Get command to install/pull this model."""
        if self.provider == 'keywords':
            return "# No installation needed"
        if use_mlx:
            return f"pip install mlx-lm && python -c \"from mlx_lm import load; load('{self.mlx_name}')\""
        return f"ollama pull {self.ollama_name}"

    def serve_cmd(self, use_mlx: bool = False) -> str:
        """Get command to start serving this model."""
        if self.provider == 'keywords':
            return "# No server needed"
        if use_mlx:
            return f"mlx_lm.server --model {self.mlx_name} --port 8080"
        return "ollama serve"


# Labeler presets
LabelerConfig.KEYWORDS = LabelerConfig(
    name="keywords", model_id="tfidf", size_b=0,
    provider="keywords", description="Built-in TF-IDF keywords, no LLM"
)
LabelerConfig.LOW = LabelerConfig(
    name="low", model_id="phi3-mini", size_b=3.8,
    provider="ollama", ollama_name="phi3:mini",
    mlx_name="mlx-community/Phi-3-mini-4k-instruct-4bit",
    description="Fast, small footprint"
)
LabelerConfig.LOW_QWEN = LabelerConfig(
    name="low-qwen", model_id="qwen2.5-1.5b", size_b=1.5,
    provider="ollama", ollama_name="qwen2.5:1.5b",
    mlx_name="mlx-community/Qwen2.5-1.5B-Instruct-4bit",
    description="Smallest, fastest"
)
LabelerConfig.MEDIUM = LabelerConfig(
    name="medium", model_id="qwen2.5-7b", size_b=7,
    provider="ollama", ollama_name="qwen2.5:7b",
    mlx_name="mlx-community/Qwen2.5-7B-Instruct-4bit",
    description="Good balance of speed/quality"
)
LabelerConfig.MEDIUM_LLAMA = LabelerConfig(
    name="medium-llama", model_id="llama3.1-8b", size_b=8,
    provider="ollama", ollama_name="llama3.1:8b",
    mlx_name="mlx-community/Llama-3.1-8B-Instruct-4bit",
    description="Strong general purpose"
)
LabelerConfig.HIGH = LabelerConfig(
    name="high", model_id="qwen2.5-14b", size_b=14,
    provider="ollama", ollama_name="qwen2.5:14b",
    mlx_name="mlx-community/Qwen2.5-14B-Instruct-4bit",
    description="Best local quality"
)


def list_configs():
    """Print available embedder and labeler configurations."""
    print("=" * 70)
    print("EMBEDDER CONFIGURATIONS")
    print("=" * 70)
    print(f"{'Name':<12} {'Model':<30} {'Dim':>6} {'Size':>8} {'Provider':<20}")
    print("-" * 70)
    for cfg in [EmbedderConfig.TFIDF, EmbedderConfig.BM25, EmbedderConfig.LOW,
                EmbedderConfig.MEDIUM, EmbedderConfig.MEDIUM_BGE, EmbedderConfig.HIGH,
                EmbedderConfig.OPENAI]:
        size = f"{cfg.size_mb}MB" if cfg.size_mb > 0 else "API/0"
        print(f"{cfg.name:<12} {cfg.model_id:<30} {cfg.dim:>6} {size:>8} {cfg.provider:<20}")

    print()
    print("=" * 70)
    print("LABELER CONFIGURATIONS")
    print("=" * 70)
    print(f"{'Name':<12} {'Model':<20} {'Size':>6} {'Ollama':<20} {'MLX':<35}")
    print("-" * 70)
    for cfg in [LabelerConfig.KEYWORDS, LabelerConfig.LOW_QWEN, LabelerConfig.LOW,
                LabelerConfig.MEDIUM, LabelerConfig.MEDIUM_LLAMA, LabelerConfig.HIGH]:
        size = f"{cfg.size_b}B" if cfg.size_b > 0 else "-"
        print(f"{cfg.name:<12} {cfg.model_id:<20} {size:>6} {cfg.ollama_name or '-':<20} {cfg.mlx_name or '-':<35}")


# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class BridgeCluster:
    """Legacy class - kept for backwards compatibility but no longer used."""
    cluster_id: int
    size: int
    coherence: float
    global_indices: List[int]
    source_buckets: Dict[int, int]
    dominant_category: str
    category_counts: List[Tuple[str, int]]


@dataclass
class DensityReport:
    """Report on density classification."""
    # Corpus stats
    corpus_size: int
    num_buckets: int

    # Bucket statistics
    mean_bucket_size: float
    median_bucket_size: int
    max_bucket_size: int

    # Density metrics
    mean_centroid_similarity: float
    mean_isolation_score: float
    mean_stability_score: float

    # PCA stats
    pca_variance_explained: float

    # Category breakdown (if provided)
    category_counts: List[Tuple[str, int]]

    def __str__(self):
        lines = [
            "",
            "=" * 70,
            "DENSITY CLASSIFICATION REPORT",
            "=" * 70,
            "",
            "CORPUS OVERVIEW",
            "-" * 40,
            f"  Corpus size:              {self.corpus_size:,}",
            f"  Buckets:                  {self.num_buckets:,}",
            "",
            "BUCKET STATISTICS",
            "-" * 40,
            f"  Mean bucket size:         {self.mean_bucket_size:.1f}",
            f"  Median bucket size:       {self.median_bucket_size:,}",
            f"  Max bucket size:          {self.max_bucket_size:,}",
            "",
            "DENSITY METRICS",
            "-" * 40,
            f"  Mean centroid similarity: {self.mean_centroid_similarity:.4f}",
            f"  Mean isolation score:     {self.mean_isolation_score:.4f}",
            f"  Mean stability score:     {self.mean_stability_score:.4f}",
            f"  PCA variance explained:   {self.pca_variance_explained:.1%}",
        ]

        if self.category_counts:
            lines.extend([
                "",
                "TOP CATEGORIES",
                "-" * 40,
            ])
            for cat, count in self.category_counts[:10]:
                pct = count / self.corpus_size * 100
                lines.append(f"  {cat[:30]:<32} {count:>5} ({pct:>5.1f}%)")

        lines.append("=" * 70)
        return "\n".join(lines)


class DensityClassifier:
    """
    Density Classifier using PCA-based LSH.

    Returns raw density metrics per item - classification is up to you.

    Example:
        >>> classifier = DensityClassifier(embedding_dim=384)
        >>> classifier.fit(embeddings, categories=categories)
        >>> print(classifier.report())
        >>>
        >>> # Get raw metrics
        >>> labels = classifier.get_labels()
        >>> sparse = labels.filter(pl.col('bucket_size') < 10)
        >>> isolated = labels.filter(pl.col('isolation_score') > 0.5)
    """

    def __init__(
        self,
        embedding_dim: int = 384,
        num_bits: int = 14,
        seed: int = 31,
        isolation_k: int = 10,
        isolation_sample_size: int = 1000,
        num_stability_seeds: int = 3,
    ):
        """
        Initialize density classifier.

        Args:
            embedding_dim: Dimensionality of embeddings
            num_bits: Bits for PCA LSH (default: 14)
            seed: Random seed
            isolation_k: Number of top neighbors for isolation score
            isolation_sample_size: Sample size for median similarity computation
            num_stability_seeds: Number of seeds for stability scoring (default: 3)
        """
        self.embedding_dim = embedding_dim
        self.num_bits = num_bits
        self.seed = seed
        self.isolation_k = isolation_k
        self.isolation_sample_size = isolation_sample_size
        self.num_stability_seeds = num_stability_seeds

        # Populated during fit()
        self.embeddings: Optional[np.ndarray] = None
        self.categories: Optional[List[str]] = None
        self.texts: Optional[List[str]] = None

        # Per-record metrics
        self._bucket_ids: Optional[np.ndarray] = None
        self._bucket_sizes: Optional[np.ndarray] = None
        self._centroid_similarities: Optional[np.ndarray] = None
        self._isolation_scores: Optional[np.ndarray] = None
        self._stability_scores: Optional[np.ndarray] = None

        # Stats
        self._report: Optional[DensityReport] = None
        self._pca_variance: float = 0.0
        self._fitted = False

        # TF-IDF components (for from_texts)
        self._vectorizer = None
        self._svd = None

        # Polars integration
        self._source_df: Optional['pl.DataFrame'] = None
        self._embedding_col: Optional[str] = None

    @classmethod
    def from_polars(
        cls,
        df: 'pl.DataFrame',
        embedding_col: str,
        category_col: Optional[str] = None,
        text_col: Optional[str] = None,
        **kwargs
    ) -> 'DensityClassifier':
        """
        Create classifier from a Polars DataFrame.

        Args:
            df: Polars DataFrame with embeddings
            embedding_col: Column name containing embedding vectors (list of floats)
            category_col: Optional column name for category labels
            text_col: Optional column name for text content (enables labeling)
            **kwargs: Additional args passed to __init__ (num_bits, seed, etc.)

        Returns:
            Fitted DensityClassifier instance

        Example:
            >>> df = pl.read_parquet("embeddings.parquet")
            >>> classifier = DensityClassifier.from_polars(
            ...     df,
            ...     embedding_col="embedding",
            ...     category_col="category"
            ... )
            >>> result = classifier.to_polars()
        """
        import polars as pl

        # Extract embeddings
        embeddings = np.array(df[embedding_col].to_list(), dtype=np.float32)

        # Extract optional columns
        categories = df[category_col].to_list() if category_col else None
        texts = df[text_col].to_list() if text_col else None

        # Create classifier
        classifier = cls(embedding_dim=embeddings.shape[1], **kwargs)

        # Store reference to source DataFrame
        classifier._source_df = df
        classifier._embedding_col = embedding_col

        # Fit
        classifier.fit(embeddings, categories=categories, texts=texts)

        return classifier

    def to_polars(self) -> 'pl.DataFrame':
        """
        Return source DataFrame with density metrics columns added.

        Returns DataFrame with original columns plus:
            - bucket_id: LSH bucket ID
            - bucket_size: Number of items in same bucket
            - centroid_similarity: Cosine similarity to bucket centroid (0-1)
            - isolation_score: How isolated the item is
            - stability_score: How stable bucket assignment is (0-1)

        Example:
            >>> classifier = DensityClassifier.from_polars(df, "embedding")
            >>> result = classifier.to_polars()
            >>> sparse = result.filter(pl.col("bucket_size") < 5)
        """
        import polars as pl

        if not self._fitted:
            raise ValueError("Must call fit() first")

        # Create labels DataFrame
        labels_df = pl.DataFrame({
            'bucket_id': self._bucket_ids.tolist(),
            'bucket_size': self._bucket_sizes.tolist(),
            'centroid_similarity': self._centroid_similarities.tolist(),
            'isolation_score': self._isolation_scores.tolist(),
            'stability_score': self._stability_scores.tolist(),
        })

        # If we have source DataFrame, join to it
        if self._source_df is not None:
            return pl.concat([self._source_df, labels_df], how="horizontal")
        else:
            # Add index column for manual joining
            n = len(self.embeddings)
            return labels_df.with_columns(pl.Series("index", list(range(n))))

    @classmethod
    def from_texts(
        cls,
        texts: List[str],
        categories: Optional[List[str]] = None,
        embedding_dim: int = 128,
        max_features: int = 10000,
        min_df: int = 2,
        max_df: float = 0.95,
        ngram_range: Tuple[int, int] = (1, 2),
        num_bits: int = 12,
        verbose: bool = True,
        **kwargs
    ) -> 'DensityClassifier':
        """
        Create classifier from raw texts using TF-IDF + SVD embeddings.

        No external embedding model required. Uses sklearn's TfidfVectorizer
        and TruncatedSVD to create dense embeddings from text.

        Args:
            texts: List of text documents
            categories: Optional category labels
            embedding_dim: SVD output dimensions (default 128)
            max_features: Max vocabulary size for TF-IDF
            min_df: Min document frequency for terms
            max_df: Max document frequency for terms
            ngram_range: N-gram range (default unigrams + bigrams)
            num_bits: Bits for PCA LSH
            verbose: Print progress
            **kwargs: Additional args passed to __init__

        Returns:
            Fitted DensityClassifier instance

        Example:
            >>> texts = ["doc about machine learning", "another ml paper", ...]
            >>> classifier = DensityClassifier.from_texts(texts, categories=cats)
            >>> print(classifier.report())
        """
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.decomposition import TruncatedSVD

        if verbose:
            print(f"Building TF-IDF matrix ({len(texts):,} documents)...")

        # Build TF-IDF matrix
        vectorizer = TfidfVectorizer(
            max_features=max_features,
            min_df=min_df,
            max_df=max_df,
            ngram_range=ngram_range,
            stop_words='english'
        )
        tfidf_matrix = vectorizer.fit_transform(texts)

        if verbose:
            print(f"  Vocabulary size: {len(vectorizer.vocabulary_):,}")
            print(f"  Matrix shape: {tfidf_matrix.shape}")

        # Reduce to dense embeddings with SVD
        n_components = min(embedding_dim, tfidf_matrix.shape[1] - 1, len(texts) - 1)

        if verbose:
            print(f"Applying SVD ({n_components} components)...")

        svd = TruncatedSVD(n_components=n_components, random_state=kwargs.get('seed', 31))
        embeddings = svd.fit_transform(tfidf_matrix).astype(np.float32)

        if verbose:
            print(f"  Variance explained: {svd.explained_variance_ratio_.sum():.1%}")

        # Create and fit classifier
        classifier = cls(
            embedding_dim=n_components,
            num_bits=num_bits,
            **kwargs
        )

        # Store vectorizer and SVD for potential later use
        classifier._vectorizer = vectorizer
        classifier._svd = svd

        classifier.fit(embeddings, categories=categories, texts=texts, verbose=verbose)

        return classifier

    def fit(
        self,
        embeddings: np.ndarray,
        categories: Optional[List[str]] = None,
        texts: Optional[List[str]] = None,
        normalize: bool = True,
        verbose: bool = True
    ) -> 'DensityClassifier':
        """
        Fit the density classifier.

        Args:
            embeddings: (n, d) array of embedding vectors
            categories: Optional list of category labels
            texts: Optional list of text content
            normalize: Whether to L2-normalize embeddings
            verbose: Print progress

        Returns:
            self (for chaining)
        """
        embeddings = np.array(embeddings, dtype=np.float32)

        if normalize:
            norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
            norms = np.where(norms > 0, norms, 1)
            embeddings = embeddings / norms

        self.embeddings = embeddings
        self.categories = categories or ["unknown"] * len(embeddings)
        self.texts = texts

        n = len(embeddings)
        d = embeddings.shape[1]

        if verbose:
            print(f"Corpus size: {n:,}, dim: {d}")

        # Stage 1: Random hash → Centroids → PCA on centroids → Re-hash
        if verbose:
            print(f"\nPCA-based LSH ({self.num_bits} bits)...")

        # Step 1a: Random hash
        rng = np.random.default_rng(self.seed)
        random_hp = rng.standard_normal((self.num_bits, d)).astype(np.float32)
        random_hp = random_hp / np.linalg.norm(random_hp, axis=1, keepdims=True)

        signs_random = (embeddings @ random_hp.T) >= 0
        powers = 2 ** np.arange(self.num_bits)
        hashes_random = (signs_random @ powers).astype(np.uint64)

        # Step 1b: Compute bucket centroids
        random_bucket_to_indices = defaultdict(list)
        for idx, h in enumerate(hashes_random):
            random_bucket_to_indices[int(h)].append(idx)

        # Build centroid matrix (only for buckets with enough items)
        centroids = []
        for bid, indices in random_bucket_to_indices.items():
            if len(indices) >= 2:
                centroid = embeddings[indices].mean(axis=0)
                norm = np.linalg.norm(centroid)
                if norm > 0:
                    centroids.append(centroid / norm)

        centroids = np.array(centroids, dtype=np.float32)

        if verbose:
            print(f"  Random hash: {len(random_bucket_to_indices):,} buckets")
            print(f"  Centroids for PCA: {len(centroids):,}")

        # Step 1c: PCA on centroids
        n_components = min(self.num_bits, len(centroids) - 1)
        pca = PCA(n_components=n_components)
        pca.fit(centroids)
        hp = pca.components_.astype(np.float32)
        self._pca_variance = float(pca.explained_variance_ratio_.sum())

        if verbose:
            print(f"  Centroid PCA variance: {self._pca_variance:.1%}")

        # Step 1d: Re-hash with PCA hyperplanes
        signs = (embeddings @ hp.T) >= 0
        hashes = (signs @ powers[:len(hp)]).astype(np.uint64)

        # Store bucket IDs
        self._bucket_ids = hashes.copy()

        # Build bucket mapping
        bucket_to_indices = defaultdict(list)
        for idx, h in enumerate(hashes):
            bucket_to_indices[int(h)].append(idx)

        num_buckets = len(bucket_to_indices)

        # Compute density metrics
        self._bucket_sizes = np.zeros(n, dtype=np.int32)
        self._centroid_similarities = np.zeros(n, dtype=np.float32)

        for bid, indices in bucket_to_indices.items():
            bucket_size = len(indices)

            # Store bucket size
            for idx in indices:
                self._bucket_sizes[idx] = bucket_size

            # Compute centroid similarity
            if bucket_size >= 2:
                bucket_embs = embeddings[indices]
                centroid = bucket_embs.mean(axis=0)
                norm = np.linalg.norm(centroid)
                if norm > 0:
                    centroid = centroid / norm
                    sims = bucket_embs @ centroid
                    for local_idx, idx in enumerate(indices):
                        self._centroid_similarities[idx] = float(sims[local_idx])
            elif bucket_size == 1:
                self._centroid_similarities[indices[0]] = 1.0

        # Compute isolation scores
        if verbose:
            print(f"  Computing isolation scores...")
        self._compute_isolation_scores()

        # Compute stability scores
        if verbose:
            print(f"  Computing stability scores ({self.num_stability_seeds} seeds)...")
        self._compute_stability_scores(hp)

        # Build report
        self._build_report(num_buckets)

        if verbose:
            print(f"  Buckets: {num_buckets:,}")
            print(f"  Mean bucket size: {self._report.mean_bucket_size:.1f}")
            print(f"  Mean isolation score: {self._report.mean_isolation_score:.4f}")

        self._fitted = True
        return self

    def _compute_isolation_scores(self):
        """Compute isolation score for each item: top_k_mean - median."""
        n = len(self.embeddings)
        self._isolation_scores = np.zeros(n, dtype=np.float32)

        # Sample for median computation
        rng = np.random.default_rng(self.seed + 12345)
        sample_size = min(self.isolation_sample_size, n)
        sample_indices = rng.choice(n, sample_size, replace=False)

        k = self.isolation_k

        for i in range(n):
            # Compute similarities to sample
            sims = self.embeddings[i] @ self.embeddings[sample_indices].T

            # Exclude self if in sample
            if i in sample_indices:
                self_pos = np.where(sample_indices == i)[0]
                if len(self_pos) > 0:
                    sims[self_pos[0]] = -2.0

            # Sort descending
            sorted_sims = np.sort(sims)[::-1]

            # Top-k mean (excluding self which would be 1.0)
            top_k = sorted_sims[:k]
            top_k_mean = top_k.mean()

            # Median
            median = np.median(sorted_sims)

            self._isolation_scores[i] = top_k_mean - median

    def _compute_stability_scores(self, hp: np.ndarray):
        """Compute stability score: how consistently items stay in same bucket across seeds."""
        n = len(self.embeddings)
        num_seeds = self.num_stability_seeds

        if num_seeds < 2:
            # No stability computation possible with fewer than 2 seeds
            self._stability_scores = np.ones(n, dtype=np.float32)
            return

        powers = 2 ** np.arange(len(hp))

        # Compute bucket assignments for each seed
        # Start offset at 1 to avoid correlation with base hyperplanes
        all_bucket_ids = []
        for seed_idx in range(num_seeds):
            seed_offset = (seed_idx + 1) * 1000
            rng = np.random.default_rng(self.seed + seed_offset)

            # Add small random perturbation to hyperplanes
            perturbation = rng.standard_normal(hp.shape).astype(np.float32) * 0.01
            perturbed_hp = hp + perturbation
            perturbed_hp = perturbed_hp / np.linalg.norm(perturbed_hp, axis=1, keepdims=True)

            signs = (self.embeddings @ perturbed_hp.T) >= 0
            hashes = (signs @ powers).astype(np.uint64)
            all_bucket_ids.append(hashes)

        # Compute stability score per item
        self._stability_scores = np.zeros(n, dtype=np.float32)
        for i in range(n):
            bucket_set = set(all_bucket_ids[s][i] for s in range(num_seeds))
            unique_buckets = len(bucket_set)
            # stability = 1 - (unique - 1) / (num_seeds - 1)
            # 1.0 = same bucket in all seeds, 0.0 = different bucket each seed
            self._stability_scores[i] = 1.0 - (unique_buckets - 1) / (num_seeds - 1)

    def _build_report(self, num_buckets: int):
        """Build the density report."""
        # Bucket statistics
        unique_bucket_sizes = {}
        for bid, size in zip(self._bucket_ids, self._bucket_sizes):
            unique_bucket_sizes[bid] = size

        sizes = list(unique_bucket_sizes.values())
        sizes.sort()

        mean_bucket = np.mean(sizes) if sizes else 0.0
        median_bucket = sizes[len(sizes) // 2] if sizes else 0
        max_bucket = sizes[-1] if sizes else 0

        # Category breakdown
        cat_counts = Counter(self.categories).most_common()

        self._report = DensityReport(
            corpus_size=len(self.embeddings),
            num_buckets=num_buckets,
            mean_bucket_size=mean_bucket,
            median_bucket_size=median_bucket,
            max_bucket_size=max_bucket,
            mean_centroid_similarity=float(self._centroid_similarities.mean()),
            mean_isolation_score=float(self._isolation_scores.mean()),
            mean_stability_score=float(self._stability_scores.mean()),
            pca_variance_explained=self._pca_variance,
            category_counts=cat_counts,
        )

    def report(self) -> DensityReport:
        """Get the density classification report."""
        if not self._fitted:
            raise ValueError("Must call fit() first")
        return self._report

    def get_bucket_ids(self) -> np.ndarray:
        """Get bucket IDs for all items."""
        if not self._fitted:
            raise ValueError("Must call fit() first")
        return self._bucket_ids.copy()

    def get_bucket_sizes(self) -> np.ndarray:
        """Get bucket sizes for all items."""
        if not self._fitted:
            raise ValueError("Must call fit() first")
        return self._bucket_sizes.copy()

    def get_centroid_similarities(self) -> np.ndarray:
        """Get centroid similarities for all items."""
        if not self._fitted:
            raise ValueError("Must call fit() first")
        return self._centroid_similarities.copy()

    def get_isolation_scores(self) -> np.ndarray:
        """Get isolation scores for all items."""
        if not self._fitted:
            raise ValueError("Must call fit() first")
        return self._isolation_scores.copy()

    def get_stability_scores(self) -> np.ndarray:
        """Get stability scores for all items (0-1, higher = more stable)."""
        if not self._fitted:
            raise ValueError("Must call fit() first")
        return self._stability_scores.copy()

    def get_labels(self) -> 'pl.DataFrame':
        """
        Get per-record labels as a Polars DataFrame.

        Returns DataFrame with columns:
            - index: Record index (0-based)
            - bucket_id: Primary LSH bucket ID
            - bucket_size: Number of items in same bucket
            - centroid_similarity: Cosine similarity to bucket centroid
            - isolation_score: How isolated the item is
            - stability_score: How stable bucket assignment is (0-1)
            - category: Category label if provided during fit

        Example:
            >>> classifier.fit(embeddings, categories=categories)
            >>> labels = classifier.get_labels()
            >>> sparse = labels.filter(pl.col('bucket_size') < 10)
        """
        import polars as pl

        if not self._fitted:
            raise ValueError("Must call fit() first")

        n = len(self.embeddings)

        return pl.DataFrame({
            'index': list(range(n)),
            'bucket_id': self._bucket_ids.tolist(),
            'bucket_size': self._bucket_sizes.tolist(),
            'centroid_similarity': self._centroid_similarities.tolist(),
            'isolation_score': self._isolation_scores.tolist(),
            'stability_score': self._stability_scores.tolist(),
            'category': self.categories,
        })

    def label_buckets(
        self,
        base_url: str = "http://localhost:11434/v1",
        model: str = "qwen2.5:7b",
        samples_per_bucket: int = 5,
        max_text_len: int = 200,
        min_bucket_size: int = 5,
        verbose: bool = True
    ) -> Dict[int, Dict]:
        """
        Generate descriptive labels for buckets using a local LLM.

        Uses OpenAI-compatible API (works with Ollama or MLX server).

        Args:
            base_url: API endpoint (Ollama: localhost:11434, MLX: localhost:8080)
            model: Model name
            samples_per_bucket: Number of representative texts to send
            max_text_len: Max length of each sample text
            min_bucket_size: Only label buckets with at least this many items
            verbose: Print progress

        Returns:
            Dict mapping bucket_id -> {'label': str, 'size': int, 'samples': List[str]}

        Example:
            >>> labels = classifier.label_buckets(
            ...     base_url="http://localhost:11434/v1",
            ...     model="qwen2.5:7b"
            ... )
            >>> print(labels[1234]['label'])
            'Reinforcement Learning'
        """
        from openai import OpenAI

        if not self._fitted:
            raise ValueError("Must call fit() first")
        if self.texts is None:
            raise ValueError("Texts required for labeling. Pass texts to fit().")

        client = OpenAI(base_url=base_url, api_key="not-needed")

        def get_label(samples: List[str]) -> str:
            """Get label from LLM for a set of samples."""
            samples_text = "\n".join(f"- {s[:max_text_len]}" for s in samples)
            prompt = f"""These are sample texts from a document cluster:
{samples_text}

What 2-5 word label describes this cluster's shared topic or theme?
Be specific and descriptive. Just output the label, nothing else.
Label:"""

            try:
                response = client.chat.completions.create(
                    model=model,
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=30,
                    temperature=0.3
                )
                return response.choices[0].message.content.strip()
            except Exception as e:
                return f"[Error: {e}]"

        def sample_from_indices(indices: List[int]) -> List[str]:
            """Get representative samples closest to centroid."""
            if len(indices) <= samples_per_bucket:
                return [self.texts[i] for i in indices]

            bucket_embs = self.embeddings[indices]
            centroid = bucket_embs.mean(axis=0)
            centroid = centroid / np.linalg.norm(centroid)
            sims = bucket_embs @ centroid
            top_local = np.argsort(sims)[-samples_per_bucket:][::-1]
            return [self.texts[indices[i]] for i in top_local]

        # Build bucket -> indices mapping
        bucket_to_indices = defaultdict(list)
        for idx, bucket_id in enumerate(self._bucket_ids):
            bucket_to_indices[int(bucket_id)].append(idx)

        # Label buckets
        buckets_to_label = [
            (bid, indices) for bid, indices in bucket_to_indices.items()
            if len(indices) >= min_bucket_size
        ]

        if verbose:
            print(f"Labeling {len(buckets_to_label)} buckets...")

        results = {}
        for i, (bucket_id, indices) in enumerate(buckets_to_label):
            samples = sample_from_indices(indices)
            label = get_label(samples)
            results[bucket_id] = {
                'label': label,
                'size': len(indices),
                'samples': samples
            }
            if verbose and (i + 1) % 10 == 0:
                print(f"  {i + 1}/{len(buckets_to_label)} buckets labeled")

        if verbose:
            print("Done.")

        return results

    def label_buckets_keywords(
        self,
        top_k: int = 3,
        min_bucket_size: int = 5,
        stopwords: Optional[set] = None,
        min_word_len: int = 3,
        use_tfidf: bool = True
    ) -> Dict[int, Dict]:
        """
        Generate labels for buckets using keyword extraction (no LLM required).

        Extracts top keywords from bucket texts using TF-IDF or frequency analysis.

        Args:
            top_k: Number of top keywords to include in label
            min_bucket_size: Only label buckets with at least this many items
            stopwords: Set of words to exclude (uses default if None)
            min_word_len: Minimum word length to consider
            use_tfidf: Use TF-IDF weighting (vs simple frequency)

        Returns:
            Dict mapping bucket_id -> {'label': str, 'keywords': List[Tuple[str, float]], 'size': int}

        Example:
            >>> labels = classifier.label_buckets_keywords()
            >>> print(labels[1234]['label'])
            'neural network training'
        """
        import re
        import math

        if not self._fitted:
            raise ValueError("Must call fit() first")
        if self.texts is None:
            raise ValueError("Texts required for labeling. Pass texts to fit().")

        # Default stopwords
        default_stopwords = {
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
            'of', 'with', 'by', 'from', 'as', 'is', 'was', 'are', 'were', 'been',
            'be', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would',
            'could', 'should', 'may', 'might', 'must', 'shall', 'can', 'need',
            'this', 'that', 'these', 'those', 'it', 'its', 'we', 'our', 'they',
            'their', 'which', 'who', 'whom', 'what', 'where', 'when', 'why', 'how',
            'all', 'each', 'every', 'both', 'few', 'more', 'most', 'other', 'some',
            'such', 'no', 'not', 'only', 'same', 'so', 'than', 'too', 'very',
            'just', 'also', 'now', 'here', 'there', 'then', 'once', 'if', 'any',
            'about', 'into', 'through', 'during', 'before', 'after', 'above',
            'below', 'between', 'under', 'over', 'out', 'up', 'down', 'off',
            'paper', 'study', 'research', 'results', 'method', 'methods', 'approach',
            'proposed', 'propose', 'show', 'shows', 'shown', 'using', 'used', 'use',
            'based', 'problem', 'problems', 'work', 'new', 'novel', 'present',
            'presented', 'demonstrate', 'demonstrates', 'experimental', 'experiments',
            'however', 'therefore', 'thus', 'hence', 'moreover', 'furthermore',
            'first', 'second', 'third', 'one', 'two', 'three', 'many', 'several',
            'various', 'different', 'important', 'significant', 'provide', 'provides',
            'consider', 'considers', 'introduce', 'introduces', 'existing', 'recent',
            'previous', 'current', 'given', 'well', 'known', 'general', 'particular',
            'specific', 'case', 'cases', 'example', 'examples', 'order', 'number',
            'large', 'small', 'high', 'low', 'best', 'better', 'good', 'simple',
            'following', 'related', 'similar', 'compared', 'performance', 'evaluate',
            'evaluated', 'analysis', 'data', 'set', 'sets', 'model', 'models'
        }
        stops = stopwords if stopwords is not None else default_stopwords

        def tokenize(text: str) -> List[str]:
            text = text.lower()
            words = re.findall(r'\b[a-z]+\b', text)
            return [w for w in words if len(w) >= min_word_len and w not in stops]

        def get_keywords(indices: List[int], corpus_freqs: Counter) -> List[Tuple[str, float]]:
            bucket_words = []
            for idx in indices:
                bucket_words.extend(tokenize(self.texts[idx]))

            if not bucket_words:
                return []

            word_counts = Counter(bucket_words)

            if use_tfidf and corpus_freqs:
                n_docs = len(self.texts)
                scores = {}
                for word, count in word_counts.items():
                    tf = count / len(bucket_words)
                    df = corpus_freqs.get(word, 1)
                    idf = math.log(n_docs / df)
                    scores[word] = tf * idf
                return sorted(scores.items(), key=lambda x: -x[1])[:top_k]
            else:
                return word_counts.most_common(top_k)

        # Build corpus document frequencies
        corpus_freqs = Counter()
        if use_tfidf:
            for text in self.texts:
                unique_words = set(tokenize(text))
                corpus_freqs.update(unique_words)

        # Build bucket -> indices mapping
        bucket_to_indices = defaultdict(list)
        for idx, bucket_id in enumerate(self._bucket_ids):
            bucket_to_indices[int(bucket_id)].append(idx)

        results = {}
        for bucket_id, indices in bucket_to_indices.items():
            if len(indices) < min_bucket_size:
                continue

            keywords = get_keywords(indices, corpus_freqs)
            label = ' '.join(kw for kw, _ in keywords) if keywords else '[no keywords]'

            results[bucket_id] = {
                'label': label,
                'keywords': keywords,
                'size': len(indices)
            }

        return results


def demo():
    """Demo with ArXiv data."""
    import polars as pl

    print("Loading ArXiv data...")
    arxiv_path = '/Users/jdonaldson/Projects/semantic-proprioception/demo/arxiv_demo_data/MiniLM-L6_embeddings.parquet'

    try:
        df = pl.read_parquet(arxiv_path)
    except FileNotFoundError:
        print(f"Data not found at {arxiv_path}")
        return

    embeddings = np.array(df['embedding'].to_list(), dtype=np.float32)
    texts = df['text'].to_list()
    categories = df['category'].to_list()

    print(f"Loaded {len(embeddings):,} embeddings")

    # Run classifier
    classifier = DensityClassifier(embedding_dim=embeddings.shape[1])
    classifier.fit(embeddings, categories=categories, texts=texts)

    # Print report
    print(classifier.report())

    # Show labels DataFrame
    print("\n" + "=" * 70)
    print("RECORD LABELS")
    print("=" * 70)
    labels = classifier.get_labels()
    print(f"\nDataFrame shape: {labels.shape}")
    print(f"\nBucket size distribution:")
    print(labels.group_by(
        pl.col('bucket_size').cut([1, 5, 10, 50, 100, 500])
    ).len().sort('bucket_size'))

    print(f"\nMost isolated items:")
    print(labels.sort('isolation_score', descending=True).head(10))


if __name__ == "__main__":
    demo()
