"""
Density Classifier: Discover structure in embedding spaces

Bridge:   Transitional items connecting different semantic clusters.
          Found via recovery PCA on sparse bucket items.

Orphans:  Genuinely unique items that don't cluster anywhere, even after
          recovery attempts. They have no semantic neighbors.

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
from typing import List, Dict, Optional, Tuple, TYPE_CHECKING, Union
from enum import Enum

if TYPE_CHECKING:
    import polars as pl
from dataclasses import dataclass, field
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
        from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
        from sklearn.decomposition import TruncatedSVD
        from scipy import sparse
        import math

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

        # Apply BM25 saturation: tf_sat = (tf * (k1 + 1)) / (tf + k1 * (1 - b + b * dl/avgdl))
        # Convert to lil_matrix for efficient element-wise modification
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
        idf = np.log((n_docs + 1) / (doc_freq + 1)) + 1  # smoothed IDF

        # Multiply each column by its IDF weight
        bm25_matrix = tf_saturated.multiply(idf)

        if verbose:
            print(f"  BM25 matrix shape: {bm25_matrix.shape}")

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
        >>> # Start server: ollama serve
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
    """A recovered cluster of diaspora items."""
    cluster_id: int
    size: int
    coherence: float
    global_indices: List[int]
    source_buckets: Dict[int, int]  # original_bucket_id -> count
    dominant_category: str
    category_counts: List[Tuple[str, int]]


@dataclass
class DensityReport:
    """Complete report on outlier classification."""
    # Corpus stats
    corpus_size: int
    num_buckets: int

    # Stage 1: Initial bucketing
    dense_items: int
    sparse_bucket_outliers: int  # Items in small buckets
    intra_bucket_outliers: int   # Items far from their bucket centroid

    # Classification
    total_outlier_candidates: int
    diaspora_count: int
    orphan_count: int
    diaspora_pct: float
    orphan_pct: float

    # Recovery stats
    recovery_clusters: int
    recovery_variance_explained: float

    # Orphan analysis
    orphan_avg_nn_similarity: float
    corpus_avg_nn_similarity: float
    orphan_isolation_score: float  # std below corpus average

    # Category breakdown
    diaspora_categories: List[Tuple[str, int]]
    orphan_categories: List[Tuple[str, int]]

    def __str__(self):
        lines = [
            "",
            "=" * 70,
            "OUTLIER CLASSIFICATION REPORT",
            "=" * 70,
            "",
            "CORPUS OVERVIEW",
            "-" * 40,
            f"  Corpus size:              {self.corpus_size:,}",
            f"  Buckets:                  {self.num_buckets:,}",
            f"  Dense items:              {self.dense_items:,} ({self.dense_items/self.corpus_size:.1%})",
            "",
            "OUTLIER IDENTIFICATION",
            "-" * 40,
            f"  Sparse bucket outliers:   {self.sparse_bucket_outliers:,}",
            f"  Intra-bucket outliers:    {self.intra_bucket_outliers:,}",
            f"  Total candidates:         {self.total_outlier_candidates:,} ({self.total_outlier_candidates/self.corpus_size:.1%})",
            "",
            "CLASSIFICATION RESULTS",
            "-" * 40,
            f"  Bridge (recovered):     {self.diaspora_count:,} ({self.diaspora_pct:.1%})",
            f"  Orphans (truly unique):   {self.orphan_count:,} ({self.orphan_pct:.1%})",
            "",
            "RECOVERY STATS",
            "-" * 40,
            f"  Recovery clusters:        {self.recovery_clusters:,}",
            f"  PCA variance explained:   {self.recovery_variance_explained:.1%}",
            "",
            "ORPHAN ISOLATION",
            "-" * 40,
            f"  Orphan avg NN sim:        {self.orphan_avg_nn_similarity:.4f}",
            f"  Corpus avg NN sim:        {self.corpus_avg_nn_similarity:.4f}",
            f"  Isolation score:          {self.orphan_isolation_score:.1f} std below average",
            "",
            "DIASPORA BY CATEGORY",
            "-" * 40,
        ]

        for cat, count in self.diaspora_categories[:7]:
            pct = count / self.diaspora_count * 100 if self.diaspora_count > 0 else 0
            lines.append(f"  {cat[:30]:<32} {count:>5} ({pct:>5.1f}%)")

        lines.extend([
            "",
            "ORPHAN BY CATEGORY",
            "-" * 40,
        ])

        for cat, count in self.orphan_categories[:7]:
            pct = count / self.orphan_count * 100 if self.orphan_count > 0 else 0
            lines.append(f"  {cat[:30]:<32} {count:>5} ({pct:>5.1f}%)")

        lines.append("=" * 70)
        return "\n".join(lines)


class DensityClassifier:
    """
    Classify outliers as Bridge or Orphans.

    Bridge: Misplaced items that find community with outlier-specific PCA
    Orphans: Genuinely unique items with no semantic neighbors

    Example:
        >>> classifier = DensityClassifier(embedding_dim=384)
        >>> classifier.fit(embeddings, categories=categories)
        >>> print(classifier.report())
        >>>
        >>> # Get specific groups
        >>> diaspora = classifier.get_diaspora()
        >>> orphans = classifier.get_orphans()
        >>>
        >>> # Analyze diaspora clusters
        >>> clusters = classifier.get_diaspora_clusters(min_size=5)
    """

    def __init__(
        self,
        embedding_dim: int = 384,
        initial_bits: int = 14,
        recovery_bits: int = 8,
        dense_threshold: int = 10,
        intra_outlier_std: float = 2.0,
        recovery_cluster_min: int = 3,
        seed: int = 31
    ):
        """
        Initialize outlier classifier.

        Args:
            embedding_dim: Dimensionality of embeddings
            initial_bits: Bits for initial PCA LSH
            recovery_bits: Bits for recovery PCA (fewer = coarser)
            dense_threshold: Min bucket size to be considered dense
            intra_outlier_std: Std threshold for intra-bucket outliers
            recovery_cluster_min: Min cluster size to be considered "recovered"
            seed: Random seed
        """
        self.embedding_dim = embedding_dim
        self.initial_bits = initial_bits
        self.recovery_bits = recovery_bits
        self.dense_threshold = dense_threshold
        self.intra_outlier_std = intra_outlier_std
        self.recovery_cluster_min = recovery_cluster_min
        self.seed = seed

        # Populated during fit()
        self.embeddings: Optional[np.ndarray] = None
        self.categories: Optional[List[str]] = None
        self.texts: Optional[List[str]] = None

        # Classification results
        self._diaspora_indices: List[int] = []
        self._orphan_indices: List[int] = []
        self._diaspora_clusters: List[BridgeCluster] = []
        self._outlier_source_buckets: Dict[int, int] = {}  # global_idx -> original bucket

        # Per-record labels (populated during fit)
        self._bucket_ids: Optional[np.ndarray] = None  # Primary bucket for each record
        self._statuses: Optional[List[str]] = None     # 'dense', 'diaspora', 'orphan'
        self._recovery_bucket_ids: Optional[Dict[int, int]] = None  # For diaspora: recovery bucket

        # Stats
        self._report: Optional[DensityReport] = None
        self._fitted = False

        # TF-IDF components (for from_texts)
        self._vectorizer = None
        self._svd = None

        # Polars integration
        self._source_df: Optional['pl.DataFrame'] = None
        self._embedding_col: Optional[str] = None

        # Density metrics (computed during fit)
        self._bucket_sizes: Optional[np.ndarray] = None
        self._centroid_similarities: Optional[np.ndarray] = None

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
            **kwargs: Additional args passed to __init__ (initial_bits, dense_threshold, etc.)

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
        Return source DataFrame with density classification columns added.

        Returns DataFrame with original columns plus:
            - bucket_id: LSH bucket ID
            - status: 'dense', 'bridge', or 'orphan'
            - bucket_size: Number of items in same bucket
            - centroid_similarity: Cosine similarity to bucket centroid (0-1)
            - recovery_bucket_id: For bridge items, their recovery bucket (null for others)

        Example:
            >>> classifier = DensityClassifier.from_polars(df, "embedding")
            >>> result = classifier.to_polars()
            >>> sparse = result.filter(pl.col("bucket_size") < 5)
        """
        import polars as pl

        if not self._fitted:
            raise ValueError("Must call fit() first")

        n = len(self.embeddings)

        # Build recovery bucket column (null for non-bridge)
        recovery_buckets = [
            self._recovery_bucket_ids.get(i) for i in range(n)
        ]

        # Create labels DataFrame
        labels_df = pl.DataFrame({
            'bucket_id': self._bucket_ids.tolist(),
            'status': [s.replace('diaspora', 'bridge') for s in self._statuses],
            'bucket_size': self._bucket_sizes.tolist(),
            'centroid_similarity': self._centroid_similarities.tolist(),
            'recovery_bucket_id': recovery_buckets,
        })

        # If we have source DataFrame, join to it
        if self._source_df is not None:
            return pl.concat([self._source_df, labels_df], how="horizontal")
        else:
            # Add index column for manual joining
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
        initial_bits: int = 12,
        recovery_bits: int = 8,
        dense_threshold: int = 10,
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
            initial_bits: Bits for initial LSH
            recovery_bits: Bits for recovery LSH
            dense_threshold: Min bucket size for dense
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
            initial_bits=initial_bits,
            recovery_bits=recovery_bits,
            dense_threshold=dense_threshold,
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
        Fit the outlier classifier.

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

        if verbose:
            print(f"Corpus size: {len(embeddings):,}")

        # Stage 1: Random hash → Centroids → PCA on centroids → Re-hash
        # This is O(n*d*b + B*d²) instead of O(n*d²) where B << n
        if verbose:
            print(f"\nStage 1: Centroid-based PCA LSH ({self.initial_bits} bits)...")

        # Step 1a: Random hash
        rng = np.random.default_rng(self.seed)
        random_hp = rng.standard_normal((self.initial_bits, embeddings.shape[1])).astype(np.float32)
        random_hp = random_hp / np.linalg.norm(random_hp, axis=1, keepdims=True)

        signs_random = (embeddings @ random_hp.T) >= 0
        powers = 2 ** np.arange(self.initial_bits)
        hashes_random = (signs_random @ powers).astype(np.uint64)

        # Step 1b: Compute bucket centroids
        random_bucket_to_indices = defaultdict(list)
        for idx, h in enumerate(hashes_random):
            random_bucket_to_indices[int(h)].append(idx)

        # Build centroid matrix (only for buckets with enough items)
        centroids = []
        for bid, indices in random_bucket_to_indices.items():
            if len(indices) >= 2:  # Need at least 2 items for meaningful centroid
                centroid = embeddings[indices].mean(axis=0)
                norm = np.linalg.norm(centroid)
                if norm > 0:
                    centroids.append(centroid / norm)

        centroids = np.array(centroids, dtype=np.float32)

        if verbose:
            print(f"  Random hash: {len(random_bucket_to_indices):,} buckets")
            print(f"  Centroids for PCA: {len(centroids):,}")

        # Step 1c: PCA on centroids
        n_components = min(self.initial_bits, len(centroids) - 1)
        pca1 = PCA(n_components=n_components)
        pca1.fit(centroids)
        hp1 = pca1.components_.astype(np.float32)

        if verbose:
            print(f"  Centroid PCA variance: {pca1.explained_variance_ratio_.sum():.1%}")

        # Step 1d: Re-hash with PCA hyperplanes
        signs = (embeddings @ hp1.T) >= 0
        hashes1 = (signs @ powers[:len(hp1)]).astype(np.uint64)
        counts1 = Counter(hashes1)

        # Store bucket IDs for all records
        self._bucket_ids = hashes1.copy()

        bucket_to_indices = defaultdict(list)
        for idx, h in enumerate(hashes1):
            bucket_to_indices[int(h)].append(idx)

        num_buckets = len(bucket_to_indices)

        # Compute density metrics: bucket_size and centroid_similarity
        self._bucket_sizes = np.zeros(len(embeddings), dtype=np.int32)
        self._centroid_similarities = np.zeros(len(embeddings), dtype=np.float32)

        for bid, indices in bucket_to_indices.items():
            bucket_size = len(indices)

            # Store bucket size for all items in this bucket
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
                # Single item is perfectly central to itself
                self._centroid_similarities[indices[0]] = 1.0

        # Identify outliers
        sparse_bucket_outliers = []
        intra_bucket_outliers = []
        self._outlier_source_buckets = {}

        for bid, indices in bucket_to_indices.items():
            bucket_size = len(indices)

            if bucket_size < self.dense_threshold:
                # Entire bucket is sparse
                for idx in indices:
                    sparse_bucket_outliers.append(idx)
                    self._outlier_source_buckets[idx] = bid
            elif bucket_size >= 3:
                # Check for intra-bucket outliers
                bucket_embs = embeddings[indices]
                centroid = bucket_embs.mean(axis=0)
                centroid = centroid / np.linalg.norm(centroid)
                sims = bucket_embs @ centroid

                if sims.std() > 0.01:
                    threshold = sims.mean() - self.intra_outlier_std * sims.std()
                    for local_idx, sim in enumerate(sims):
                        if sim < threshold:
                            global_idx = indices[local_idx]
                            intra_bucket_outliers.append(global_idx)
                            self._outlier_source_buckets[global_idx] = bid

        # Combine all outlier candidates
        all_outliers = list(set(sparse_bucket_outliers + intra_bucket_outliers))

        if verbose:
            print(f"  Sparse bucket outliers: {len(sparse_bucket_outliers):,}")
            print(f"  Intra-bucket outliers: {len(intra_bucket_outliers):,}")
            print(f"  Total candidates: {len(all_outliers):,}")

        # Stage 2: Recovery attempt
        if verbose:
            print(f"\nStage 2: Recovery PCA ({self.recovery_bits} bits)...")

        if len(all_outliers) < 10:
            if verbose:
                print("  Too few outliers for recovery analysis")
            self._diaspora_indices = []
            self._orphan_indices = all_outliers
            self._recovery_bucket_ids = {}
            # Build status labels
            self._statuses = ['dense'] * len(embeddings)
            for idx in all_outliers:
                self._statuses[idx] = 'orphan'
            self._fitted = True
            self._build_report(
                num_buckets, len(sparse_bucket_outliers),
                len(intra_bucket_outliers), 0, 0.0
            )
            return self

        outlier_embs = embeddings[all_outliers]

        pca2 = PCA(n_components=min(self.recovery_bits, len(all_outliers) - 1))
        pca2.fit(outlier_embs)
        hp2 = pca2.components_[:self.recovery_bits].astype(np.float32)
        recovery_variance = float(pca2.explained_variance_ratio_.sum())

        signs2 = (outlier_embs @ hp2.T) >= 0
        powers2 = 2 ** np.arange(hp2.shape[0])
        hashes2 = (signs2 @ powers2).astype(np.uint64)
        counts2 = Counter(hashes2)

        # Classify: Bridge vs Orphans
        recovery_bucket_to_local = defaultdict(list)
        for local_idx, h in enumerate(hashes2):
            recovery_bucket_to_local[int(h)].append(local_idx)

        diaspora_indices = []
        orphan_indices = []
        self._recovery_bucket_ids = {}

        for local_idx, h in enumerate(hashes2):
            global_idx = all_outliers[local_idx]
            if counts2[h] >= self.recovery_cluster_min:
                diaspora_indices.append(global_idx)
                self._recovery_bucket_ids[global_idx] = int(h)
            else:
                orphan_indices.append(global_idx)

        self._diaspora_indices = diaspora_indices
        self._orphan_indices = orphan_indices

        # Build per-record status labels
        self._statuses = ['dense'] * len(embeddings)
        for idx in diaspora_indices:
            self._statuses[idx] = 'diaspora'
        for idx in orphan_indices:
            self._statuses[idx] = 'orphan'

        if verbose:
            print(f"  Bridge (recovered): {len(diaspora_indices):,}")
            print(f"  Orphans (unique): {len(orphan_indices):,}")

        # Build diaspora clusters
        self._diaspora_clusters = []
        cluster_id = 0

        for recovery_bid, local_indices in recovery_bucket_to_local.items():
            if len(local_indices) < self.recovery_cluster_min:
                continue

            global_indices = [all_outliers[i] for i in local_indices]

            # Coherence
            cluster_embs = embeddings[global_indices]
            n = len(cluster_embs)
            sim_matrix = cluster_embs @ cluster_embs.T
            coherence = float((sim_matrix.sum() - n) / (n * (n - 1)))

            # Source buckets
            source_buckets = Counter(
                self._outlier_source_buckets[idx] for idx in global_indices
            )

            # Categories
            cluster_cats = [self.categories[i] for i in global_indices]
            cat_counts = Counter(cluster_cats).most_common()
            dominant_cat = cat_counts[0][0] if cat_counts else "unknown"

            self._diaspora_clusters.append(BridgeCluster(
                cluster_id=cluster_id,
                size=len(global_indices),
                coherence=coherence,
                global_indices=global_indices,
                source_buckets=dict(source_buckets),
                dominant_category=dominant_cat,
                category_counts=cat_counts
            ))
            cluster_id += 1

        self._diaspora_clusters.sort(key=lambda c: -c.size)

        # Build report
        recovery_clusters = sum(1 for c in counts2.values() if c >= self.recovery_cluster_min)
        self._build_report(
            num_buckets, len(sparse_bucket_outliers),
            len(intra_bucket_outliers), recovery_clusters, recovery_variance
        )

        self._fitted = True
        return self

    def _build_report(
        self,
        num_buckets: int,
        sparse_count: int,
        intra_count: int,
        recovery_clusters: int,
        recovery_variance: float
    ):
        """Build the outlier report."""
        total_candidates = len(self._diaspora_indices) + len(self._orphan_indices)

        # Orphan isolation analysis
        if len(self._orphan_indices) > 0 and len(self.embeddings) > 100:
            orphan_embs = self.embeddings[self._orphan_indices]
            all_sims = orphan_embs @ self.embeddings.T

            for i, idx in enumerate(self._orphan_indices):
                all_sims[i, idx] = -1  # Exclude self

            orphan_nn_sims = all_sims.max(axis=1)
            orphan_avg_nn = float(orphan_nn_sims.mean())

            # Corpus baseline
            rng = np.random.default_rng(42)
            sample_indices = rng.choice(len(self.embeddings), min(1000, len(self.embeddings)), replace=False)
            sample_embs = self.embeddings[sample_indices]
            sample_sims = sample_embs @ self.embeddings.T
            for i, idx in enumerate(sample_indices):
                sample_sims[i, idx] = -1
            corpus_nn_sims = sample_sims.max(axis=1)
            corpus_avg_nn = float(corpus_nn_sims.mean())
            corpus_std_nn = float(corpus_nn_sims.std())

            isolation_score = (corpus_avg_nn - orphan_avg_nn) / corpus_std_nn if corpus_std_nn > 0 else 0
        else:
            orphan_avg_nn = 0.0
            corpus_avg_nn = 0.0
            isolation_score = 0.0

        # Category breakdowns
        diaspora_cats = Counter(self.categories[i] for i in self._diaspora_indices)
        orphan_cats = Counter(self.categories[i] for i in self._orphan_indices)

        dense_items = len(self.embeddings) - total_candidates

        self._report = DensityReport(
            corpus_size=len(self.embeddings),
            num_buckets=num_buckets,
            dense_items=dense_items,
            sparse_bucket_outliers=sparse_count,
            intra_bucket_outliers=intra_count,
            total_outlier_candidates=total_candidates,
            diaspora_count=len(self._diaspora_indices),
            orphan_count=len(self._orphan_indices),
            diaspora_pct=len(self._diaspora_indices) / total_candidates if total_candidates > 0 else 0,
            orphan_pct=len(self._orphan_indices) / total_candidates if total_candidates > 0 else 0,
            recovery_clusters=recovery_clusters,
            recovery_variance_explained=recovery_variance,
            orphan_avg_nn_similarity=orphan_avg_nn,
            corpus_avg_nn_similarity=corpus_avg_nn,
            orphan_isolation_score=isolation_score,
            diaspora_categories=diaspora_cats.most_common(),
            orphan_categories=orphan_cats.most_common()
        )

    def report(self) -> DensityReport:
        """Get the outlier classification report."""
        if not self._fitted:
            raise ValueError("Must call fit() first")
        return self._report

    def get_diaspora(self) -> List[int]:
        """Get indices of diaspora items."""
        if not self._fitted:
            raise ValueError("Must call fit() first")
        return list(self._diaspora_indices)

    def get_orphans(self) -> List[int]:
        """Get indices of orphan items."""
        if not self._fitted:
            raise ValueError("Must call fit() first")
        return list(self._orphan_indices)

    def get_diaspora_clusters(self, min_size: int = 3) -> List[BridgeCluster]:
        """Get diaspora clusters above min_size."""
        if not self._fitted:
            raise ValueError("Must call fit() first")
        return [c for c in self._diaspora_clusters if c.size >= min_size]

    def get_labels(self) -> 'pl.DataFrame':
        """
        Get per-record labels as a Polars DataFrame.

        Returns DataFrame with columns:
            - index: Record index (0-based)
            - bucket_id: Primary LSH bucket ID
            - status: 'dense', 'bridge', or 'orphan'
            - bucket_size: Number of items in same bucket
            - centroid_similarity: Cosine similarity to bucket centroid (0-1)
            - recovery_bucket_id: For bridge items, their recovery bucket (null for others)
            - category: Category label if provided during fit

        Example:
            >>> classifier.fit(embeddings, categories=categories)
            >>> labels = classifier.get_labels()
            >>> orphans = labels.filter(pl.col('status') == 'orphan')
            >>> sparse = labels.filter(pl.col('bucket_size') < 5)
        """
        import polars as pl

        if not self._fitted:
            raise ValueError("Must call fit() first")

        n = len(self.embeddings)

        # Build recovery bucket column (null for non-bridge)
        recovery_buckets = [
            self._recovery_bucket_ids.get(i) for i in range(n)
        ]

        return pl.DataFrame({
            'index': list(range(n)),
            'bucket_id': self._bucket_ids.tolist(),
            'status': [s.replace('diaspora', 'bridge') for s in self._statuses],
            'bucket_size': self._bucket_sizes.tolist(),
            'centroid_similarity': self._centroid_similarities.tolist(),
            'recovery_bucket_id': recovery_buckets,
            'category': self.categories,
        })

    def label_buckets(
        self,
        base_url: str = "http://localhost:11434/v1",
        model: str = "qwen2.5:7b",
        samples_per_bucket: int = 5,
        max_text_len: int = 200,
        include_diaspora: bool = True,
        min_bucket_size: int = 5,
        verbose: bool = True
    ) -> Dict[str, Dict]:
        """
        Generate descriptive labels for buckets using a local LLM.

        Uses OpenAI-compatible API (works with Ollama or MLX server).

        Args:
            base_url: API endpoint (Ollama: localhost:11434, MLX: localhost:8080)
            model: Model name
            samples_per_bucket: Number of representative texts to send
            max_text_len: Max length of each sample text
            include_diaspora: Also label diaspora recovery clusters
            min_bucket_size: Only label buckets with at least this many items
            verbose: Print progress

        Returns:
            Dict with 'dense' and optionally 'diaspora' keys, each mapping
            bucket_id -> {'label': str, 'size': int, 'samples': List[str]}

        Example:
            >>> labels = classifier.label_buckets(
            ...     base_url="http://localhost:11434/v1",
            ...     model="qwen2.5:7b"
            ... )
            >>> print(labels['dense'][1234]['label'])
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

        results = {'dense': {}}

        # Build bucket -> indices mapping for dense buckets
        dense_buckets = defaultdict(list)
        for idx, (bucket_id, status) in enumerate(zip(self._bucket_ids, self._statuses)):
            if status == 'dense':
                dense_buckets[int(bucket_id)].append(idx)

        # Label dense buckets
        buckets_to_label = [
            (bid, indices) for bid, indices in dense_buckets.items()
            if len(indices) >= min_bucket_size
        ]

        if verbose:
            print(f"Labeling {len(buckets_to_label)} dense buckets...")

        for i, (bucket_id, indices) in enumerate(buckets_to_label):
            samples = sample_from_indices(indices)
            label = get_label(samples)
            results['dense'][bucket_id] = {
                'label': label,
                'size': len(indices),
                'samples': samples
            }
            if verbose and (i + 1) % 10 == 0:
                print(f"  {i + 1}/{len(buckets_to_label)} buckets labeled")

        # Label diaspora clusters
        if include_diaspora and self._diaspora_clusters:
            results['diaspora'] = {}
            diaspora_to_label = [
                c for c in self._diaspora_clusters
                if c.size >= min_bucket_size
            ]

            if verbose:
                print(f"Labeling {len(diaspora_to_label)} diaspora clusters...")

            for i, cluster in enumerate(diaspora_to_label):
                samples = sample_from_indices(cluster.global_indices)
                label = get_label(samples)
                results['diaspora'][cluster.cluster_id] = {
                    'label': label,
                    'size': cluster.size,
                    'coherence': cluster.coherence,
                    'samples': samples
                }
                if verbose and (i + 1) % 10 == 0:
                    print(f"  {i + 1}/{len(diaspora_to_label)} diaspora clusters labeled")

        if verbose:
            print("Done.")

        return results

    def label_buckets_keywords(
        self,
        top_k: int = 3,
        min_bucket_size: int = 5,
        include_diaspora: bool = True,
        stopwords: Optional[set] = None,
        min_word_len: int = 3,
        use_tfidf: bool = True
    ) -> Dict[str, Dict]:
        """
        Generate labels for buckets using keyword extraction (no LLM required).

        Extracts top keywords from bucket texts using TF-IDF or frequency analysis.

        Args:
            top_k: Number of top keywords to include in label
            min_bucket_size: Only label buckets with at least this many items
            include_diaspora: Also label diaspora recovery clusters
            stopwords: Set of words to exclude (uses default if None)
            min_word_len: Minimum word length to consider
            use_tfidf: Use TF-IDF weighting (vs simple frequency)

        Returns:
            Dict with 'dense' and optionally 'diaspora' keys, each mapping
            bucket_id -> {'label': str, 'keywords': List[Tuple[str, float]], 'size': int}

        Example:
            >>> labels = classifier.label_buckets_keywords()
            >>> print(labels['dense'][1234]['label'])
            'neural network training'
        """
        import re
        from collections import Counter
        import math

        if not self._fitted:
            raise ValueError("Must call fit() first")
        if self.texts is None:
            raise ValueError("Texts required for labeling. Pass texts to fit().")

        # Default stopwords (common English + academic)
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
            # Academic common words
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
            """Extract words from text."""
            text = text.lower()
            words = re.findall(r'\b[a-z]+\b', text)
            return [w for w in words if len(w) >= min_word_len and w not in stops]

        def get_keywords(indices: List[int], corpus_freqs: Counter) -> List[Tuple[str, float]]:
            """Extract top keywords from bucket texts."""
            # Count words in this bucket
            bucket_words = []
            for idx in indices:
                bucket_words.extend(tokenize(self.texts[idx]))

            if not bucket_words:
                return []

            word_counts = Counter(bucket_words)

            if use_tfidf and corpus_freqs:
                # TF-IDF: upweight words rare in corpus but common in bucket
                n_docs = len(self.texts)
                scores = {}
                for word, count in word_counts.items():
                    tf = count / len(bucket_words)
                    df = corpus_freqs.get(word, 1)
                    idf = math.log(n_docs / df)
                    scores[word] = tf * idf
                return sorted(scores.items(), key=lambda x: -x[1])[:top_k]
            else:
                # Simple frequency
                return word_counts.most_common(top_k)

        # Build corpus document frequencies for TF-IDF
        corpus_freqs = Counter()
        if use_tfidf:
            for text in self.texts:
                unique_words = set(tokenize(text))
                corpus_freqs.update(unique_words)

        results = {'dense': {}}

        # Build bucket -> indices mapping for dense buckets
        dense_buckets = defaultdict(list)
        for idx, (bucket_id, status) in enumerate(zip(self._bucket_ids, self._statuses)):
            if status == 'dense':
                dense_buckets[int(bucket_id)].append(idx)

        # Label dense buckets
        for bucket_id, indices in dense_buckets.items():
            if len(indices) < min_bucket_size:
                continue

            keywords = get_keywords(indices, corpus_freqs)
            label = ' '.join(kw for kw, _ in keywords) if keywords else '[no keywords]'

            results['dense'][bucket_id] = {
                'label': label,
                'keywords': keywords,
                'size': len(indices)
            }

        # Label diaspora clusters
        if include_diaspora and self._diaspora_clusters:
            results['diaspora'] = {}

            for cluster in self._diaspora_clusters:
                if cluster.size < min_bucket_size:
                    continue

                keywords = get_keywords(cluster.global_indices, corpus_freqs)
                label = ' '.join(kw for kw, _ in keywords) if keywords else '[no keywords]'

                results['diaspora'][cluster.cluster_id] = {
                    'label': label,
                    'keywords': keywords,
                    'size': cluster.size,
                    'coherence': cluster.coherence
                }

        return results

    def print_diaspora_cluster(self, cluster: BridgeCluster, n_samples: int = 5):
        """Print details of a diaspora cluster."""
        print(f"\nBridge Cluster {cluster.cluster_id}")
        print(f"  Size: {cluster.size}")
        print(f"  Coherence: {cluster.coherence:.4f}")
        print(f"  From {len(cluster.source_buckets)} original buckets")
        print(f"  Categories: {cluster.category_counts[:3]}")

        if self.texts:
            print(f"  Samples:")
            centroid = self.embeddings[cluster.global_indices].mean(axis=0)
            centroid = centroid / np.linalg.norm(centroid)
            sims = self.embeddings[cluster.global_indices] @ centroid
            top_local = np.argsort(sims)[-n_samples:][::-1]

            for local_idx in top_local:
                global_idx = cluster.global_indices[local_idx]
                text = self.texts[global_idx][:70]
                print(f"    - {text}...")

    def print_orphans(self, n_samples: int = 10):
        """Print sample orphan items."""
        if not self._fitted:
            raise ValueError("Must call fit() first")

        print(f"\nSample Orphans ({len(self._orphan_indices)} total):")

        # Sort by isolation (lowest NN similarity first)
        if len(self._orphan_indices) > 0:
            orphan_embs = self.embeddings[self._orphan_indices]
            all_sims = orphan_embs @ self.embeddings.T
            for i, idx in enumerate(self._orphan_indices):
                all_sims[i, idx] = -1
            nn_sims = all_sims.max(axis=1)
            order = np.argsort(nn_sims)

            for i in order[:n_samples]:
                idx = self._orphan_indices[i]
                cat = self.categories[idx][:25]
                text = self.texts[idx][:50] if self.texts else ""
                print(f"  [NN={nn_sims[i]:.3f}] {cat}: {text}...")


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

    # Show diaspora clusters
    print("\n" + "=" * 70)
    print("TOP DIASPORA CLUSTERS")
    print("=" * 70)

    clusters = classifier.get_diaspora_clusters(min_size=10)
    for cluster in clusters[:3]:
        classifier.print_diaspora_cluster(cluster)

    # Show orphans
    print("\n" + "=" * 70)
    print("MOST ISOLATED ORPHANS")
    print("=" * 70)
    classifier.print_orphans(n_samples=10)

    # Show labels DataFrame
    print("\n" + "=" * 70)
    print("RECORD LABELS")
    print("=" * 70)
    labels = classifier.get_labels()
    print(f"\nDataFrame shape: {labels.shape}")
    print(f"\nStatus counts:")
    print(labels.group_by('status').len().sort('len', descending=True))
    print(f"\nSample diaspora records:")
    print(labels.filter(pl.col('status') == 'diaspora').head(5))
    print(f"\nAll orphan records:")
    print(labels.filter(pl.col('status') == 'orphan'))


if __name__ == "__main__":
    demo()
