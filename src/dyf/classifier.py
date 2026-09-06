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

import logging
from typing import TYPE_CHECKING

import numpy as np

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    import pandas as pd
    import polars as pl
from collections import Counter, defaultdict
from dataclasses import dataclass

from .configs import EmbedderConfig, LabelerConfig, list_configs  # noqa: F401

# =============================================================================
# Module-level constants and helpers
# =============================================================================

_DEFAULT_ACADEMIC_STOPWORDS: frozenset[str] = frozenset(
    {
        "the",
        "a",
        "an",
        "and",
        "or",
        "but",
        "in",
        "on",
        "at",
        "to",
        "for",
        "of",
        "with",
        "by",
        "from",
        "as",
        "is",
        "was",
        "are",
        "were",
        "been",
        "be",
        "have",
        "has",
        "had",
        "do",
        "does",
        "did",
        "will",
        "would",
        "could",
        "should",
        "may",
        "might",
        "must",
        "shall",
        "can",
        "need",
        "this",
        "that",
        "these",
        "those",
        "it",
        "its",
        "we",
        "our",
        "they",
        "their",
        "which",
        "who",
        "whom",
        "what",
        "where",
        "when",
        "why",
        "how",
        "all",
        "each",
        "every",
        "both",
        "few",
        "more",
        "most",
        "other",
        "some",
        "such",
        "no",
        "not",
        "only",
        "same",
        "so",
        "than",
        "too",
        "very",
        "just",
        "also",
        "now",
        "here",
        "there",
        "then",
        "once",
        "if",
        "any",
        "about",
        "into",
        "through",
        "during",
        "before",
        "after",
        "above",
        "below",
        "between",
        "under",
        "over",
        "out",
        "up",
        "down",
        "off",
        "paper",
        "study",
        "research",
        "results",
        "method",
        "methods",
        "approach",
        "proposed",
        "propose",
        "show",
        "shows",
        "shown",
        "using",
        "used",
        "use",
        "based",
        "problem",
        "problems",
        "work",
        "new",
        "novel",
        "present",
        "presented",
        "demonstrate",
        "demonstrates",
        "experimental",
        "experiments",
        "however",
        "therefore",
        "thus",
        "hence",
        "moreover",
        "furthermore",
        "first",
        "second",
        "third",
        "one",
        "two",
        "three",
        "many",
        "several",
        "various",
        "different",
        "important",
        "significant",
        "provide",
        "provides",
        "consider",
        "considers",
        "introduce",
        "introduces",
        "existing",
        "recent",
        "previous",
        "current",
        "given",
        "well",
        "known",
        "general",
        "particular",
        "specific",
        "case",
        "cases",
        "example",
        "examples",
        "order",
        "number",
        "large",
        "small",
        "high",
        "low",
        "best",
        "better",
        "good",
        "simple",
        "following",
        "related",
        "similar",
        "compared",
        "performance",
        "evaluate",
        "evaluated",
        "analysis",
        "data",
        "set",
        "sets",
        "model",
        "models",
    }
)


def _tfidf_keywords(
    texts: list[str],
    indices: list[int],
    corpus_freqs: Counter,
    stops: set[str] | frozenset[str],
    min_word_len: int,
    top_k: int,
    use_tfidf: bool,
) -> list[tuple[str, float]]:
    """Extract top-k keywords from a bucket using TF-IDF or frequency.

    Parameters
    ----------
    texts : list[str]
        Full corpus texts (indexed by *indices*).
    indices : list[int]
        Row indices belonging to the bucket.
    corpus_freqs : Counter
        Document-frequency counts across the whole corpus.
    stops : set[str] | frozenset[str]
        Stopwords to exclude.
    min_word_len : int
        Minimum word length to keep.
    top_k : int
        Number of keywords to return.
    use_tfidf : bool
        Whether to apply TF-IDF weighting (vs raw frequency).
    """
    import math
    import re

    def _tokenize(text: str) -> list[str]:
        text = text.lower()
        words = re.findall(r"\b[a-z]+\b", text)
        return [w for w in words if len(w) >= min_word_len and w not in stops]

    bucket_words: list[str] = []
    for idx in indices:
        bucket_words.extend(_tokenize(texts[idx]))

    if not bucket_words:
        return []

    word_counts = Counter(bucket_words)

    if use_tfidf and corpus_freqs:
        n_docs = len(texts)
        scores: dict[str, float] = {}
        for word, count in word_counts.items():
            tf = count / len(bucket_words)
            df = corpus_freqs.get(word, 1)
            idf = math.log(n_docs / df)
            scores[word] = tf * idf
        return sorted(scores.items(), key=lambda x: -x[1])[:top_k]
    else:
        return word_counts.most_common(top_k)


def _sample_near_centroid(
    indices: list[int],
    embeddings: np.ndarray,
    texts: list[str],
    k: int,
    rng: np.random.Generator | None = None,
) -> list[str]:
    """Return up to *k* texts closest to the bucket centroid.

    Parameters
    ----------
    indices : list[int]
        Row indices belonging to the bucket.
    embeddings : np.ndarray
        Full embedding matrix (n, d).
    texts : list[str]
        Full corpus texts.
    k : int
        Maximum number of samples to return.
    rng : np.random.Generator | None
        Unused — kept for API symmetry with the original nested function.
    """
    if len(indices) <= k:
        return [texts[i] for i in indices]

    bucket_embs = embeddings[indices]
    centroid = bucket_embs.mean(axis=0)
    centroid = centroid / np.linalg.norm(centroid)
    sims = bucket_embs @ centroid
    top_local = np.argsort(sims)[-k:][::-1]
    return [texts[indices[i]] for i in top_local]


# =============================================================================
# Data Classes
# =============================================================================


def _fmt_optional(value: float | None) -> str:
    """Render a diagnostic that may not have been computed.

    An uncomputed value says so rather than printing a number in the same format as a
    measured one — the whole point of making these Optional.
    """
    return "not computed" if value is None else f"{value:.4f}"


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
    #: None when the diagnostic was not computed. It used to be filled with a
    #: placeholder — 1.0 for stability, which is the *maximum* score, so a skipped
    #: computation was reported as a perfect one.
    mean_isolation_score: float | None
    mean_stability_score: float | None

    # PCA stats
    pca_variance_explained: float

    # Category breakdown (if provided)
    category_counts: list[tuple[str, int]]

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
            f"  Mean isolation score:     {_fmt_optional(self.mean_isolation_score)}",
            f"  Mean stability score:     {_fmt_optional(self.mean_stability_score)}",
            f"  PCA variance explained:   {self.pca_variance_explained:.1%}",
        ]

        if self.category_counts:
            lines.extend(
                [
                    "",
                    "TOP CATEGORIES",
                    "-" * 40,
                ]
            )
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
        num_stability_seeds: int = 0,
    ):
        """
        Initialize density classifier.

        Args:
            embedding_dim: Dimensionality of embeddings
            num_bits: Bits for PCA LSH (default: 14)
            seed: Random seed
            isolation_k: Number of top neighbors for isolation score
            isolation_sample_size: Sample size for median similarity computation
            num_stability_seeds: Number of seeds for stability scoring (default: 0, off)
        """
        self.embedding_dim = embedding_dim
        self.num_bits = num_bits
        self.seed = seed
        self.isolation_k = isolation_k
        self.isolation_sample_size = isolation_sample_size
        self.num_stability_seeds = num_stability_seeds

        # Populated during fit()
        self.embeddings: np.ndarray | None = None
        self.categories: list[str] | None = None
        self.texts: list[str] | None = None

        # Per-record metrics
        self._bucket_ids: np.ndarray | None = None
        self._bucket_sizes: np.ndarray | None = None
        self._centroid_similarities: np.ndarray | None = None
        self._isolation_scores: np.ndarray | None = None
        self._stability_scores: np.ndarray | None = None

        # Stats
        self._report: DensityReport | None = None
        self._pca_variance: float = 0.0
        self._fitted = False

        # TF-IDF components (for from_texts)
        self._vectorizer = None
        self._svd = None

        # Polars integration
        self._source_df: pl.DataFrame | None = None
        self._embedding_col: str | None = None

        # Pandas integration
        self._source_pandas_df: pd.DataFrame | None = None

    @classmethod
    def from_polars(
        cls,
        df: "pl.DataFrame",
        embedding_col: str,
        category_col: str | None = None,
        text_col: str | None = None,
        **kwargs,
    ) -> "DensityClassifier":
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

    def to_polars(self) -> "pl.DataFrame":
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
        labels_df = pl.DataFrame(
            {
                "bucket_id": self._bucket_ids.tolist(),
                "bucket_size": self._bucket_sizes.tolist(),
                "centroid_similarity": self._centroid_similarities.tolist(),
                "isolation_score": self._isolation_scores.tolist(),
                "stability_score": self._stability_score_column(),
            }
        )

        # If we have source DataFrame, join to it
        if self._source_df is not None:
            return pl.concat([self._source_df, labels_df], how="horizontal")
        else:
            # Add index column for manual joining
            n = len(self.embeddings)
            return labels_df.with_columns(pl.Series("index", list(range(n))))

    @classmethod
    def from_pandas(
        cls,
        df: "pd.DataFrame",
        embedding_col: str,
        category_col: str | None = None,
        text_col: str | None = None,
        **kwargs,
    ) -> "DensityClassifier":
        """
        Create classifier from a Pandas DataFrame.

        Args:
            df: Pandas DataFrame with embeddings
            embedding_col: Column name containing embedding vectors (list of floats)
            category_col: Optional column name for category labels
            text_col: Optional column name for text content (enables labeling)
            **kwargs: Additional args passed to __init__ (num_bits, seed, etc.)

        Returns:
            Fitted DensityClassifier instance

        Example:
            >>> df = pd.read_parquet("embeddings.parquet")
            >>> classifier = DensityClassifier.from_pandas(
            ...     df,
            ...     embedding_col="embedding",
            ...     category_col="category"
            ... )
            >>> result = classifier.to_pandas()
        """
        # Extract embeddings
        embeddings = np.array(df[embedding_col].tolist(), dtype=np.float32)

        # Extract optional columns
        categories = df[category_col].tolist() if category_col else None
        texts = df[text_col].tolist() if text_col else None

        # Create classifier
        classifier = cls(embedding_dim=embeddings.shape[1], **kwargs)

        # Store reference to source DataFrame
        classifier._source_pandas_df = df
        classifier._embedding_col = embedding_col

        # Fit
        classifier.fit(embeddings, categories=categories, texts=texts)

        return classifier

    def to_pandas(self) -> "pd.DataFrame":
        """
        Return source DataFrame with density metrics columns added.

        Returns DataFrame with original columns plus:
            - bucket_id: LSH bucket ID
            - bucket_size: Number of items in same bucket
            - centroid_similarity: Cosine similarity to bucket centroid (0-1)
            - isolation_score: How isolated the item is
            - stability_score: How stable bucket assignment is (0-1)

        Example:
            >>> classifier = DensityClassifier.from_pandas(df, "embedding")
            >>> result = classifier.to_pandas()
            >>> sparse = result[result["bucket_size"] < 5]
        """
        import pandas as pd

        if not self._fitted:
            raise ValueError("Must call fit() first")

        # Create labels DataFrame
        labels_df = pd.DataFrame(
            {
                "bucket_id": self._bucket_ids.tolist(),
                "bucket_size": self._bucket_sizes.tolist(),
                "centroid_similarity": self._centroid_similarities.tolist(),
                "isolation_score": self._isolation_scores.tolist(),
                "stability_score": self._stability_score_column(),
            }
        )

        # If we have source DataFrame, join to it
        if self._source_pandas_df is not None:
            return pd.concat([self._source_pandas_df.reset_index(drop=True), labels_df], axis=1)
        else:
            # Add index column for manual joining
            labels_df["index"] = range(len(self.embeddings))
            return labels_df

    @classmethod
    def from_texts(
        cls,
        texts: list[str],
        categories: list[str] | None = None,
        embedding_dim: int = 128,
        max_features: int = 10000,
        min_df: int = 2,
        max_df: float = 0.95,
        ngram_range: tuple[int, int] = (1, 2),
        num_bits: int = 12,
        verbose: bool = True,
        **kwargs,
    ) -> "DensityClassifier":
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
        from sklearn.decomposition import TruncatedSVD
        from sklearn.feature_extraction.text import TfidfVectorizer

        if verbose:
            logger.info(f"Building TF-IDF matrix ({len(texts):,} documents)...")

        # Build TF-IDF matrix
        vectorizer = TfidfVectorizer(
            max_features=max_features, min_df=min_df, max_df=max_df, ngram_range=ngram_range, stop_words="english"
        )
        tfidf_matrix = vectorizer.fit_transform(texts)

        if verbose:
            logger.debug(f"  Vocabulary size: {len(vectorizer.vocabulary_):,}")
            logger.debug(f"  Matrix shape: {tfidf_matrix.shape}")

        # Reduce to dense embeddings with SVD
        n_components = min(embedding_dim, tfidf_matrix.shape[1] - 1, len(texts) - 1)

        if verbose:
            logger.info(f"Applying SVD ({n_components} components)...")

        svd = TruncatedSVD(n_components=n_components, random_state=kwargs.get("seed", 31))
        embeddings = svd.fit_transform(tfidf_matrix).astype(np.float32)

        if verbose:
            logger.debug(f"  Variance explained: {svd.explained_variance_ratio_.sum():.1%}")

        # Create and fit classifier
        classifier = cls(embedding_dim=n_components, num_bits=num_bits, **kwargs)

        # Store vectorizer and SVD for potential later use
        classifier._vectorizer = vectorizer
        classifier._svd = svd

        classifier.fit(embeddings, categories=categories, texts=texts, verbose=verbose)

        return classifier

    def fit(
        self,
        embeddings: np.ndarray,
        categories: list[str] | None = None,
        texts: list[str] | None = None,
        normalize: bool = True,
        verbose: bool = True,
    ) -> "DensityClassifier":
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
            logger.info(f"Corpus size: {n:,}, dim: {d}")

        # Stage 1: PCA-based LSH hashing
        if verbose:
            logger.info(f"PCA-based LSH ({self.num_bits} bits)...")

        hashes, hp = self._pca_hash(embeddings, self.num_bits, self.seed, verbose)
        self._bucket_ids = hashes.copy()

        # Build bucket mapping
        bucket_to_indices = defaultdict(list)
        for idx, h in enumerate(hashes):
            bucket_to_indices[int(h)].append(idx)

        num_buckets = len(bucket_to_indices)

        # Compute density metrics (bucket sizes + centroid similarities)
        self._compute_density_metrics(embeddings, bucket_to_indices, n)

        # Compute isolation scores
        if verbose:
            logger.debug("  Computing isolation scores...")
        self._compute_isolation_scores()

        # Compute stability scores
        if verbose:
            logger.debug(f"  Computing stability scores ({self.num_stability_seeds} seeds)...")
        self._compute_stability_scores(hp)

        # Build report
        self._build_report(num_buckets)

        if verbose:
            logger.debug(f"  Buckets: {num_buckets:,}")
            logger.debug(f"  Mean bucket size: {self._report.mean_bucket_size:.1f}")
            logger.debug(f"  Mean isolation score: {_fmt_optional(self._report.mean_isolation_score)}")

        self._fitted = True
        return self

    def _pca_hash(
        self,
        embeddings: np.ndarray,
        num_bits: int,
        seed: int,
        verbose: bool = False,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Two-stage PCA-based LSH hashing.

        1. Random hyperplane hash to form initial buckets.
        2. Compute bucket centroids, run PCA on them.
        3. Re-hash using PCA components as hyperplanes.

        Returns
        -------
        hashes : np.ndarray
            (n,) uint64 hash codes.
        hp : np.ndarray
            (num_bits, d) PCA hyperplane matrix (needed for stability scoring).
        """
        d = embeddings.shape[1]

        # Step 1a: Random hash
        rng = np.random.default_rng(seed)
        random_hp = rng.standard_normal((num_bits, d)).astype(np.float32)
        random_hp = random_hp / np.linalg.norm(random_hp, axis=1, keepdims=True)

        signs_random = (embeddings @ random_hp.T) >= 0
        powers = 2 ** np.arange(num_bits)
        hashes_random = (signs_random @ powers).astype(np.uint64)

        # Step 1b: Compute bucket centroids
        random_bucket_to_indices: dict[int, list[int]] = defaultdict(list)
        for idx, h in enumerate(hashes_random):
            random_bucket_to_indices[int(h)].append(idx)

        centroids: list[np.ndarray] = []
        for bid, indices in random_bucket_to_indices.items():
            if len(indices) >= 2:
                centroid = embeddings[indices].mean(axis=0)
                norm = np.linalg.norm(centroid)
                if norm > 0:
                    centroids.append(centroid / norm)

        centroids_arr = np.array(centroids, dtype=np.float32)

        if verbose:
            logger.debug(f"  Random hash: {len(random_bucket_to_indices):,} buckets")
            logger.debug(f"  Centroids for PCA: {len(centroids_arr):,}")

        # Step 1c: PCA on centroids
        from sklearn.decomposition import PCA

        n_components = min(num_bits, len(centroids_arr) - 1)
        pca = PCA(n_components=n_components)
        pca.fit(centroids_arr)
        hp = pca.components_.astype(np.float32)
        self._pca_variance = float(pca.explained_variance_ratio_.sum())

        if verbose:
            logger.debug(f"  Centroid PCA variance: {self._pca_variance:.1%}")

        # Step 1d: Re-hash with PCA hyperplanes
        signs = (embeddings @ hp.T) >= 0
        hashes = (signs @ powers[: len(hp)]).astype(np.uint64)

        return hashes, hp

    def _compute_density_metrics(
        self,
        embeddings: np.ndarray,
        bucket_to_indices: dict[int, list[int]],
        n: int,
    ) -> None:
        """Compute per-item bucket sizes and centroid similarities.

        Populates ``self._bucket_sizes`` and ``self._centroid_similarities``.
        """
        self._bucket_sizes = np.zeros(n, dtype=np.int32)
        self._centroid_similarities = np.zeros(n, dtype=np.float32)

        for bid, indices in bucket_to_indices.items():
            bucket_size = len(indices)

            for idx in indices:
                self._bucket_sizes[idx] = bucket_size

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

    # Rows per similarity block. The full n x sample_size block would be 800MB at
    # n=200k, sample=1000; this bounds it to ~32MB.
    _ISOLATION_ROW_CHUNK = 8192

    def _compute_isolation_scores(self):
        """Compute isolation score for each item: top_k_mean - median.

        The similarity block is a matrix product — embeddings(n x d) @ sample(d x m) —
        so it is computed in one BLAS call per chunk. The previous version looped over
        items and re-evaluated ``self.embeddings[sample_indices]`` *inside* the loop,
        which is a fancy-index copy of the entire m x d sample rebuilt n times, then
        sorted all m similarities to read two positions out of them. Measured at
        n=20,000 d=384: 1.92s before, 0.23s after, identical scores.
        """
        n = len(self.embeddings)

        # Sample for median computation
        rng = np.random.default_rng(self.seed + 12345)
        sample_size = min(self.isolation_sample_size, n)
        sample_indices = rng.choice(n, sample_size, replace=False)

        k = min(self.isolation_k, sample_size)
        m = sample_size

        # np.median averages the two central order statistics for even m. Preserve that
        # exactly — taking a single element instead shifts every score by ~1e-3, which
        # is small enough to look like float noise and is not.
        lo = m // 2 - 1 if m % 2 == 0 else m // 2
        hi = m // 2
        kth = sorted({lo, hi, m - k})

        sample = np.ascontiguousarray(self.embeddings[sample_indices])
        # Where each sampled item sits in the block, so it can mask its own similarity.
        pos_of = {int(j): p for p, j in enumerate(sample_indices)}

        scores = np.empty(n, dtype=np.float32)
        for start in range(0, n, self._ISOLATION_ROW_CHUNK):
            stop = min(start + self._ISOLATION_ROW_CHUNK, n)
            sims = self.embeddings[start:stop] @ sample.T

            # Exclude self, as the original did, by pushing it below every real
            # similarity rather than removing it — the sample stays m wide.
            for r in range(stop - start):
                p = pos_of.get(start + r)
                if p is not None:
                    sims[r, p] = -2.0

            part = np.partition(sims, kth, axis=1)
            median = (part[:, lo] + part[:, hi]) / 2.0
            scores[start:stop] = part[:, m - k :].mean(axis=1) - median

        self._isolation_scores = scores

    def _compute_stability_scores(self, hp: np.ndarray):
        """Compute stability score: how consistently items stay in same bucket across seeds."""
        n = len(self.embeddings)
        num_seeds = self.num_stability_seeds

        if num_seeds < 2:
            # Stability cannot be measured with fewer than 2 seeds: there is nothing to
            # compare across. Leave it absent rather than filling 1.0 — that is the
            # *maximum* score, so a skipped computation read as a perfect result, and
            # `report()` printed it in the same format as a measured value.
            self._stability_scores = None
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
            mean_isolation_score=(float(self._isolation_scores.mean()) if self._isolation_scores is not None else None),
            mean_stability_score=(float(self._stability_scores.mean()) if self._stability_scores is not None else None),
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

    def _stability_score_column(self) -> list:
        """Stability as a DataFrame column, null when it was not computed.

        A null column is honest about the absence; a column of 1.0 would claim every
        item was maximally stable.
        """
        if self._stability_scores is None:
            return [None] * len(self._bucket_ids)
        return self._stability_scores.tolist()

    def get_stability_scores(self) -> np.ndarray | None:
        """Get stability scores for all items (0-1, higher = more stable).

        None when stability was not computed — which includes the default
        ``num_stability_seeds < 2``, where there is nothing to compare across.
        """
        if not self._fitted:
            raise ValueError("Must call fit() first")
        if self._stability_scores is None:
            return None
        return self._stability_scores.copy()

    def get_labels(self) -> "pl.DataFrame":
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

        return pl.DataFrame(
            {
                "index": list(range(n)),
                "bucket_id": self._bucket_ids.tolist(),
                "bucket_size": self._bucket_sizes.tolist(),
                "centroid_similarity": self._centroid_similarities.tolist(),
                "isolation_score": self._isolation_scores.tolist(),
                "stability_score": self._stability_score_column(),
                "category": self.categories,
            }
        )

    def label_buckets(
        self,
        base_url: str = "http://localhost:11434/v1",
        model: str = "qwen2.5:7b",
        samples_per_bucket: int = 5,
        max_text_len: int = 200,
        min_bucket_size: int = 5,
        verbose: bool = True,
    ) -> dict[int, dict]:
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

        def get_label(samples: list[str]) -> str:
            """Get label from LLM for a set of samples."""
            samples_text = "\n".join(f"- {s[:max_text_len]}" for s in samples)
            prompt = f"""These are sample texts from a document cluster:
{samples_text}

What 2-5 word label describes this cluster's shared topic or theme?
Be specific and descriptive. Just output the label, nothing else.
Label:"""

            try:
                response = client.chat.completions.create(
                    model=model, messages=[{"role": "user", "content": prompt}], max_tokens=30, temperature=0.3
                )
                return response.choices[0].message.content.strip()
            except Exception as e:
                logger.debug("LLM labeling failed: %s", e)
                return f"[Error: {e}]"

        # Build bucket -> indices mapping
        bucket_to_indices = defaultdict(list)
        for idx, bucket_id in enumerate(self._bucket_ids):
            bucket_to_indices[int(bucket_id)].append(idx)

        # Label buckets
        buckets_to_label = [
            (bid, indices) for bid, indices in bucket_to_indices.items() if len(indices) >= min_bucket_size
        ]

        if verbose:
            logger.info(f"Labeling {len(buckets_to_label)} buckets...")

        results = {}
        for i, (bucket_id, indices) in enumerate(buckets_to_label):
            samples = _sample_near_centroid(
                indices,
                self.embeddings,
                self.texts,
                samples_per_bucket,
            )
            label = get_label(samples)
            results[bucket_id] = {"label": label, "size": len(indices), "samples": samples}
            if verbose and (i + 1) % 10 == 0:
                logger.info(f"  {i + 1}/{len(buckets_to_label)} buckets labeled")

        if verbose:
            logger.info("Done.")

        return results

    def label_buckets_keywords(
        self,
        top_k: int = 3,
        min_bucket_size: int = 5,
        stopwords: set | None = None,
        min_word_len: int = 3,
        use_tfidf: bool = True,
    ) -> dict[int, dict]:
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

        if not self._fitted:
            raise ValueError("Must call fit() first")
        if self.texts is None:
            raise ValueError("Texts required for labeling. Pass texts to fit().")

        stops = stopwords if stopwords is not None else _DEFAULT_ACADEMIC_STOPWORDS

        def _tokenize_local(text: str) -> list[str]:
            text = text.lower()
            words = re.findall(r"\b[a-z]+\b", text)
            return [w for w in words if len(w) >= min_word_len and w not in stops]

        # Build corpus document frequencies
        corpus_freqs: Counter = Counter()
        if use_tfidf:
            for text in self.texts:
                unique_words = set(_tokenize_local(text))
                corpus_freqs.update(unique_words)

        # Build bucket -> indices mapping
        bucket_to_indices = defaultdict(list)
        for idx, bucket_id in enumerate(self._bucket_ids):
            bucket_to_indices[int(bucket_id)].append(idx)

        results = {}
        for bucket_id, indices in bucket_to_indices.items():
            if len(indices) < min_bucket_size:
                continue

            keywords = _tfidf_keywords(
                self.texts,
                indices,
                corpus_freqs,
                stops,
                min_word_len,
                top_k,
                use_tfidf,
            )
            label = " ".join(kw for kw, _ in keywords) if keywords else "[no keywords]"

            results[bucket_id] = {"label": label, "keywords": keywords, "size": len(indices)}

        return results
