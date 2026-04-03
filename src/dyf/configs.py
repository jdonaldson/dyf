"""
Embedding and labeling configuration presets.

EmbedderConfig: Configuration for text embedding models (TF-IDF, sentence-transformers, OpenAI).
LabelerConfig: Configuration for LLM-based bucket labeling (Ollama, MLX).

Example:
    >>> from dyf import EmbedderConfig, LabelerConfig
    >>> embeddings = EmbedderConfig.MEDIUM.embed(texts)
    >>> labeler_kwargs = LabelerConfig.MEDIUM.as_kwargs()
"""

from dataclasses import dataclass

import numpy as np


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

    def embed(self, texts: list[str], batch_size: int = 32, verbose: bool = True) -> np.ndarray:
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

    def _embed_tfidf(self, texts: list[str], verbose: bool) -> np.ndarray:
        """TF-IDF + SVD embeddings."""
        from sklearn.decomposition import TruncatedSVD
        from sklearn.feature_extraction.text import TfidfVectorizer

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

    def _embed_bm25(self, texts: list[str], verbose: bool) -> np.ndarray:
        """BM25-weighted + SVD embeddings (saturated term frequencies)."""
        from scipy import sparse
        from sklearn.decomposition import TruncatedSVD
        from sklearn.feature_extraction.text import CountVectorizer

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

    def _embed_sentence_transformers(self, texts: list[str], batch_size: int, verbose: bool) -> np.ndarray:
        """Sentence-transformers embeddings."""
        from sentence_transformers import SentenceTransformer

        if verbose:
            print(f"Loading {self.model_id}...")

        model = SentenceTransformer(self.model_id, trust_remote_code=True)
        embeddings = model.encode(
            texts, batch_size=batch_size,
            show_progress_bar=verbose,
            convert_to_numpy=True
        )
        return embeddings.astype(np.float32)

    def _embed_openai(self, texts: list[str], batch_size: int, verbose: bool) -> np.ndarray:
        """OpenAI API embeddings."""
        import os

        from openai import OpenAI

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

    def as_kwargs(self, use_mlx: bool = False) -> dict:
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
