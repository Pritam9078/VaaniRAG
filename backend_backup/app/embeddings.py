"""
embeddings.py
-------------
Pluggable embedding backend.

Production recommendation (see README): sentence-transformers
'all-MiniLM-L6-v2' (384-dim) or a multilingual variant such as
'paraphrase-multilingual-MiniLM-L12-v2' for Indic-language coverage.

This sandbox has no route to huggingface.co, so pretrained embedding
weights cannot be downloaded here. To keep the pipeline fully runnable
offline, the default backend below is a TF-IDF + TruncatedSVD pipeline
fit directly on the corpus, which behaves as a legitimate (if weaker)
dense embedding for demonstration and latency-benchmarking purposes.
Swap in `SentenceTransformerEmbedder` (stubbed below) once you have
network access to download real weights.
"""

from __future__ import annotations

import numpy as np
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer


class BaseEmbedder:
    dim: int

    def fit(self, corpus: list[str]) -> BaseEmbedder:
        raise NotImplementedError

    def encode(self, texts: list[str]) -> np.ndarray:
        raise NotImplementedError


class TfidfSvdEmbedder(BaseEmbedder):
    """Offline-friendly dense embedding proxy: TF-IDF -> SVD (LSA)."""

    def __init__(self, dim: int = 128):
        self.dim = dim
        self.vectorizer = TfidfVectorizer(
            analyzer="word",
            token_pattern=r"(?u)[A-Za-z\u0900-\u097F]{2,}",
            max_features=20000,
        )
        self.svd = None

    def fit(self, corpus: list[str]) -> TfidfSvdEmbedder:
        X = self.vectorizer.fit_transform(corpus)
        n_components = min(self.dim, max(2, min(X.shape) - 1))
        self.svd = TruncatedSVD(n_components=n_components, random_state=42)
        self.svd.fit(X)
        self.dim = n_components
        return self

    def encode(self, texts: list[str]) -> np.ndarray:
        X = self.vectorizer.transform(texts)
        vecs = self.svd.transform(X)
        norms = np.linalg.norm(vecs, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return (vecs / norms).astype("float32")


class SentenceTransformerEmbedder(BaseEmbedder):
    """Real embedding backend for production use once model weights are
    reachable (e.g. running outside this sandbox, or with weights
    pre-cached). Not used by default here.

        pip install sentence-transformers
        model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
    """

    def __init__(self, model_name: str = "paraphrase-multilingual-MiniLM-L12-v2"):
        from sentence_transformers import SentenceTransformer  # noqa: local import
        self.model = SentenceTransformer(model_name)
        self.dim = self.model.get_sentence_embedding_dimension()

    def fit(self, corpus: list[str]) -> SentenceTransformerEmbedder:
        return self  # pretrained, no fitting needed

    def encode(self, texts: list[str]) -> np.ndarray:
        vecs = self.model.encode(texts, normalize_embeddings=True)
        return np.asarray(vecs, dtype="float32")


def get_embedder(backend: str = "tfidf_svd", **kwargs) -> BaseEmbedder:
    if backend == "tfidf_svd":
        return TfidfSvdEmbedder(**kwargs)
    if backend == "sentence_transformer":
        return SentenceTransformerEmbedder(**kwargs)
    raise ValueError(f"Unknown embedding backend: {backend}")
