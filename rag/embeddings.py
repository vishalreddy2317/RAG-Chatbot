"""Embedding back-ends.

* :class:`EmbeddingModel` wraps a local Sentence Transformers model (free, runs
  on CPU).  Vectors are L2-normalised so inner product == cosine similarity.
* :class:`HashingEmbedder` is a dependency-free, deterministic bag-of-words
  embedder used by the unit tests so they run offline in milliseconds.
"""

from __future__ import annotations

import hashlib
import re
from typing import Callable, Protocol, Sequence

import numpy as np

from rag.config import DEFAULT_EMBEDDING_MODEL

ProgressCallback = Callable[[int, int], None]


class Embedder(Protocol):
    model_name: str

    @property
    def dimension(self) -> int: ...

    def embed_texts(self, texts: Sequence[str], batch_size: int = 32,
                    progress_callback: ProgressCallback | None = None) -> np.ndarray: ...

    def embed_query(self, text: str) -> np.ndarray: ...


def _normalize(matrix: np.ndarray) -> np.ndarray:
    matrix = np.asarray(matrix, dtype="float32")
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return matrix / norms


class EmbeddingModel:
    """Lazy-loading wrapper around ``sentence_transformers.SentenceTransformer``."""

    def __init__(self, model_name: str = DEFAULT_EMBEDDING_MODEL, device: str | None = None):
        self.model_name = model_name
        self.device = device
        self._model = None

    @property
    def model(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer  # imported lazily: slow import

            self._model = SentenceTransformer(self.model_name, device=self.device)
        return self._model

    @property
    def dimension(self) -> int:
        getter = getattr(self.model, "get_embedding_dimension", None) or self.model.get_sentence_embedding_dimension
        return int(getter())

    def embed_texts(self, texts: Sequence[str], batch_size: int = 32,
                    progress_callback: ProgressCallback | None = None) -> np.ndarray:
        texts = list(texts)
        if not texts:
            return np.zeros((0, self.dimension), dtype="float32")
        parts: list[np.ndarray] = []
        total = len(texts)
        for start in range(0, total, batch_size):
            batch = texts[start:start + batch_size]
            vectors = self.model.encode(
                batch,
                batch_size=batch_size,
                convert_to_numpy=True,
                normalize_embeddings=True,
                show_progress_bar=False,
            )
            parts.append(np.asarray(vectors, dtype="float32"))
            if progress_callback is not None:
                progress_callback(min(start + len(batch), total), total)
        return _normalize(np.vstack(parts))

    def embed_query(self, text: str) -> np.ndarray:
        return self.embed_texts([text])[0]


_TOKEN = re.compile(r"[a-z0-9]+")


class HashingEmbedder:
    """Deterministic hashed bag-of-words embedder (unigrams + bigrams).

    It captures lexical overlap only - good enough to unit-test the vector
    store, retriever, citations and pipeline without downloading a model.
    """

    model_name = "hashing-bow"

    def __init__(self, dimension: int = 256):
        self._dimension = dimension

    @property
    def dimension(self) -> int:
        return self._dimension

    def _embed_one(self, text: str) -> np.ndarray:
        vec = np.zeros(self._dimension, dtype="float32")
        tokens = _TOKEN.findall(text.lower())
        features = tokens + [f"{a}_{b}" for a, b in zip(tokens, tokens[1:])]
        for feature in features:
            digest = hashlib.md5(feature.encode("utf-8")).digest()
            idx = int.from_bytes(digest[:4], "little") % self._dimension
            sign = 1.0 if digest[4] % 2 == 0 else -1.0
            vec[idx] += sign
        return vec

    def embed_texts(self, texts: Sequence[str], batch_size: int = 32,
                    progress_callback: ProgressCallback | None = None) -> np.ndarray:
        texts = list(texts)
        if not texts:
            return np.zeros((0, self._dimension), dtype="float32")
        matrix = np.vstack([self._embed_one(t) for t in texts])
        if progress_callback is not None:
            progress_callback(len(texts), len(texts))
        return _normalize(matrix)

    def embed_query(self, text: str) -> np.ndarray:
        return self.embed_texts([text])[0]
