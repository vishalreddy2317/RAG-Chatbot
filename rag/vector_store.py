"""FAISS-backed vector store that keeps chunk metadata next to the vectors.

Uses ``IndexFlatIP`` (exact inner-product search).  Because all embeddings are
L2-normalised, the inner product equals cosine similarity, so scores are in
``[-1, 1]`` and comparable across queries.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import faiss
import numpy as np

from rag.chunking import Chunk


@dataclass
class SearchResult:
    chunk: Chunk
    score: float  # cosine similarity
    rank: int     # 1-based rank in the result list


class FaissVectorStore:
    def __init__(self, dimension: int):
        if dimension <= 0:
            raise ValueError("dimension must be positive")
        self.dimension = int(dimension)
        self._index = faiss.IndexFlatIP(self.dimension)
        self._chunks: list[Chunk] = []
        self._embeddings = np.zeros((0, self.dimension), dtype="float32")

    # ------------------------------------------------------------------ basics
    def __len__(self) -> int:
        return int(self._index.ntotal)

    @property
    def chunks(self) -> list[Chunk]:
        return list(self._chunks)

    def is_empty(self) -> bool:
        return len(self) == 0

    # --------------------------------------------------------------- mutation
    def add(self, chunks: Sequence[Chunk], embeddings: np.ndarray) -> int:
        chunks = list(chunks)
        if not chunks:
            return 0
        matrix = np.ascontiguousarray(np.asarray(embeddings, dtype="float32"))
        if matrix.ndim != 2 or matrix.shape[0] != len(chunks):
            raise ValueError(f"embeddings must have shape ({len(chunks)}, {self.dimension}), got {matrix.shape}")
        if matrix.shape[1] != self.dimension:
            raise ValueError(f"embedding dimension {matrix.shape[1]} != store dimension {self.dimension}")
        faiss.normalize_L2(matrix)
        self._index.add(matrix)
        self._chunks.extend(chunks)
        self._embeddings = np.vstack([self._embeddings, matrix])
        return len(chunks)

    def remove_document(self, doc_name: str) -> int:
        keep = [i for i, c in enumerate(self._chunks) if c.doc_name != doc_name]
        removed = len(self._chunks) - len(keep)
        if removed == 0:
            return 0
        self._chunks = [self._chunks[i] for i in keep]
        self._embeddings = self._embeddings[keep] if keep else np.zeros((0, self.dimension), dtype="float32")
        self._rebuild()
        return removed

    def clear(self) -> None:
        self._chunks = []
        self._embeddings = np.zeros((0, self.dimension), dtype="float32")
        self._rebuild()

    def _rebuild(self) -> None:
        self._index = faiss.IndexFlatIP(self.dimension)
        if len(self._embeddings):
            self._index.add(np.ascontiguousarray(self._embeddings))

    # ----------------------------------------------------------------- search
    def search(self, query_embedding: np.ndarray, top_k: int = 5,
               doc_names: Iterable[str] | None = None) -> list[SearchResult]:
        if self.is_empty() or top_k <= 0:
            return []
        query = np.ascontiguousarray(np.asarray(query_embedding, dtype="float32").reshape(1, -1))
        if query.shape[1] != self.dimension:
            raise ValueError(f"query dimension {query.shape[1]} != store dimension {self.dimension}")
        faiss.normalize_L2(query)
        allowed = set(doc_names) if doc_names else None
        # When filtering by document we over-fetch and filter afterwards.
        k = len(self) if allowed else min(top_k, len(self))
        scores, ids = self._index.search(query, k)
        results: list[SearchResult] = []
        for score, idx in zip(scores[0], ids[0]):
            if idx < 0:
                continue
            chunk = self._chunks[int(idx)]
            if allowed is not None and chunk.doc_name not in allowed:
                continue
            results.append(SearchResult(chunk=chunk, score=float(score), rank=len(results) + 1))
            if len(results) >= top_k:
                break
        return results

    # ------------------------------------------------------------- documents
    def document_names(self) -> list[str]:
        seen: dict[str, None] = {}
        for c in self._chunks:
            seen.setdefault(c.doc_name, None)
        return list(seen)

    def document_stats(self) -> dict[str, dict]:
        stats: dict[str, dict] = {}
        for c in self._chunks:
            entry = stats.setdefault(c.doc_name, {"chunks": 0, "pages": set()})
            entry["chunks"] += 1
            entry["pages"].add(c.page_number)
        return {name: {"chunks": v["chunks"], "pages": len(v["pages"])} for name, v in stats.items()}

    # ------------------------------------------------------------ persistence
    def save(self, directory: str | Path) -> None:
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self._index, str(directory / "index.faiss"))
        np.save(directory / "embeddings.npy", self._embeddings)
        with open(directory / "chunks.json", "w", encoding="utf-8") as fh:
            json.dump({"dimension": self.dimension, "chunks": [c.to_dict() for c in self._chunks]}, fh)

    @classmethod
    def load(cls, directory: str | Path) -> "FaissVectorStore":
        directory = Path(directory)
        with open(directory / "chunks.json", encoding="utf-8") as fh:
            payload = json.load(fh)
        store = cls(int(payload["dimension"]))
        store._chunks = [Chunk.from_dict(d) for d in payload["chunks"]]
        store._embeddings = np.load(directory / "embeddings.npy").astype("float32")
        store._index = faiss.read_index(str(directory / "index.faiss"))
        if store._index.ntotal != len(store._chunks):  # defensive: keep metadata and vectors in sync
            store._rebuild()
        return store
