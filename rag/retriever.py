"""Semantic retrieval: embed the question, search FAISS, decide whether the
best match is strong enough to count as evidence.

The evidence gate is hybrid.  A result set counts as evidence when the best
cosine score reaches ``score_threshold`` **and** either (a) the score is very
strong (``strong_score``) or (b) at least one content word of the query
appears in the retrieved passages.  Small embedding models rate *topically*
similar text highly (e.g. "library pets policy" vs. any library policy page);
the lexical check stops those false positives without blocking paraphrased
questions, which usually still share a keyword or clear the strong threshold.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Iterable

from rag.citations import query_terms
from rag.embeddings import Embedder
from rag.vector_store import FaissVectorStore, SearchResult

_WORD = re.compile(r"[a-z0-9]+")


def _stem(term: str, min_len: int = 3) -> str:
    return term[: max(min_len, len(term) - 1)] if term.endswith("s") else term


def lexical_overlap(query: str, texts: Iterable[str], min_term_len: int = 3,
                    ignore_terms: Iterable[str] | None = None) -> bool:
    """True when any informative query word (prefix-matched, so 'fines' ~ 'fine') occurs in ``texts``.

    ``ignore_terms`` are words too common in the corpus to be informative (e.g. "library" in a
    library policy); they are skipped unless *every* query word is on that list.
    """
    terms = [t for t in query_terms(query) if len(t) >= min_term_len]
    if not terms:
        return True  # nothing to check (e.g. a query made only of stopwords)
    ignored = set(ignore_terms or ())
    informative = [t for t in terms if t not in ignored]
    if informative:
        terms = informative
    words: set[str] = set()
    for text in texts:
        words.update(_WORD.findall(text.lower()))
    for term in terms:
        stem = _stem(term, min_term_len)
        if term in words or any(w.startswith(stem) for w in words):
            return True
    return False


def corpus_frequent_terms(texts: Iterable[str], max_fraction: float = 0.3, min_chunks: int = 4) -> set[str]:
    """Words that occur in more than ``max_fraction`` of the given chunks: corpus-specific stopwords."""
    texts = list(texts)
    if len(texts) < min_chunks:
        return set()
    counts: dict[str, int] = {}
    for text in texts:
        for w in set(_WORD.findall(text.lower())):
            counts[w] = counts.get(w, 0) + 1
    limit = len(texts) * max_fraction
    frequent = {w for w, n in counts.items() if n > limit}
    # also treat singular/plural variants of frequent words as frequent
    return frequent | {_stem(w) for w in frequent}


def document_frequent_terms(chunks, max_fraction: float = 0.3, min_chunks: int = 4) -> set[str]:
    """Union of per-document frequent terms (so "library" is ignored for a library policy
    even when other documents are loaded alongside it)."""
    by_doc: dict[str, list[str]] = {}
    for c in chunks:
        by_doc.setdefault(c.doc_name, []).append(c.text)
    frequent: set[str] = set()
    for texts in by_doc.values():
        frequent |= corpus_frequent_terms(texts, max_fraction, min_chunks)
    return frequent


@dataclass
class RetrievalResult:
    query: str
    results: list[SearchResult] = field(default_factory=list)
    threshold: float = 0.0
    strong_threshold: float = 0.50
    lexical_match: bool = True

    @property
    def best_score(self) -> float:
        return self.results[0].score if self.results else 0.0

    @property
    def has_evidence(self) -> bool:
        if not self.results or self.best_score < self.threshold:
            return False
        return self.lexical_match or self.best_score >= self.strong_threshold

    @property
    def chunks(self):
        return [r.chunk for r in self.results]


class Retriever:
    def __init__(self, embedder: Embedder, store: FaissVectorStore, top_k: int = 5,
                 score_threshold: float = 0.30, strong_score: float = 0.50):
        self.embedder = embedder
        self.store = store
        self.top_k = top_k
        self.score_threshold = score_threshold
        self.strong_score = strong_score
        self._frequent_cache: tuple[int, set[str]] = (-1, set())

    def frequent_terms(self) -> set[str]:
        """Corpus-specific stopwords, cached until the store changes size."""
        size = len(self.store)
        if self._frequent_cache[0] != size:
            self._frequent_cache = (size, document_frequent_terms(self.store.chunks))
        return self._frequent_cache[1]

    def retrieve(self, query: str, top_k: int | None = None, doc_names: Iterable[str] | None = None) -> RetrievalResult:
        query = (query or "").strip()
        k = top_k or self.top_k
        if not query or self.store.is_empty():
            return RetrievalResult(query=query, threshold=self.score_threshold, strong_threshold=self.strong_score)
        query_vector = self.embedder.embed_query(query)
        results = self.store.search(query_vector, top_k=k, doc_names=doc_names)
        frequent = self.frequent_terms()
        overlap = lexical_overlap(query, (r.chunk.text for r in results), ignore_terms=frequent)
        return RetrievalResult(query=query, results=results, threshold=self.score_threshold,
                               strong_threshold=self.strong_score, lexical_match=overlap)
