"""Build page-level citations with query-focused excerpts."""

from __future__ import annotations

import html
import re
from dataclasses import asdict, dataclass

from rag.chunking import sentence_spans
from rag.vector_store import SearchResult

STOPWORDS = {
    "a", "an", "the", "and", "or", "but", "if", "then", "of", "to", "in", "on", "at", "by", "for", "with",
    "about", "as", "is", "are", "was", "were", "be", "been", "being", "it", "its", "this", "that", "these",
    "those", "i", "you", "he", "she", "we", "they", "them", "my", "your", "our", "their", "what", "which",
    "who", "whom", "when", "where", "why", "how", "do", "does", "did", "can", "could", "should", "would",
    "will", "shall", "may", "might", "have", "has", "had", "not", "no", "so", "than", "too", "very", "me",
    "there", "here", "from", "into", "over", "under", "again", "any", "all", "some", "such", "tell", "please",
}

_WORD = re.compile(r"[a-z0-9]+")


@dataclass
class Citation:
    doc_name: str
    page_number: int
    excerpt: str
    score: float
    chunk_id: str
    rank: int

    @property
    def label(self) -> str:
        return f"{self.doc_name}, p. {self.page_number}"

    def to_dict(self) -> dict:
        data = asdict(self)
        data["label"] = self.label
        return data

    @classmethod
    def from_dict(cls, data: dict) -> "Citation":
        data = {k: v for k, v in data.items() if k != "label"}
        return cls(**data)


def query_terms(query: str) -> list[str]:
    """Content words from the query (lower-cased, stopwords removed, de-duplicated)."""
    seen: dict[str, None] = {}
    for tok in _WORD.findall(query.lower()):
        if len(tok) > 1 and tok not in STOPWORDS:
            seen.setdefault(tok, None)
    return list(seen)


def _term_hits(text: str, terms: list[str]) -> int:
    words = set(_WORD.findall(text.lower()))
    return sum(1 for t in terms if t in words or any(w.startswith(t) for w in words))


def best_excerpt(text: str, query: str, max_chars: int = 320) -> str:
    """Return a window of consecutive sentences (<= max_chars) with the most query-term hits."""
    text = " ".join(text.split())
    if len(text) <= max_chars:
        return text
    terms = query_terms(query)
    spans = sentence_spans(text) or [(0, len(text))]
    sentences = [text[s:e].strip() for s, e in spans]

    best_window, best_hits = None, -1
    for i in range(len(sentences)):
        window: list[str] = []
        length = 0
        for sentence in sentences[i:]:
            if length + len(sentence) + (1 if window else 0) > max_chars:
                break
            window.append(sentence)
            length += len(sentence) + 1
        if not window:  # single sentence longer than max_chars: trim it
            window = [sentences[i][: max_chars - 1]]
        hits = _term_hits(" ".join(window), terms)
        if hits > best_hits:
            best_window, best_hits = window, hits
            if hits == len(terms) and terms:
                break
    excerpt = " ".join(best_window or [text[:max_chars]])
    if not text.startswith(excerpt):
        excerpt = "… " + excerpt
    if not text.endswith(excerpt.lstrip("… ")):
        excerpt = excerpt + " …"
    return excerpt


def build_citations(results: list[SearchResult], query: str, max_excerpt_chars: int = 320) -> list[Citation]:
    citations: list[Citation] = []
    for result in results:
        chunk = result.chunk
        citations.append(
            Citation(
                doc_name=chunk.doc_name,
                page_number=chunk.page_number,
                excerpt=best_excerpt(chunk.text, query, max_excerpt_chars),
                score=round(float(result.score), 4),
                chunk_id=chunk.chunk_id,
                rank=result.rank,
            )
        )
    return citations


def unique_labels(citations: list[Citation]) -> list[str]:
    """Distinct ``doc, p. N`` labels in rank order."""
    seen: dict[str, None] = {}
    for c in citations:
        seen.setdefault(c.label, None)
    return list(seen)


def highlight_html(text: str, query: str) -> str:
    """HTML-escape ``text`` and wrap query terms in ``<mark>`` for display."""
    escaped = html.escape(text)
    terms = sorted(query_terms(query), key=len, reverse=True)
    if not terms:
        return escaped
    pattern = re.compile(r"\b(" + "|".join(re.escape(t) for t in terms) + r")\w*", re.IGNORECASE)
    return pattern.sub(lambda m: f"<mark>{m.group(0)}</mark>", escaped)
