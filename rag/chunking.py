"""Sentence-aware, overlapping text chunking that preserves page numbers.

Chunks never cross page boundaries, so every chunk maps to exactly one
``(document, page)`` pair, which is what makes page-level citations exact.
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from typing import Iterable

from rag.pdf_loader import PageText

# A sentence ends at ./!/? (optionally followed by closing quotes/brackets) and
# whitespace, or at a paragraph break.
_BOUNDARY = re.compile(r"[.!?]+[\"')\]]*\s+|\n\s*\n")


@dataclass
class Chunk:
    chunk_id: str
    doc_name: str
    page_number: int
    text: str
    chunk_index: int   # running index within the document
    start_char: int    # offset of the chunk inside the page text
    end_char: int

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "Chunk":
        return cls(**data)

    @property
    def citation_label(self) -> str:
        return f"{self.doc_name}, p. {self.page_number}"


def sentence_spans(text: str) -> list[tuple[int, int]]:
    """Return ``(start, end)`` character spans of sentences in ``text``."""
    spans: list[tuple[int, int]] = []
    start = 0
    for match in _BOUNDARY.finditer(text):
        end = match.end()
        if text[start:end].strip():
            spans.append((start, end))
        start = end
    if text[start:].strip():
        spans.append((start, len(text)))
    return spans


def split_sentences(text: str) -> list[str]:
    """Convenience wrapper returning stripped sentence strings."""
    return [text[s:e].strip() for s, e in sentence_spans(text)]


def _hard_split(start: int, end: int, chunk_size: int, chunk_overlap: int) -> list[tuple[int, int]]:
    """Split an over-long sentence into fixed windows."""
    step = max(1, chunk_size - chunk_overlap)
    windows = []
    pos = start
    while pos < end:
        win_end = min(pos + chunk_size, end)
        windows.append((pos, win_end))
        if win_end == end:
            break
        pos += step
    return windows


def chunk_text(text: str, chunk_size: int = 800, chunk_overlap: int = 150) -> list[tuple[int, int, str]]:
    """Chunk a single page's text.  Returns ``(start, end, chunk_text)`` tuples.

    Sentences are packed greedily up to ``chunk_size`` characters; the next
    chunk re-includes trailing sentences so that neighbours share at least
    ``chunk_overlap`` characters (where sentence lengths allow).
    """
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if chunk_overlap < 0 or chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must be >= 0 and smaller than chunk_size")
    if not text or not text.strip():
        return []

    units: list[tuple[int, int]] = []
    for s, e in sentence_spans(text):
        if e - s <= chunk_size:
            units.append((s, e))
        else:
            units.extend(_hard_split(s, e, chunk_size, chunk_overlap))

    chunks: list[tuple[int, int, str]] = []
    i, n = 0, len(units)
    while i < n:
        cur_start = units[i][0]
        j = i
        while j + 1 < n and units[j + 1][1] - cur_start <= chunk_size:
            j += 1
        cur_end = units[j][1]
        raw = text[cur_start:cur_end]
        lead = len(raw) - len(raw.lstrip())
        trail = len(raw) - len(raw.rstrip())
        if raw.strip():
            chunks.append((cur_start + lead, cur_end - trail, raw.strip()))
        if j + 1 >= n:
            break
        # Step back over trailing sentences to create the overlap, always making progress.
        k = j + 1
        while k - 1 > i and cur_end - units[k - 1][0] <= chunk_overlap:
            k -= 1
        i = k
    return chunks


def chunk_pages(pages: Iterable[PageText], chunk_size: int = 800, chunk_overlap: int = 150) -> list[Chunk]:
    """Chunk every page of a document, tagging chunks with document name and page number."""
    chunks: list[Chunk] = []
    index = 0
    for page in pages:
        if not page.text or not page.text.strip():
            continue
        for start, end, body in chunk_text(page.text, chunk_size, chunk_overlap):
            chunks.append(
                Chunk(
                    chunk_id=f"{page.doc_name}::p{page.page_number}::c{index}",
                    doc_name=page.doc_name,
                    page_number=page.page_number,
                    text=body,
                    chunk_index=index,
                    start_char=start,
                    end_char=end,
                )
            )
            index += 1
    return chunks
