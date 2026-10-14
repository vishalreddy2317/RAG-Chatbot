"""PDF text extraction with page-level metadata and friendly failure modes.

Uses PyMuPDF (``fitz``).  Every page keeps its 1-based page number so that
citations can point users back to the exact page.  The loader never raises for
bad input; it returns a :class:`LoadedDocument` whose ``status`` and
``message`` explain what went wrong (scanned PDF, empty file, encrypted file,
not a PDF, ...).
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Union

try:  # PyMuPDF >= 1.24 exposes the `pymupdf` name; older versions only have `fitz`.
    import pymupdf
except ImportError:  # pragma: no cover
    import fitz as pymupdf

PdfSource = Union[bytes, bytearray, str, Path]


class DocumentStatus(str, Enum):
    OK = "ok"                # text extracted from every page
    PARTIAL = "partial"      # usable, but some pages had no text layer
    SCANNED = "scanned"      # image-only pages, no text layer -> needs OCR
    EMPTY = "empty"          # zero pages / zero bytes / no text at all
    ENCRYPTED = "encrypted"  # password protected
    INVALID = "invalid"      # not a PDF or corrupt


@dataclass
class PageText:
    doc_name: str
    page_number: int  # 1-based, as printed in PDF viewers
    text: str

    @property
    def char_count(self) -> int:
        return len(self.text)


@dataclass
class LoadedDocument:
    doc_name: str
    pages: list[PageText] = field(default_factory=list)
    num_pages: int = 0
    status: DocumentStatus = DocumentStatus.EMPTY
    message: str = ""

    @property
    def text_pages(self) -> list[PageText]:
        return [p for p in self.pages if p.text.strip()]

    @property
    def total_chars(self) -> int:
        return sum(p.char_count for p in self.pages)

    @property
    def is_usable(self) -> bool:
        return self.status in (DocumentStatus.OK, DocumentStatus.PARTIAL) and bool(self.text_pages)


def is_pdf_bytes(data: bytes) -> bool:
    """True when the data starts with a PDF header (allowed within the first 1 KB)."""
    return b"%PDF" in data[:1024]


_HYPHEN_BREAK = re.compile(r"(\w)-\n(?=[a-z])")
_SPACES_BEFORE_NEWLINE = re.compile(r"[ \t]+\n")
_MANY_NEWLINES = re.compile(r"\n{3,}")
_MANY_SPACES = re.compile(r"[ \t]{2,}")


def clean_text(text: str) -> str:
    """Normalise extracted text: fix hyphenated line breaks, collapse whitespace,
    drop control characters.  Paragraph breaks (blank lines) are preserved."""
    if not text:
        return ""
    text = text.replace("\r\n", "\n").replace("\r", "\n").replace("\x00", "")
    text = _HYPHEN_BREAK.sub(r"\1", text)
    text = _SPACES_BEFORE_NEWLINE.sub("\n", text)
    text = _MANY_NEWLINES.sub("\n\n", text)
    text = _MANY_SPACES.sub(" ", text)
    text = "".join(ch for ch in text if ch.isprintable() or ch in "\n\t")
    return text.strip()


_DIGITS = re.compile(r"\d+")


def remove_repeated_lines(pages: list["PageText"], min_pages: int = 3, min_fraction: float = 0.5) -> int:
    """Strip running headers/footers: short first/last lines that repeat on most pages.

    Digits are ignored when comparing lines so "Page 3 of 8" and "Page 4 of 8" count as the
    same footer.  Returns the number of lines removed.
    """
    if len(pages) < min_pages:
        return 0
    counts: dict[str, int] = {}
    for page in pages:
        lines = [ln.strip() for ln in page.text.split("\n") if ln.strip()]
        candidates = {_DIGITS.sub("#", ln) for ln in (lines[:2] + lines[-2:]) if len(ln) <= 120}
        for key in candidates:
            counts[key] = counts.get(key, 0) + 1
    threshold = max(min_pages, int(len(pages) * min_fraction + 0.999))
    repeated = {key for key, n in counts.items() if n >= threshold}
    if not repeated:
        return 0
    removed = 0
    for page in pages:
        lines = page.text.split("\n")
        kept = []
        for i, ln in enumerate(lines):
            edge = i < 2 or i >= len(lines) - 2
            if edge and ln.strip() and _DIGITS.sub("#", ln.strip()) in repeated:
                removed += 1
                continue
            kept.append(ln)
        page.text = clean_text("\n".join(kept))
    return removed


def _resolve_source(source: PdfSource, doc_name: str | None) -> tuple[bytes, str]:
    if isinstance(source, (str, Path)):
        path = Path(source)
        return path.read_bytes(), doc_name or path.name
    return bytes(source), doc_name or "document.pdf"


def load_pdf(source: PdfSource, doc_name: str | None = None, *, min_chars_per_page: int = 25) -> LoadedDocument:
    """Extract page texts from a PDF given as bytes or a file path.

    Never raises for bad input: inspect ``result.status`` / ``result.message``.
    """
    data, name = _resolve_source(source, doc_name)

    if not data:
        return LoadedDocument(name, status=DocumentStatus.EMPTY, message=f"'{name}' is empty (0 bytes).")
    if not is_pdf_bytes(data):
        return LoadedDocument(
            name,
            status=DocumentStatus.INVALID,
            message=f"'{name}' is not a PDF file. Only PDF documents are supported.",
        )

    try:
        doc = pymupdf.open(stream=data, filetype="pdf")
    except Exception as exc:  # PyMuPDF raises several exception types for corrupt files
        return LoadedDocument(
            name,
            status=DocumentStatus.INVALID,
            message=f"'{name}' could not be opened as a PDF (it may be corrupt): {exc}",
        )

    with doc:
        if doc.needs_pass:
            return LoadedDocument(
                name,
                num_pages=doc.page_count,
                status=DocumentStatus.ENCRYPTED,
                message=f"'{name}' is password protected. Remove the password and upload it again.",
            )
        if doc.page_count == 0:
            return LoadedDocument(name, status=DocumentStatus.EMPTY, message=f"'{name}' contains no pages.")

        pages: list[PageText] = []
        image_only_pages = 0
        for index, page in enumerate(doc):
            try:
                text = clean_text(page.get_text("text"))
            except Exception:  # pragma: no cover - extremely rare per-page failure
                text = ""
            if len(text) < min_chars_per_page:  # too little text to be useful: treat as no text layer
                if page.get_images(full=False):
                    image_only_pages += 1
                text = ""
            pages.append(PageText(doc_name=name, page_number=index + 1, text=text))

    remove_repeated_lines(pages)
    text_pages = [p for p in pages if len(p.text) >= min_chars_per_page]
    total = len(pages)

    if not text_pages:
        if image_only_pages:
            return LoadedDocument(
                name, pages, total, DocumentStatus.SCANNED,
                f"'{name}' looks like a scanned PDF ({image_only_pages} of {total} pages are images with no "
                "text layer). Run OCR first (for example with `ocrmypdf input.pdf output.pdf`) and upload the result.",
            )
        return LoadedDocument(
            name, pages, total, DocumentStatus.EMPTY,
            f"'{name}' has {total} page(s) but no extractable text.",
        )

    if len(text_pages) < total:
        missing = total - len(text_pages)
        return LoadedDocument(
            name, pages, total, DocumentStatus.PARTIAL,
            f"'{name}': {missing} of {total} pages had no extractable text (possibly scanned images) and were skipped.",
        )

    return LoadedDocument(name, pages, total, DocumentStatus.OK, f"Extracted text from {total} page(s).")
