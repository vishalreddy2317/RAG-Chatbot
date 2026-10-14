"""Shared fixtures.  Most tests use the offline HashingEmbedder; the few that need the
real Sentence Transformers model are marked ``slow`` and share one cached model."""

from __future__ import annotations

import os
from pathlib import Path

import pymupdf
import pytest

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

from rag.config import SAMPLE_DATA_DIR, Settings  # noqa: E402
from rag.embeddings import EmbeddingModel, HashingEmbedder  # noqa: E402
from rag.pdf_loader import PageText  # noqa: E402

PAGE_TEXTS = [
    "Enrollment. There is no enrollment fee and no credit check. Applicants complete the online form. "
    "Cancellations require 30 days' written notice and take effect at the end of the billing cycle.",
    "Billing. Members receive a bill credit at the community solar credit rate of 11.7 cents per kilowatt-hour. "
    "Subscription payments are billed on the 5th of each month. Unused credits roll over to the following month.",
    "Maintenance. Planned maintenance is scheduled twice a year, in April and October. "
    "Unplanned outages lasting more than 24 hours are reported to members by email within 48 hours.",
]


def make_text_pdf(path: Path, pages: list[str] = PAGE_TEXTS, header: str | None = None) -> Path:
    doc = pymupdf.open()
    for i, text in enumerate(pages, start=1):
        page = doc.new_page(width=612, height=792)
        if header:
            page.insert_text((72, 50), header, fontsize=9)
        page.insert_textbox(pymupdf.Rect(72, 90, 540, 720), text, fontsize=11, fontname="helv")
        if header:
            page.insert_text((72, 760), f"Page {i}", fontsize=9)
    doc.save(path)
    doc.close()
    return path


def make_image_only_pdf(path: Path, pages: int = 2) -> Path:
    """A 'scanned' PDF: pages contain only a raster image, no text layer."""
    pix = pymupdf.Pixmap(pymupdf.csRGB, pymupdf.IRect(0, 0, 200, 100), False)
    pix.clear_with(200)
    png = pix.tobytes("png")
    doc = pymupdf.open()
    for _ in range(pages):
        page = doc.new_page(width=612, height=792)
        page.insert_image(pymupdf.Rect(72, 72, 540, 400), stream=png)
    doc.save(path)
    doc.close()
    return path


def make_blank_pdf(path: Path, pages: int = 2) -> Path:
    doc = pymupdf.open()
    for _ in range(pages):
        doc.new_page()
    doc.save(path)
    doc.close()
    return path


def make_encrypted_pdf(path: Path) -> Path:
    doc = pymupdf.open()
    page = doc.new_page()
    page.insert_text((72, 72), "Secret text that needs a password.")
    doc.save(path, encryption=pymupdf.PDF_ENCRYPT_AES_256, user_pw="secret", owner_pw="owner")
    doc.close()
    return path


@pytest.fixture
def text_pdf(tmp_path: Path) -> Path:
    return make_text_pdf(tmp_path / "handbook.pdf")


@pytest.fixture
def header_pdf(tmp_path: Path) -> Path:
    return make_text_pdf(tmp_path / "with_header.pdf", header="Orion Handbook - Confidential")


@pytest.fixture
def scanned_pdf(tmp_path: Path) -> Path:
    return make_image_only_pdf(tmp_path / "scanned.pdf")


@pytest.fixture
def blank_pdf(tmp_path: Path) -> Path:
    return make_blank_pdf(tmp_path / "blank.pdf")


@pytest.fixture
def encrypted_pdf(tmp_path: Path) -> Path:
    return make_encrypted_pdf(tmp_path / "locked.pdf")


@pytest.fixture
def pages() -> list[PageText]:
    return [PageText("handbook.pdf", i, text) for i, text in enumerate(PAGE_TEXTS, start=1)]


@pytest.fixture
def hashing_embedder() -> HashingEmbedder:
    return HashingEmbedder(dimension=256)


@pytest.fixture
def fast_settings() -> Settings:
    return Settings(chunk_size=300, chunk_overlap=60, top_k=3, score_threshold=0.05, strong_score=0.5,
                    answer_mode="extractive")


@pytest.fixture(scope="session")
def real_embedder() -> EmbeddingModel:
    embedder = EmbeddingModel()
    embedder.embed_query("warm up")
    return embedder


@pytest.fixture
def sample_pdfs() -> list[Path]:
    paths = [SAMPLE_DATA_DIR / "orion_solar_handbook.pdf", SAMPLE_DATA_DIR / "lumen_library_policy.pdf"]
    missing = [p for p in paths if not p.exists()]
    if missing:
        pytest.skip(f"sample PDFs missing: {missing}; run sample_data/make_sample_pdfs.py")
    return paths
