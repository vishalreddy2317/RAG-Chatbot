from pathlib import Path

from rag.pdf_loader import DocumentStatus, clean_text, is_pdf_bytes, load_pdf, remove_repeated_lines, PageText


def test_loads_text_pdf_with_page_numbers(text_pdf: Path):
    doc = load_pdf(text_pdf)
    assert doc.status == DocumentStatus.OK
    assert doc.is_usable
    assert doc.num_pages == 3
    assert [p.page_number for p in doc.pages] == [1, 2, 3]
    assert all(p.doc_name == "handbook.pdf" for p in doc.pages)
    assert "11.7 cents" in doc.pages[1].text
    assert "April and October" in doc.pages[2].text


def test_accepts_bytes_and_custom_name(text_pdf: Path):
    doc = load_pdf(text_pdf.read_bytes(), doc_name="custom.pdf")
    assert doc.status == DocumentStatus.OK
    assert doc.doc_name == "custom.pdf"
    assert doc.pages[0].doc_name == "custom.pdf"


def test_empty_bytes_reported_as_empty():
    doc = load_pdf(b"", doc_name="nothing.pdf")
    assert doc.status == DocumentStatus.EMPTY
    assert not doc.is_usable
    assert "0 bytes" in doc.message


def test_non_pdf_bytes_reported_as_invalid():
    doc = load_pdf(b"hello, this is a text file", doc_name="notes.txt")
    assert doc.status == DocumentStatus.INVALID
    assert "not a PDF" in doc.message
    assert not doc.is_usable


def test_corrupt_pdf_reported_as_invalid():
    doc = load_pdf(b"%PDF-1.7 garbage garbage garbage", doc_name="broken.pdf")
    assert doc.status in (DocumentStatus.INVALID, DocumentStatus.EMPTY)
    assert not doc.is_usable


def test_scanned_pdf_detected(scanned_pdf: Path):
    doc = load_pdf(scanned_pdf)
    assert doc.status == DocumentStatus.SCANNED
    assert "OCR" in doc.message
    assert not doc.is_usable


def test_blank_pdf_reported_as_empty(blank_pdf: Path):
    doc = load_pdf(blank_pdf)
    assert doc.status == DocumentStatus.EMPTY
    assert doc.num_pages == 2
    assert not doc.is_usable


def test_encrypted_pdf_detected(encrypted_pdf: Path):
    doc = load_pdf(encrypted_pdf)
    assert doc.status == DocumentStatus.ENCRYPTED
    assert "password" in doc.message.lower()


def test_partial_pdf_when_some_pages_blank(tmp_path: Path):
    from tests.conftest import make_text_pdf

    path = make_text_pdf(tmp_path / "partial.pdf", pages=["First page has real text in it.", "", "Third page has some real text in it too."])
    doc = load_pdf(path)
    assert doc.status == DocumentStatus.PARTIAL
    assert doc.is_usable
    assert len(doc.text_pages) == 2
    assert "1 of 3" in doc.message


def test_running_headers_removed(header_pdf: Path):
    doc = load_pdf(header_pdf)
    assert doc.status == DocumentStatus.OK
    for page in doc.pages:
        assert "Confidential" not in page.text
        assert not page.text.startswith("Page ")


def test_is_pdf_bytes():
    assert is_pdf_bytes(b"%PDF-1.4\n...")
    assert is_pdf_bytes(b"\xef\xbb\xbf   %PDF-1.7")
    assert not is_pdf_bytes(b"PK\x03\x04 zip file")


def test_clean_text_normalises_whitespace_and_hyphenation():
    raw = "This docu-\nment has   extra   spaces.\n\n\n\nAnd   many\r\nnewlines.  "
    cleaned = clean_text(raw)
    assert "document" in cleaned
    assert "   " not in cleaned
    assert "\n\n\n" not in cleaned
    assert cleaned.endswith("newlines.")


def test_remove_repeated_lines_ignores_short_documents():
    pages = [PageText("d.pdf", 1, "Header\nbody one"), PageText("d.pdf", 2, "Header\nbody two")]
    assert remove_repeated_lines(pages) == 0
    assert pages[0].text.startswith("Header")
