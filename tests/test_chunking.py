import pytest

from rag.chunking import Chunk, chunk_pages, chunk_text, sentence_spans, split_sentences
from rag.pdf_loader import PageText


def test_sentence_spans_cover_text_in_order():
    text = "First sentence. Second one! Third? Fourth without end"
    spans = sentence_spans(text)
    assert [text[s:e].strip() for s, e in spans] == ["First sentence.", "Second one!", "Third?", "Fourth without end"]
    assert spans[0][0] == 0 and spans[-1][1] == len(text)


def test_split_sentences_handles_paragraph_breaks():
    assert split_sentences("Alpha beta\n\nGamma delta.") == ["Alpha beta", "Gamma delta."]


def test_chunks_respect_size_and_overlap():
    text = " ".join(f"Sentence number {i} is here." for i in range(60))
    chunks = chunk_text(text, chunk_size=120, chunk_overlap=40)
    assert len(chunks) > 1
    for start, end, body in chunks:
        assert len(body) <= 120
        assert text[start:end] == body  # offsets map back to the source text
    for (s1, e1, _), (s2, e2, _) in zip(chunks, chunks[1:]):
        assert s2 > s1, "chunks must advance"
        assert s2 < e1, "consecutive chunks must overlap"
        assert e1 - s2 >= 25, "overlap should be at least one trailing sentence"


def test_chunks_cover_entire_text():
    text = " ".join(f"Token{i} words words." for i in range(100))
    chunks = chunk_text(text, chunk_size=200, chunk_overlap=50)
    covered = set()
    for s, e, _ in chunks:
        covered.update(range(s, e))
    missing = [i for i, ch in enumerate(text) if not ch.isspace() and i not in covered]
    assert not missing


def test_overlong_sentence_is_hard_split():
    text = "x" * 1000  # no sentence boundaries at all
    chunks = chunk_text(text, chunk_size=300, chunk_overlap=50)
    assert all(len(c[2]) <= 300 for c in chunks)
    assert chunks[0][0] == 0 and chunks[-1][1] == 1000


def test_short_text_is_single_chunk():
    assert chunk_text("Just one short sentence.", 800, 150) == [(0, 24, "Just one short sentence.")]


def test_empty_text_returns_nothing():
    assert chunk_text("", 800, 150) == []
    assert chunk_text("   \n  ", 800, 150) == []


@pytest.mark.parametrize("size,overlap", [(0, 0), (100, 100), (100, 150), (100, -1)])
def test_invalid_parameters_raise(size, overlap):
    with pytest.raises(ValueError):
        chunk_text("some text here.", size, overlap)


def test_chunk_pages_preserves_doc_name_and_page_numbers(pages):
    chunks = chunk_pages(pages, chunk_size=120, chunk_overlap=30)
    assert chunks, "expected chunks"
    assert {c.doc_name for c in chunks} == {"handbook.pdf"}
    assert {c.page_number for c in chunks} == {1, 2, 3}
    assert [c.chunk_index for c in chunks] == list(range(len(chunks)))
    assert len({c.chunk_id for c in chunks}) == len(chunks)
    for c in chunks:
        page = pages[c.page_number - 1]
        assert page.text[c.start_char:c.end_char] == c.text
        assert c.citation_label == f"handbook.pdf, p. {c.page_number}"


def test_chunk_pages_never_crosses_page_boundary(pages):
    chunks = chunk_pages(pages, chunk_size=5000, chunk_overlap=100)
    assert len(chunks) == 3  # one chunk per page, even though all pages fit in one chunk by size


def test_chunk_pages_skips_empty_pages():
    pages = [PageText("d.pdf", 1, "Real text on page one."), PageText("d.pdf", 2, "   "), PageText("d.pdf", 3, "")]
    chunks = chunk_pages(pages)
    assert [c.page_number for c in chunks] == [1]


def test_chunk_round_trips_through_dict():
    chunk = Chunk("d.pdf::p1::c0", "d.pdf", 1, "text", 0, 0, 4)
    assert Chunk.from_dict(chunk.to_dict()) == chunk
