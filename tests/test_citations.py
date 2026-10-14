from rag.chunking import Chunk
from rag.citations import Citation, best_excerpt, build_citations, highlight_html, query_terms, unique_labels
from rag.vector_store import SearchResult


def _result(text, doc="guide.pdf", page=4, score=0.7, rank=1):
    return SearchResult(Chunk(f"{doc}::p{page}::c{rank}", doc, page, text, rank, 0, len(text)), score, rank)


def test_query_terms_drops_stopwords_and_duplicates():
    assert query_terms("What is the credit rate for the credit?") == ["credit", "rate"]
    assert query_terms("the of and") == []


def test_citation_label_and_dict_round_trip():
    c = Citation("guide.pdf", 4, "excerpt", 0.5, "guide.pdf::p4::c1", 1)
    assert c.label == "guide.pdf, p. 4"
    data = c.to_dict()
    assert data["label"] == "guide.pdf, p. 4"
    assert Citation.from_dict(data) == c


def test_build_citations_keeps_doc_page_and_rank():
    results = [_result("Alpha text about fees.", page=2, rank=1), _result("Beta text about rates.", page=7, score=0.4, rank=2)]
    citations = build_citations(results, "fees")
    assert [(c.doc_name, c.page_number, c.rank) for c in citations] == [("guide.pdf", 2, 1), ("guide.pdf", 7, 2)]
    assert citations[0].score == 0.7
    assert unique_labels(citations) == ["guide.pdf, p. 2", "guide.pdf, p. 7"]


def test_best_excerpt_focuses_on_query_terms_and_respects_length():
    filler = " ".join(f"Filler sentence number {i} says nothing useful." for i in range(20))
    text = filler + " The enrollment fee is zero dollars for everyone. " + filler
    excerpt = best_excerpt(text, "What is the enrollment fee?", max_chars=120)
    assert "enrollment fee" in excerpt
    assert len(excerpt) <= 130  # ellipsis markers allowed
    assert excerpt.startswith("…") and excerpt.endswith("…")


def test_best_excerpt_returns_short_text_unchanged():
    assert best_excerpt("Short text.", "anything", max_chars=100) == "Short text."


def test_highlight_html_escapes_and_marks_terms():
    out = highlight_html("Fees <b>apply</b> to credit cards & more", "credit fees")
    assert "<b>" not in out and "&lt;b&gt;" in out
    assert "<mark>Fees</mark>" in out and "<mark>credit</mark>" in out
    assert "&amp;" in out


def test_highlight_without_terms_only_escapes():
    assert highlight_html("a < b", "the") == "a &lt; b"
