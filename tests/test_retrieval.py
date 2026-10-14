import pytest

from rag.chunking import chunk_pages
from rag.retriever import Retriever, corpus_frequent_terms, lexical_overlap
from rag.vector_store import FaissVectorStore


def _build(embedder, pages, threshold=0.05):
    store = FaissVectorStore(embedder.dimension)
    chunks = chunk_pages(pages, chunk_size=200, chunk_overlap=40)
    store.add(chunks, embedder.embed_texts([c.text for c in chunks]))
    return Retriever(embedder, store, top_k=3, score_threshold=threshold)


def test_retriever_finds_expected_page_with_hashing_embedder(hashing_embedder, pages):
    retriever = _build(hashing_embedder, pages)
    result = retriever.retrieve("When is planned maintenance scheduled?")
    assert result.results
    assert result.results[0].chunk.page_number == 3
    assert result.has_evidence


def test_empty_query_and_empty_store(hashing_embedder, pages):
    retriever = _build(hashing_embedder, pages)
    assert retriever.retrieve("   ").results == []
    assert not retriever.retrieve("").has_evidence
    empty = Retriever(hashing_embedder, FaissVectorStore(hashing_embedder.dimension))
    assert empty.retrieve("anything").results == []


def test_threshold_blocks_weak_matches(hashing_embedder, pages):
    retriever = _build(hashing_embedder, pages, threshold=0.99)
    result = retriever.retrieve("planned maintenance")
    assert result.results  # something is always nearest...
    assert not result.has_evidence  # ...but it does not count as evidence


def test_lexical_overlap_helpers():
    assert lexical_overlap("overdue fines", ["The library charges no fine."])
    assert not lexical_overlap("pets allowed", ["The library charges no fine."])
    assert lexical_overlap("the of and", ["anything"])  # no content words -> nothing to check
    assert not lexical_overlap("library pets", ["library rules"], ignore_terms={"library"})
    frequent = corpus_frequent_terms(["library books", "library hours", "library cards", "music"], max_fraction=0.5)
    assert "library" in frequent and "music" not in frequent


@pytest.mark.slow
def test_real_model_retrieves_correct_pages(real_embedder, sample_pdfs):
    from rag.pdf_loader import load_pdf

    store = FaissVectorStore(real_embedder.dimension)
    for path in sample_pdfs:
        doc = load_pdf(path)
        chunks = chunk_pages(doc.text_pages)
        store.add(chunks, real_embedder.embed_texts([c.text for c in chunks]))
    retriever = Retriever(real_embedder, store, top_k=3, score_threshold=0.30)

    cases = [
        ("What is the community solar credit rate for 2024?", "orion_solar_handbook.pdf", 3),
        ("How much does a non-resident library card cost?", "lumen_library_policy.pdf", 1),
        ("How much does 3D printing cost?", "lumen_library_policy.pdf", 5),
    ]
    for question, doc, page in cases:
        result = retriever.retrieve(question)
        assert result.has_evidence, question
        top = result.results[0].chunk
        assert (top.doc_name, top.page_number) == (doc, page), question

    off_topic = retriever.retrieve("What is the capital of France?")
    assert not off_topic.has_evidence
