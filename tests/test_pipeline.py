from pathlib import Path

from rag.followup import is_followup, rewrite_for_retrieval
from rag.generators import NO_EVIDENCE_MESSAGE
from rag.pdf_loader import DocumentStatus
from rag.pipeline import EMPTY_QUESTION_MESSAGE, NO_DOCUMENTS_MESSAGE, RAGPipeline


def _pipeline(fast_settings, hashing_embedder):
    return RAGPipeline(fast_settings, embedder=hashing_embedder)


def test_ask_without_documents_or_question(fast_settings, hashing_embedder):
    p = _pipeline(fast_settings, hashing_embedder)
    assert p.ask("What is the fee?").answer == NO_DOCUMENTS_MESSAGE
    assert p.ask("   ").answer == EMPTY_QUESTION_MESSAGE


def test_ingest_text_pdf_and_answer_with_citation(fast_settings, hashing_embedder, text_pdf: Path):
    p = _pipeline(fast_settings, hashing_embedder)
    progress = []
    report = p.ingest(text_pdf, progress_callback=lambda d, t: progress.append((d, t)))
    assert report.ok and report.status == DocumentStatus.OK
    assert report.num_pages == 3 and report.num_chunks >= 3
    assert progress and progress[-1][0] == progress[-1][1]
    assert p.document_names() == ["handbook.pdf"] if callable(p.document_names) else p.document_names == ["handbook.pdf"]

    result = p.ask("When is planned maintenance scheduled?")
    assert result.has_evidence
    assert "April and October" in result.answer
    assert result.citations and result.citations[0].doc_name == "handbook.pdf"
    assert {c.page_number for c in result.citations} >= {3}
    assert result.grounding is not None and result.grounding.score == 1.0


def test_unusable_files_are_reported_not_indexed(fast_settings, hashing_embedder, scanned_pdf, blank_pdf, encrypted_pdf):
    p = _pipeline(fast_settings, hashing_embedder)
    assert p.ingest(scanned_pdf).status == DocumentStatus.SCANNED
    assert p.ingest(blank_pdf).status == DocumentStatus.EMPTY
    assert p.ingest(encrypted_pdf).status == DocumentStatus.ENCRYPTED
    assert p.ingest(b"plain text", "notes.txt").status == DocumentStatus.INVALID
    assert p.ingest(b"", "empty.pdf").status == DocumentStatus.EMPTY
    assert not p.has_documents
    assert len(p.reports) == 5


def test_no_evidence_message_for_off_topic_question(fast_settings, hashing_embedder, text_pdf):
    p = _pipeline(fast_settings, hashing_embedder)
    p.ingest(text_pdf)
    result = p.ask("Who won the football match yesterday?")
    assert not result.has_evidence
    assert result.answer == NO_EVIDENCE_MESSAGE
    assert result.grounding.abstained


def test_followup_rewrite_uses_previous_question(fast_settings, hashing_embedder, text_pdf):
    p = _pipeline(fast_settings, hashing_embedder)
    p.ingest(text_pdf)
    history = [{"role": "user", "content": "When is planned maintenance scheduled?"},
               {"role": "assistant", "content": "April and October."}]
    result = p.ask("How long does it take?", history=history)
    assert result.rewritten
    assert "planned maintenance" in result.retrieval_query
    assert is_followup("How long does it take?")
    assert not is_followup("What is the community solar credit rate for 2024?")
    assert rewrite_for_retrieval("Fresh standalone question about billing credits", history) == "Fresh standalone question about billing credits"


def test_reupload_replaces_and_remove_clears(fast_settings, hashing_embedder, text_pdf):
    p = _pipeline(fast_settings, hashing_embedder)
    first = p.ingest(text_pdf)
    second = p.ingest(text_pdf)
    assert second.replaced and len(p.store) == first.num_chunks
    assert p.remove_document("handbook.pdf")
    assert not p.has_documents and not p.remove_document("handbook.pdf")
    p.ingest(text_pdf)
    p.clear_documents()
    assert not p.has_documents and p.reports == {}


def test_ask_result_serialises(fast_settings, hashing_embedder, text_pdf):
    p = _pipeline(fast_settings, hashing_embedder)
    p.ingest(text_pdf)
    data = p.ask("What is the credit rate?").to_dict()
    assert set(data) >= {"question", "answer", "citations", "has_evidence", "grounding_score", "seconds"}
    assert data["citations"][0]["label"].startswith("handbook.pdf, p. ")
