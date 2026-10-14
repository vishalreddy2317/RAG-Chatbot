"""End-to-end orchestration: ingest PDFs -> chunk -> embed -> FAISS; and
question -> retrieve -> generate -> cite -> grounding check."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Callable, Iterable

from rag.chunking import chunk_pages
from rag.citations import Citation, build_citations
from rag.config import Settings
from rag.embeddings import Embedder, EmbeddingModel, ProgressCallback
from rag.followup import rewrite_for_retrieval
from rag.generators import (
    NO_EVIDENCE_MESSAGE,
    BaseGenerator,
    ExtractiveGenerator,
    GeneratedAnswer,
    build_generator,
)
from rag.grounding import GroundingReport, grounding_score
from rag.pdf_loader import DocumentStatus, PdfSource, load_pdf
from rag.retriever import RetrievalResult, Retriever
from rag.vector_store import FaissVectorStore

NO_DOCUMENTS_MESSAGE = "No documents are loaded yet. Upload a PDF or load the sample documents, then ask a question."
EMPTY_QUESTION_MESSAGE = "Please type a question."


@dataclass
class IngestReport:
    doc_name: str
    status: DocumentStatus
    message: str
    num_pages: int = 0
    num_text_pages: int = 0
    num_chunks: int = 0
    seconds: float = 0.0
    replaced: bool = False

    @property
    def ok(self) -> bool:
        return self.status in (DocumentStatus.OK, DocumentStatus.PARTIAL) and self.num_chunks > 0

    @property
    def warning(self) -> bool:
        return self.status == DocumentStatus.PARTIAL


@dataclass
class AskResult:
    question: str
    retrieval_query: str
    answer: str
    mode: str
    model: str
    citations: list[Citation] = field(default_factory=list)
    retrieval: RetrievalResult | None = None
    grounding: GroundingReport | None = None
    has_evidence: bool = False
    error: str | None = None
    fallback_used: bool = False
    seconds: float = 0.0

    @property
    def rewritten(self) -> bool:
        return self.retrieval_query.strip() != self.question.strip()

    def to_dict(self) -> dict:
        return {
            "question": self.question,
            "retrieval_query": self.retrieval_query,
            "answer": self.answer,
            "mode": self.mode,
            "model": self.model,
            "citations": [c.to_dict() for c in self.citations],
            "has_evidence": self.has_evidence,
            "best_score": self.retrieval.best_score if self.retrieval else 0.0,
            "grounding": self.grounding.label if self.grounding else None,
            "grounding_score": self.grounding.score if self.grounding else None,
            "error": self.error,
            "fallback_used": self.fallback_used,
            "seconds": round(self.seconds, 3),
        }


class RAGPipeline:
    """Owns the embedder, vector store, retriever and generator for one session."""

    def __init__(self, settings: Settings | None = None, embedder: Embedder | None = None,
                 generator: BaseGenerator | None = None, answer_mode: str | None = None):
        self.settings = settings or Settings.from_env()
        self.embedder = embedder or EmbeddingModel(self.settings.embedding_model)
        self.store = FaissVectorStore(self.embedder.dimension)
        self.retriever = Retriever(self.embedder, self.store, self.settings.top_k, self.settings.score_threshold,
                                   self.settings.strong_score)
        self.generator = generator or build_generator(self.settings, self.embedder, answer_mode)
        self.reports: dict[str, IngestReport] = {}

    # ----------------------------------------------------------- documents
    @property
    def document_names(self) -> list[str]:
        return self.store.document_names()

    def document_stats(self) -> dict[str, dict]:
        return self.store.document_stats()

    @property
    def has_documents(self) -> bool:
        return not self.store.is_empty()

    def ingest(self, source: PdfSource, doc_name: str | None = None,
               progress_callback: ProgressCallback | None = None) -> IngestReport:
        """Load, chunk, embed and index one PDF.  Unusable files are reported, not raised."""
        started = time.perf_counter()
        loaded = load_pdf(source, doc_name, min_chars_per_page=self.settings.min_chars_per_page)
        report = IngestReport(loaded.doc_name, loaded.status, loaded.message, loaded.num_pages, len(loaded.text_pages))
        if not loaded.is_usable:
            report.seconds = time.perf_counter() - started
            self.reports[loaded.doc_name] = report
            return report

        chunks = chunk_pages(loaded.text_pages, self.settings.chunk_size, self.settings.chunk_overlap)
        if not chunks:
            report.status = DocumentStatus.EMPTY
            report.message = f"'{loaded.doc_name}' produced no text chunks."
            report.seconds = time.perf_counter() - started
            self.reports[loaded.doc_name] = report
            return report

        if loaded.doc_name in self.document_names:  # re-upload replaces the old copy
            self.store.remove_document(loaded.doc_name)
            report.replaced = True

        embeddings = self.embedder.embed_texts([c.text for c in chunks], progress_callback=progress_callback)
        report.num_chunks = self.store.add(chunks, embeddings)
        report.seconds = time.perf_counter() - started
        self.reports[loaded.doc_name] = report
        return report

    def ingest_many(self, sources: Iterable[tuple[PdfSource, str | None]],
                    progress_callback: Callable[[str, int, int], None] | None = None) -> list[IngestReport]:
        reports = []
        for source, name in sources:
            cb = (lambda done, total, _n=name: progress_callback(_n or "document", done, total)) if progress_callback else None
            reports.append(self.ingest(source, name, cb))
        return reports

    def remove_document(self, doc_name: str) -> bool:
        removed = self.store.remove_document(doc_name) > 0
        self.reports.pop(doc_name, None)
        return removed

    def clear_documents(self) -> None:
        self.store.clear()
        self.reports.clear()

    # ------------------------------------------------------------ answering
    def set_answer_mode(self, mode: str) -> None:
        self.generator = build_generator(self.settings, self.embedder, mode)

    @property
    def answer_mode(self) -> str:
        return self.generator.mode

    def retrieve(self, question: str, history: list[dict] | None = None, top_k: int | None = None,
                 doc_names: Iterable[str] | None = None) -> RetrievalResult:
        query = rewrite_for_retrieval(question, history)
        return self.retriever.retrieve(query, top_k=top_k, doc_names=doc_names)

    def ask(self, question: str, history: list[dict] | None = None, top_k: int | None = None,
            doc_names: Iterable[str] | None = None, score_threshold: float | None = None) -> AskResult:
        started = time.perf_counter()
        question = (question or "").strip()
        mode, model = self.generator.mode, self.generator.describe()

        if not question:
            return AskResult(question, question, EMPTY_QUESTION_MESSAGE, mode, model, seconds=time.perf_counter() - started)
        if not self.has_documents:
            return AskResult(question, question, NO_DOCUMENTS_MESSAGE, mode, model, seconds=time.perf_counter() - started)

        retrieval = self.retrieve(question, history, top_k, doc_names)
        if score_threshold is not None:
            retrieval.threshold = score_threshold

        if not retrieval.has_evidence:
            return AskResult(
                question, retrieval.query, NO_EVIDENCE_MESSAGE, mode, model,
                citations=build_citations(retrieval.results, retrieval.query),
                retrieval=retrieval, grounding=GroundingReport(score=1.0, abstained=True),
                has_evidence=False, seconds=time.perf_counter() - started,
            )

        generated: GeneratedAnswer = self.generator.generate(question, retrieval.results, history)
        fallback_used = False
        error = generated.error
        if not generated.ok:
            # LLM unavailable/failed -> degrade gracefully to the extractive answer.
            fallback = ExtractiveGenerator(self.embedder, self.settings.extractive_max_sentences)
            generated = fallback.generate(question, retrieval.results, history)
            fallback_used = True
            mode, model = generated.mode, generated.model
        else:
            mode, model = generated.mode, generated.model

        citations = build_citations(retrieval.results, retrieval.query)
        grounding = grounding_score(generated.text, [r.chunk.text for r in retrieval.results])
        return AskResult(
            question, retrieval.query, generated.text, mode, model, citations, retrieval, grounding,
            has_evidence=True, error=error, fallback_used=fallback_used, seconds=time.perf_counter() - started,
        )
