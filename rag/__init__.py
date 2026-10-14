"""AI-Powered Document Assistant using Retrieval-Augmented Generation (RAG).

Public API
----------
- :class:`rag.config.Settings` – configuration loaded from environment variables
- :class:`rag.pipeline.RAGPipeline` – ingest PDFs and answer questions
- :mod:`rag.pdf_loader`, :mod:`rag.chunking`, :mod:`rag.embeddings`,
  :mod:`rag.vector_store`, :mod:`rag.retriever`, :mod:`rag.generators`,
  :mod:`rag.citations`, :mod:`rag.grounding` – individual building blocks
"""

from rag.config import Settings
from rag.pipeline import AskResult, IngestReport, RAGPipeline

__all__ = ["Settings", "RAGPipeline", "AskResult", "IngestReport"]
__version__ = "1.0.0"
