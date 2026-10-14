"""Streamlit UI for the AI-Powered Document Assistant (RAG).

Run with:  streamlit run app.py
"""

from __future__ import annotations

import hashlib
import html
import json
import os
import time
from pathlib import Path

import streamlit as st

from rag.citations import Citation, highlight_html
from rag.config import SAMPLE_DATA_DIR, Settings
from rag.embeddings import EmbeddingModel
from rag.generators import describe_modes
from rag.pdf_loader import DocumentStatus
from rag.pipeline import AskResult, IngestReport, RAGPipeline

# --------------------------------------------------------------------------- page setup
st.set_page_config(
    page_title="AI Document Assistant",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={"About": "Ask questions about your PDFs and get answers with page-level citations (RAG)."},
)

CSS = """
<style>
.block-container { padding-top: 1.6rem; }
.cite { border-left: 4px solid #2563EB; background: rgba(37, 99, 235, 0.07); padding: .6rem .85rem;
        border-radius: 8px; margin-bottom: .55rem; }
.cite-head { display: flex; justify-content: space-between; gap: .5rem; font-weight: 600;
             font-size: .9rem; margin-bottom: .3rem; }
.cite-body { font-size: .9rem; line-height: 1.5; }
.cite-body mark { background: #FDE68A; color: #1F2937; padding: 0 2px; border-radius: 3px; }
.score { font-family: ui-monospace, SFMono-Regular, Menlo, monospace; font-weight: 500; opacity: .75;
         white-space: nowrap; }
.badge { display: inline-block; padding: .15rem .55rem; border-radius: 999px; font-size: .78rem;
         font-weight: 600; background: rgba(37, 99, 235, .12); color: #1D4ED8; margin-right: .35rem; }
.badge.ok { background: rgba(22, 163, 74, .14); color: #15803D; }
.badge.warn { background: rgba(217, 119, 6, .16); color: #B45309; }
.doc-row { font-size: .88rem; line-height: 1.35; }
.hero { padding: 1.2rem 1.4rem; border-radius: 12px; background: rgba(37, 99, 235, .06);
        border: 1px solid rgba(37, 99, 235, .15); margin-bottom: 1rem; }
.hero h3 { margin: 0 0 .4rem 0; }
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)

SECRET_KEYS = (
    "ANSWER_MODE", "HOSTED_PROVIDER", "OPENAI_API_KEY", "OPENAI_BASE_URL", "OPENAI_MODEL",
    "ANTHROPIC_API_KEY", "ANTHROPIC_MODEL", "OLLAMA_BASE_URL", "OLLAMA_MODEL", "EMBEDDING_MODEL",
    "TOP_K", "SCORE_THRESHOLD", "STRONG_SCORE", "CHUNK_SIZE", "CHUNK_OVERLAP",
)

STATUS_ICON = {
    DocumentStatus.OK: "✅", DocumentStatus.PARTIAL: "⚠️", DocumentStatus.SCANNED: "🖼️",
    DocumentStatus.EMPTY: "📭", DocumentStatus.ENCRYPTED: "🔒", DocumentStatus.INVALID: "🚫",
}


# --------------------------------------------------------------------------- helpers
def load_secrets_into_env() -> None:
    """Streamlit Cloud stores config in st.secrets; expose it as env vars for rag.config."""
    try:
        secrets = st.secrets
        for key in SECRET_KEYS:
            if key in secrets and key not in os.environ and str(secrets[key]).strip():
                os.environ[key] = str(secrets[key])
    except Exception:  # no secrets.toml present - perfectly fine
        pass


@st.cache_resource(show_spinner="Loading embedding model (first run downloads ~90 MB)…")
def get_embedder(model_name: str) -> EmbeddingModel:
    embedder = EmbeddingModel(model_name)
    embedder.embed_query("warm up")  # trigger the download/load once, inside the cache
    return embedder


@st.cache_data(show_spinner=False)
def load_sample_files() -> list[tuple[str, bytes]]:
    manifest = json.loads((SAMPLE_DATA_DIR / "sample_questions.json").read_text(encoding="utf-8"))
    files = []
    for name in manifest["documents"]:
        path = SAMPLE_DATA_DIR / name
        if path.exists():
            files.append((name, path.read_bytes()))
    return files


@st.cache_data(show_spinner=False)
def load_sample_questions() -> list[str]:
    manifest = json.loads((SAMPLE_DATA_DIR / "sample_questions.json").read_text(encoding="utf-8"))
    return list(manifest["questions"])


@st.cache_data(ttl=60, show_spinner=False)
def cached_modes(ollama_url: str, ollama_model: str, provider: str, has_key: bool, hosted_model: str, emb: str):
    settings = Settings(ollama_base_url=ollama_url, ollama_model=ollama_model, hosted_provider=provider,
                        openai_api_key="x" if (has_key and provider == "openai") else "",
                        anthropic_api_key="x" if (has_key and provider == "anthropic") else "",
                        openai_model=hosted_model, anthropic_model=hosted_model, embedding_model=emb)
    return [(m.mode, m.available, m.detail) for m in describe_modes(settings)]


def get_settings() -> Settings:
    if "settings" not in st.session_state:
        load_secrets_into_env()
        st.session_state.settings = Settings.from_env()
    return st.session_state.settings


def get_pipeline() -> RAGPipeline:
    if "pipeline" not in st.session_state:
        settings = get_settings()
        embedder = get_embedder(settings.embedding_model)
        try:
            st.session_state.pipeline = RAGPipeline(settings, embedder=embedder)
        except ValueError as exc:  # e.g. ANSWER_MODE=hosted without a key
            st.session_state.setdefault("notices", []).append(("warning", f"{exc} Falling back to extractive mode."))
            st.session_state.pipeline = RAGPipeline(settings, embedder=embedder, answer_mode="extractive")
    return st.session_state.pipeline


def init_state() -> None:
    settings = get_settings()
    defaults = {
        "messages": [],          # chat history: dicts with role/content (+ answer metadata)
        "processed": {},         # upload fingerprint -> doc name
        "notices": [],           # (level, text) shown in the sidebar
        "pending_question": None,
        "sample_loaded": False,
        "top_k": settings.top_k,
        "threshold": settings.score_threshold,
        "show_scores": True,
    }
    for key, value in defaults.items():
        st.session_state.setdefault(key, value)


def fingerprint(name: str, data: bytes) -> str:
    return f"{name}:{len(data)}:{hashlib.md5(data).hexdigest()}"


def notice_for(report: IngestReport) -> tuple[str, str]:
    icon = STATUS_ICON.get(report.status, "ℹ️")
    if report.ok and not report.warning:
        extra = " · replaced previous copy" if report.replaced else ""
        return "success", f"{icon} **{report.doc_name}** · {report.num_pages} pages · {report.num_chunks} chunks{extra}"
    if report.ok and report.warning:
        return "warning", f"{icon} {report.message} Indexed {report.num_chunks} chunks from {report.num_text_pages} pages."
    return "error", f"{icon} {report.message}"


def ingest_files(files: list[tuple[str, bytes]], container) -> None:
    pipeline = get_pipeline()
    with container.status(f"Processing {len(files)} file(s)…", expanded=True) as status:
        for name, data in files:
            bar = st.progress(0.0, text=f"Reading **{name}**")

            def on_progress(done: int, total: int, _bar=bar, _name=name) -> None:
                _bar.progress(done / max(total, 1), text=f"Embedding **{_name}**: {done}/{total} chunks")

            report = pipeline.ingest(data, name, progress_callback=on_progress)
            bar.progress(1.0, text=f"Finished **{name}**")
            st.session_state.notices.append(notice_for(report))
            st.session_state.processed[fingerprint(name, data)] = name
        status.update(label=f"Processed {len(files)} file(s)", state="complete", expanded=False)


def answer_to_message(result: AskResult) -> dict:
    return {
        "role": "assistant",
        "content": result.answer,
        "citations": [c.to_dict() for c in result.citations],
        "has_evidence": result.has_evidence,
        "mode": result.mode,
        "model": result.model,
        "grounding_label": result.grounding.label if result.grounding else None,
        "grounding_score": result.grounding.score if result.grounding else None,
        "unsupported": result.grounding.unsupported if result.grounding else [],
        "retrieval_query": result.retrieval_query,
        "rewritten": result.rewritten,
        "best_score": result.retrieval.best_score if result.retrieval else 0.0,
        "fallback_used": result.fallback_used,
        "error": result.error,
        "seconds": result.seconds,
    }


# --------------------------------------------------------------------------- rendering
def render_sources(message: dict) -> None:
    citations = [Citation.from_dict(c) for c in message.get("citations", [])]
    if not citations:
        return
    has_evidence = message.get("has_evidence", False)
    query = message.get("retrieval_query", "")
    show_scores = st.session_state.get("show_scores", True)
    title = f"📚 Sources ({len(citations)})" if has_evidence else "🔎 Closest passages (below the relevance threshold)"
    with st.expander(title, expanded=has_evidence):
        if not has_evidence:
            st.caption("These passages were the nearest matches but were not similar enough to support an answer.")
        for c in citations:
            score = f'<span class="score" title="cosine similarity">{c.score:.2f}</span>' if show_scores else ""
            st.markdown(
                f'<div class="cite"><div class="cite-head"><span>📄 {html.escape(c.doc_name)} · page {c.page_number}</span>'
                f'{score}</div><div class="cite-body">{highlight_html(c.excerpt, query)}</div></div>',
                unsafe_allow_html=True,
            )


def render_answer_meta(message: dict) -> None:
    badges = []
    mode = message.get("mode", "")
    if mode:
        badges.append(f'<span class="badge">{html.escape(mode)}</span>')
    label = message.get("grounding_label")
    score = message.get("grounding_score")
    if label and message.get("has_evidence"):
        cls = "ok" if (score is None or score >= 0.999) else "warn"
        badges.append(f'<span class="badge {cls}">{html.escape(label)}</span>')
    if message.get("seconds"):
        badges.append(f'<span class="badge">{message["seconds"]:.1f}s</span>')
    if badges:
        st.markdown(" ".join(badges), unsafe_allow_html=True)
    if message.get("rewritten"):
        st.caption(f"Follow-up detected — searched for: “{message['retrieval_query']}”")
    if message.get("fallback_used"):
        st.warning(f"The LLM could not answer ({message.get('error')}). Showing the extractive answer instead.", icon="⚠️")
    unsupported = message.get("unsupported") or []
    if unsupported and message.get("has_evidence"):
        with st.expander(f"⚠️ {len(unsupported)} sentence(s) not found in the sources — verify before relying on them"):
            for s in unsupported:
                st.markdown(f"- {s}")


def render_message(message: dict) -> None:
    with st.chat_message(message["role"], avatar="🧑‍💻" if message["role"] == "user" else "📄"):
        if message["role"] == "assistant" and not message.get("has_evidence") and message.get("citations") is not None:
            st.info(message["content"], icon="🤔")
        else:
            st.markdown(message["content"])
        if message["role"] == "assistant":
            render_answer_meta(message)
            render_sources(message)


def handle_question(question: str) -> None:
    pipeline = get_pipeline()
    history = list(st.session_state.messages)
    user_message = {"role": "user", "content": question}
    st.session_state.messages.append(user_message)
    render_message(user_message)
    with st.chat_message("assistant", avatar="📄"):
        with st.spinner("Searching your documents…"):
            result = pipeline.ask(
                question,
                history=history,
                top_k=st.session_state.top_k,
                score_threshold=st.session_state.threshold,
            )
        message = answer_to_message(result)
        if not result.has_evidence:
            st.info(result.answer, icon="🤔")
        else:
            st.markdown(result.answer)
        render_answer_meta(message)
        render_sources(message)
    st.session_state.messages.append(message)


def pick_sample_question() -> None:
    value = st.session_state.get("sample_pill")
    if value:
        st.session_state.pending_question = value
        st.session_state.sample_pill = None


# --------------------------------------------------------------------------- sidebar
def sidebar() -> None:
    pipeline = get_pipeline()
    settings = get_settings()
    st.sidebar.title("📄 Document Assistant")
    st.sidebar.caption("Upload PDFs, ask questions, verify answers with page citations.")

    uploads = st.sidebar.file_uploader(
        "Upload PDF documents",
        accept_multiple_files=True,
        help="Text-based PDFs only. Scanned PDFs need OCR first.",
    )
    new_files = []
    for f in uploads or []:
        data = f.getvalue()
        if fingerprint(f.name, data) not in st.session_state.processed:
            new_files.append((f.name, data))
    if new_files:
        ingest_files(new_files, st.sidebar)

    if st.sidebar.button("📂 Load sample documents", use_container_width=True, disabled=st.session_state.sample_loaded):
        samples = load_sample_files()
        if samples:
            ingest_files(samples, st.sidebar)
            st.session_state.sample_loaded = True
            st.rerun()
        else:
            st.sidebar.error("Sample PDFs not found. Run `python sample_data/make_sample_pdfs.py`.")

    # ---- upload notices
    if st.session_state.notices:
        with st.sidebar.container():
            for level, text in st.session_state.notices[-6:]:
                getattr(st.sidebar, level)(text)
            if st.sidebar.button("Dismiss messages", use_container_width=True):
                st.session_state.notices = []
                st.rerun()

    # ---- document manager
    st.sidebar.subheader("Loaded documents")
    stats = pipeline.document_stats()
    if not stats:
        st.sidebar.caption("No documents yet.")
    for name, s in stats.items():
        col_name, col_btn = st.sidebar.columns([4, 1.4])
        col_name.markdown(f'<div class="doc-row">📄 <b>{html.escape(name)}</b><br/>{s["pages"]} pages · {s["chunks"]} chunks</div>',
                          unsafe_allow_html=True)
        if col_btn.button("🗑️", key=f"rm::{name}", help=f"Remove {name}", use_container_width=True):
            pipeline.remove_document(name)
            st.session_state.processed = {k: v for k, v in st.session_state.processed.items() if v != name}
            if name in {n for n, _ in load_sample_files()}:
                st.session_state.sample_loaded = False
            st.rerun()

    # ---- chat / document actions
    c1, c2 = st.sidebar.columns(2)
    if c1.button("🧹 Clear chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()
    if c2.button("🗑️ Remove all", use_container_width=True, disabled=not stats):
        pipeline.clear_documents()
        st.session_state.processed = {}
        st.session_state.sample_loaded = False
        st.session_state.messages = []
        st.rerun()

    # ---- settings
    with st.sidebar.expander("⚙️ Settings", expanded=False):
        modes = cached_modes(settings.ollama_base_url, settings.ollama_model, settings.hosted_provider,
                             settings.has_hosted_key, settings.hosted_model, settings.embedding_model)
        available = [m for m, ok, _ in modes if ok]
        labels = {"extractive": "Extractive (no LLM, no key)", "ollama": "Local LLM (Ollama)", "hosted": "Hosted LLM API"}
        current = pipeline.answer_mode if pipeline.answer_mode in available else "extractive"
        chosen = st.radio("Answer mode", available, index=available.index(current),
                          format_func=lambda m: labels.get(m, m))
        if chosen != pipeline.answer_mode:
            pipeline.set_answer_mode(chosen)
        for mode, ok, detail in modes:
            st.caption(f"{'🟢' if ok else '⚪'} **{labels.get(mode, mode)}** — {detail}")
        st.session_state.top_k = st.slider("Passages to retrieve (top-k)", 1, 10, st.session_state.top_k)
        st.session_state.threshold = st.slider("Relevance threshold", 0.0, 0.8, float(st.session_state.threshold), 0.05,
                                               help="Minimum cosine similarity for a passage to count as evidence.")
        st.session_state.show_scores = st.checkbox("Show similarity scores", st.session_state.show_scores)

    with st.sidebar.expander("ℹ️ How it works"):
        st.markdown(
            "1. PDF text is extracted per page and split into overlapping chunks.\n"
            "2. Chunks are embedded with a local Sentence Transformers model and stored in FAISS.\n"
            "3. Your question is embedded and the most similar chunks are retrieved.\n"
            "4. An answer is built only from those passages and cited by document and page.\n"
            "5. A grounding check flags any answer sentence not found in the sources."
        )


# --------------------------------------------------------------------------- main
def main() -> None:
    init_state()
    sidebar()
    pipeline = get_pipeline()

    st.title("AI-Powered Document Assistant")
    stats = pipeline.document_stats()
    pages = sum(s["pages"] for s in stats.values())
    chunks = sum(s["chunks"] for s in stats.values())
    st.caption(
        f"Retrieval-Augmented Generation · {len(stats)} document(s) · {pages} pages · {chunks} chunks · "
        f"answer mode: **{pipeline.answer_mode}**"
    )

    if not pipeline.has_documents:
        st.markdown(
            '<div class="hero"><h3>Ask questions about your PDFs — answers come with page citations.</h3>'
            "<ol><li>Upload one or more text-based PDFs in the sidebar (or load the sample documents).</li>"
            "<li>Ask a question in natural language. Follow-up questions are supported.</li>"
            "<li>Open the <b>Sources</b> panel under each answer to verify the page and excerpt.</li></ol>"
            "<p>No API key is needed: the default mode extracts answers directly from your documents. "
            "Connect Ollama or a hosted LLM in <b>Settings</b> for fluent, synthesised answers.</p></div>",
            unsafe_allow_html=True,
        )
        if st.button("📂 Try it with the sample documents", type="primary"):
            samples = load_sample_files()
            ingest_files(samples, st.sidebar)
            st.session_state.sample_loaded = True
            st.rerun()
        st.markdown("**Example questions you can ask once documents are loaded:**")
        for q in load_sample_questions()[:4]:
            st.markdown(f"- {q}")
        st.chat_input("Upload a document first…", disabled=True)
        return

    if st.session_state.sample_loaded:
        questions = load_sample_questions()
        if hasattr(st, "pills"):
            st.pills("Try a sample question", questions, selection_mode="single", key="sample_pill",
                     on_change=pick_sample_question, label_visibility="collapsed")
        else:  # older Streamlit versions
            cols = st.columns(3)
            for i, q in enumerate(questions):
                if cols[i % 3].button(q, key=f"sq{i}", use_container_width=True):
                    st.session_state.pending_question = q

    for message in st.session_state.messages:
        render_message(message)

    question = st.chat_input("Ask a question about your documents…")
    pending = st.session_state.pop("pending_question", None)
    question = question or pending
    if question:
        handle_question(question)


if __name__ == "__main__" or True:  # Streamlit executes the module directly
    main()
