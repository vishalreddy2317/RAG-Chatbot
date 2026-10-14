"""Answer generators.

Three interchangeable back-ends implement :class:`BaseGenerator`:

* :class:`ExtractiveGenerator` - **no API key, no LLM**.  Ranks sentences from
  the retrieved passages by similarity to the question and returns the best
  ones with page citations.  Always available.
* :class:`OllamaGenerator` - a small local instruct model served by Ollama
  (``ollama run llama3.2:3b``).  Free, private, needs ~4 GB RAM.
* :class:`HostedGenerator` - optional hosted API (any OpenAI-compatible
  endpoint such as OpenAI/Groq/OpenRouter, or the Anthropic Messages API).
  Only used when a key is configured.

All LLM back-ends receive the same grounded prompt and the UI shows a
grounding score for every answer, whichever generator produced it.
"""

from __future__ import annotations

import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field

import numpy as np
import requests

from rag.chunking import split_sentences
from rag.citations import query_terms
from rag.config import Settings
from rag.embeddings import Embedder
from rag.vector_store import SearchResult

NO_EVIDENCE_MESSAGE = (
    "I couldn't find relevant information in the uploaded documents to answer that question. "
    "Try rephrasing it, or check that the document containing the answer has been uploaded."
)

NO_DIRECT_ANSWER_MESSAGE = (
    "The documents mention related topics, but I couldn't find a passage that directly answers that question. "
    "The closest passages are listed in the sources below."
)

SYSTEM_PROMPT = """You are a careful document assistant. Answer the user's question using ONLY the numbered context passages provided.
Rules:
1. If the passages do not contain the answer, reply exactly: "I couldn't find that in the uploaded documents." Do not guess.
2. Never use outside knowledge or make assumptions beyond the passages.
3. After every claim, cite its source in square brackets using the document name and page exactly as given, e.g. [handbook.pdf, p. 3].
4. Be concise and factual. Use the wording of the passages where possible."""


@dataclass
class GeneratedAnswer:
    text: str
    mode: str                 # extractive | ollama | hosted
    model: str
    used_chunk_ids: list[str] = field(default_factory=list)
    error: str | None = None

    @property
    def ok(self) -> bool:
        return self.error is None and bool(self.text.strip())


# ---------------------------------------------------------------- prompting
def build_context_block(results: list[SearchResult], max_chars: int = 6000) -> tuple[str, list[str]]:
    """Format retrieved passages as numbered blocks within a character budget."""
    blocks: list[str] = []
    used: list[str] = []
    total = 0
    for i, r in enumerate(results, start=1):
        body = " ".join(r.chunk.text.split())
        block = f"[{i}] ({r.chunk.doc_name}, p. {r.chunk.page_number})\n{body}"
        if total + len(block) > max_chars and blocks:
            break
        if len(block) > max_chars:  # single oversized block: truncate instead of dropping all context
            block = block[: max_chars - 1] + "…"
        blocks.append(block)
        used.append(r.chunk.chunk_id)
        total += len(block)
    return "\n\n".join(blocks), used


def format_history(history: list[dict] | None, max_turns: int = 3) -> str:
    if not history:
        return ""
    turns = [m for m in history if m.get("role") in ("user", "assistant") and m.get("content")]
    turns = turns[-(max_turns * 2):]
    lines = []
    for m in turns:
        role = "User" if m["role"] == "user" else "Assistant"
        content = " ".join(str(m["content"]).split())
        if len(content) > 500:
            content = content[:500] + "…"
        lines.append(f"{role}: {content}")
    return "\n".join(lines)


def build_user_prompt(question: str, results: list[SearchResult], history: list[dict] | None = None,
                      max_context_chars: int = 6000) -> tuple[str, list[str]]:
    context, used = build_context_block(results, max_context_chars)
    history_block = format_history(history)
    parts = []
    if history_block:
        parts.append("Conversation so far (for resolving references like 'it' or 'that'):\n" + history_block)
    parts.append("Context passages:\n" + context)
    parts.append(f"Question: {question}\n\nAnswer (with [document, p. N] citations):")
    return "\n\n".join(parts), used


# --------------------------------------------------------------- generators
class BaseGenerator(ABC):
    mode: str = "base"

    @abstractmethod
    def generate(self, question: str, results: list[SearchResult], history: list[dict] | None = None) -> GeneratedAnswer:
        ...

    def describe(self) -> str:
        return self.mode


_MIN_SENTENCE_CHARS = 25
_MAX_SENTENCE_CHARS = 600
_MIN_SENTENCE_WORDS = 5
_WORD_RE = re.compile(r"[A-Za-z][A-Za-z'-]*")


def _term_fraction(sentence: str, terms: list[str]) -> float:
    if not terms:
        return 0.0
    words = set(re.findall(r"[a-z0-9]+", sentence.lower()))
    hits = sum(1 for t in terms if t in words or any(w.startswith(t) for w in words))
    return hits / len(terms)


def is_answer_sentence(sentence: str) -> bool:
    """Filter out headings, running titles, list numbers and questions."""
    if not (_MIN_SENTENCE_CHARS <= len(sentence) <= _MAX_SENTENCE_CHARS):
        return False
    if sentence.rstrip().endswith("?"):
        return False
    words = _WORD_RE.findall(sentence)
    if len(words) < _MIN_SENTENCE_WORDS:
        return False
    capitalised = sum(1 for w in words if w[0].isupper())
    if capitalised / len(words) > 0.6:  # Title Case heading or running header
        return False
    return True


class ExtractiveGenerator(BaseGenerator):
    """Sentence-ranking extractive answers: no LLM required."""

    mode = "extractive"

    def __init__(self, embedder: Embedder, max_sentences: int = 3, min_score: float = 0.25,
                 relative_cutoff: float = 0.8):
        self.embedder = embedder
        self.max_sentences = max_sentences
        self.min_score = min_score
        self.relative_cutoff = relative_cutoff

    def describe(self) -> str:
        return f"extractive sentence ranking ({self.embedder.model_name})"

    def generate(self, question: str, results: list[SearchResult], history: list[dict] | None = None) -> GeneratedAnswer:
        if not results:
            return GeneratedAnswer(NO_EVIDENCE_MESSAGE, self.mode, self.describe())

        candidates: list[tuple[str, SearchResult]] = []
        seen: set[str] = set()
        for r in results:
            for sentence in split_sentences(r.chunk.text):
                sentence = " ".join(sentence.split())
                key = sentence.lower()
                if key not in seen and is_answer_sentence(sentence):
                    seen.add(key)
                    candidates.append((sentence, r))

        if not candidates:  # chunks with no proper sentences: fall back to the best passage
            top = results[0]
            text = " ".join(top.chunk.text.split())[:_MAX_SENTENCE_CHARS]
            return GeneratedAnswer(f"{text} [{top.chunk.citation_label}]", self.mode, self.describe(), [top.chunk.chunk_id])

        sentence_vectors = self.embedder.embed_texts([s for s, _ in candidates])
        query_vector = self.embedder.embed_query(question)
        sentence_scores = sentence_vectors @ query_vector
        # Blend sentence relevance with the parent chunk's retrieval score and a small
        # lexical bonus (fraction of query content words present in the sentence).
        terms = query_terms(question)
        lexical = np.array([_term_fraction(s, terms) for s, _ in candidates], dtype="float32")
        chunk_scores = np.array([r.score for _, r in candidates], dtype="float32")
        blended = 0.65 * sentence_scores + 0.20 * chunk_scores + 0.15 * lexical
        order = np.argsort(-blended)

        if float(sentence_scores[order[0]]) < self.min_score:
            # Chunks were similar enough to count as evidence, but no single sentence answers the question.
            return GeneratedAnswer(NO_DIRECT_ANSWER_MESSAGE, self.mode, self.describe())

        # Keep sentences that are both above the absolute floor and reasonably close to the best
        # one, so a strong first sentence is not padded with loosely related ones.
        best = float(sentence_scores[order[0]])
        cutoff = max(self.min_score, best * self.relative_cutoff)
        chosen: list[tuple[str, SearchResult, float]] = []
        for idx in order:
            if len(chosen) >= self.max_sentences:
                break
            if float(sentence_scores[idx]) < cutoff:
                break
            sentence, r = candidates[int(idx)]
            chosen.append((sentence, r, float(blended[idx])))

        lines = [f"{sentence} [{r.chunk.citation_label}]" for sentence, r, _ in chosen]
        used = list(dict.fromkeys(r.chunk.chunk_id for _, r, _ in chosen))
        return GeneratedAnswer("\n\n".join(lines), self.mode, self.describe(), used)


class OllamaGenerator(BaseGenerator):
    """Local LLM via the Ollama HTTP API (http://localhost:11434)."""

    mode = "ollama"

    def __init__(self, base_url: str = "http://localhost:11434", model: str = "llama3.2:3b",
                 timeout: int = 60, max_context_chars: int = 6000):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout = timeout
        self.max_context_chars = max_context_chars

    def describe(self) -> str:
        return f"Ollama · {self.model}"

    @staticmethod
    def available_models(base_url: str, timeout: float = 1.5) -> list[str]:
        """Return model names served by Ollama, or [] when it is not reachable."""
        try:
            resp = requests.get(f"{base_url.rstrip('/')}/api/tags", timeout=timeout)
            resp.raise_for_status()
            return [m.get("name", "") for m in resp.json().get("models", []) if m.get("name")]
        except Exception:
            return []

    @classmethod
    def is_available(cls, base_url: str, model: str | None = None, timeout: float = 1.5) -> tuple[bool, str]:
        models = cls.available_models(base_url, timeout)
        if not models:
            return False, f"Ollama is not reachable at {base_url} (install from https://ollama.com and run `ollama serve`)."
        if model and model not in models and model.split(":")[0] not in {m.split(":")[0] for m in models}:
            return False, f"Ollama is running but model '{model}' is not pulled (run `ollama pull {model}`). Available: {', '.join(models)}"
        return True, f"Ollama reachable at {base_url}"

    def generate(self, question: str, results: list[SearchResult], history: list[dict] | None = None) -> GeneratedAnswer:
        if not results:
            return GeneratedAnswer(NO_EVIDENCE_MESSAGE, self.mode, self.describe())
        prompt, used = build_user_prompt(question, results, history, self.max_context_chars)
        payload = {
            "model": self.model,
            "stream": False,
            "options": {"temperature": 0.1, "num_ctx": 4096},
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
        }
        try:
            resp = requests.post(f"{self.base_url}/api/chat", json=payload, timeout=self.timeout)
            resp.raise_for_status()
            text = (resp.json().get("message") or {}).get("content", "").strip()
            if not text:
                return GeneratedAnswer("", self.mode, self.describe(), used, error="Ollama returned an empty answer.")
            return GeneratedAnswer(text, self.mode, self.describe(), used)
        except requests.RequestException as exc:
            return GeneratedAnswer("", self.mode, self.describe(), used, error=f"Ollama request failed: {exc}")
        except ValueError as exc:
            return GeneratedAnswer("", self.mode, self.describe(), used, error=f"Ollama returned invalid JSON: {exc}")


class HostedGenerator(BaseGenerator):
    """Optional hosted LLM (OpenAI-compatible chat completions or Anthropic Messages)."""

    mode = "hosted"

    def __init__(self, provider: str, api_key: str, model: str, base_url: str = "https://api.openai.com/v1",
                 timeout: int = 60, max_context_chars: int = 6000, max_tokens: int = 600):
        if provider not in ("openai", "anthropic"):
            raise ValueError("provider must be 'openai' or 'anthropic'")
        if not api_key:
            raise ValueError("an API key is required for the hosted generator")
        self.provider = provider
        self.api_key = api_key
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.max_context_chars = max_context_chars
        self.max_tokens = max_tokens

    def describe(self) -> str:
        return f"{self.provider} · {self.model}"

    def _call_openai(self, prompt: str) -> str:
        resp = requests.post(
            f"{self.base_url}/chat/completions",
            headers={"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"},
            json={
                "model": self.model,
                "temperature": 0.1,
                "max_tokens": self.max_tokens,
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
            },
            timeout=self.timeout,
        )
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]["content"].strip()

    def _call_anthropic(self, prompt: str) -> str:
        resp = requests.post(
            "https://api.anthropic.com/v1/messages",
            headers={
                "x-api-key": self.api_key,
                "anthropic-version": "2023-06-01",
                "Content-Type": "application/json",
            },
            json={
                "model": self.model,
                "max_tokens": self.max_tokens,
                "temperature": 0.1,
                "system": SYSTEM_PROMPT,
                "messages": [{"role": "user", "content": prompt}],
            },
            timeout=self.timeout,
        )
        resp.raise_for_status()
        blocks = resp.json().get("content", [])
        return "".join(b.get("text", "") for b in blocks if b.get("type") == "text").strip()

    def generate(self, question: str, results: list[SearchResult], history: list[dict] | None = None) -> GeneratedAnswer:
        if not results:
            return GeneratedAnswer(NO_EVIDENCE_MESSAGE, self.mode, self.describe())
        prompt, used = build_user_prompt(question, results, history, self.max_context_chars)
        try:
            text = self._call_anthropic(prompt) if self.provider == "anthropic" else self._call_openai(prompt)
            if not text:
                return GeneratedAnswer("", self.mode, self.describe(), used, error="The hosted model returned an empty answer.")
            return GeneratedAnswer(text, self.mode, self.describe(), used)
        except requests.HTTPError as exc:
            detail = exc.response.text[:300] if exc.response is not None else str(exc)
            return GeneratedAnswer("", self.mode, self.describe(), used, error=f"Hosted API error: {detail}")
        except (requests.RequestException, KeyError, IndexError, ValueError) as exc:
            return GeneratedAnswer("", self.mode, self.describe(), used, error=f"Hosted API request failed: {exc}")


# ----------------------------------------------------------------- factory
@dataclass
class ModeInfo:
    mode: str
    available: bool
    detail: str


def describe_modes(settings: Settings) -> list[ModeInfo]:
    """Report which answer modes can be used right now (for the UI and logs)."""
    ollama_ok, ollama_detail = OllamaGenerator.is_available(settings.ollama_base_url, settings.ollama_model)
    if settings.has_hosted_key:
        hosted_detail = f"{settings.hosted_provider} · {settings.hosted_model}"
    else:
        key_name = "ANTHROPIC_API_KEY" if settings.hosted_provider == "anthropic" else "OPENAI_API_KEY"
        hosted_detail = f"no API key configured (set {key_name})"
    return [
        ModeInfo("extractive", True, f"sentence ranking with {settings.embedding_model.split('/')[-1]} (no API key)"),
        ModeInfo("ollama", ollama_ok, ollama_detail),
        ModeInfo("hosted", settings.has_hosted_key, hosted_detail),
    ]


def resolve_answer_mode(settings: Settings) -> str:
    """Map ``auto`` to the best available concrete mode."""
    if settings.answer_mode != "auto":
        return settings.answer_mode
    ok, _ = OllamaGenerator.is_available(settings.ollama_base_url, settings.ollama_model)
    if ok:
        return "ollama"
    if settings.has_hosted_key:
        return "hosted"
    return "extractive"


def build_generator(settings: Settings, embedder: Embedder, mode: str | None = None) -> BaseGenerator:
    mode = mode or resolve_answer_mode(settings)
    if mode == "ollama":
        return OllamaGenerator(settings.ollama_base_url, settings.ollama_model, settings.request_timeout,
                               settings.max_context_chars)
    if mode == "hosted":
        if not settings.has_hosted_key:
            raise ValueError("Hosted mode selected but no API key is configured.")
        return HostedGenerator(settings.hosted_provider, settings.hosted_api_key, settings.hosted_model,
                               settings.openai_base_url, settings.request_timeout, settings.max_context_chars)
    if mode in ("extractive", "auto"):
        return ExtractiveGenerator(embedder, settings.extractive_max_sentences)
    raise ValueError(f"Unknown answer mode: {mode}")


_CITATION_RE = re.compile(r"\[[^\[\]]*?,\s*p\.\s*\d+\]")


def strip_citations(text: str) -> str:
    """Remove inline ``[doc, p. N]`` markers (used by evaluation keyword checks)."""
    return " ".join(_CITATION_RE.sub("", text).split())
