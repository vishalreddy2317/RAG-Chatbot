from rag.chunking import Chunk
from rag.generators import (
    NO_DIRECT_ANSWER_MESSAGE,
    NO_EVIDENCE_MESSAGE,
    ExtractiveGenerator,
    OllamaGenerator,
    build_context_block,
    build_user_prompt,
    is_answer_sentence,
    strip_citations,
)
from rag.vector_store import SearchResult


def _result(text, page, score=0.6, rank=1, doc="handbook.pdf"):
    return SearchResult(Chunk(f"{doc}::p{page}::c{rank}", doc, page, text, rank, 0, len(text)), score, rank)


RESULTS = [
    _result("Planned maintenance is scheduled twice a year, in April and October. Outages are reported by email.", 3),
    _result("There is no enrollment fee and no credit check. Cancellations require 30 days' written notice.", 1, 0.4, 2),
]


def test_extractive_answer_contains_relevant_sentence_and_citation(hashing_embedder):
    gen = ExtractiveGenerator(hashing_embedder, max_sentences=2, min_score=0.0)
    answer = gen.generate("When is planned maintenance scheduled?", RESULTS)
    assert answer.ok and answer.mode == "extractive"
    assert "April and October" in answer.text
    assert "[handbook.pdf, p. 3]" in answer.text
    assert answer.used_chunk_ids[0] == "handbook.pdf::p3::c1"


def test_extractive_without_results_abstains(hashing_embedder):
    gen = ExtractiveGenerator(hashing_embedder)
    assert gen.generate("anything", []).text == NO_EVIDENCE_MESSAGE


def test_extractive_reports_no_direct_answer_when_sentences_are_weak(hashing_embedder):
    gen = ExtractiveGenerator(hashing_embedder, min_score=0.99)
    answer = gen.generate("completely unrelated query words", RESULTS)
    assert answer.text == NO_DIRECT_ANSWER_MESSAGE


def test_is_answer_sentence_filters_headings_and_questions():
    assert is_answer_sentence("Planned maintenance is scheduled twice a year, in April and October.")
    assert not is_answer_sentence("Can I subscribe if I already have rooftop solar?")
    assert not is_answer_sentence("Orion Community Solar Program - Member Handbook")
    assert not is_answer_sentence("3. Billing")


def test_prompt_contains_context_question_and_history():
    history = [{"role": "user", "content": "What is the fee?"}, {"role": "assistant", "content": "No fee."}]
    prompt, used = build_user_prompt("Does it apply to renters?", RESULTS, history, max_context_chars=2000)
    assert "[1] (handbook.pdf, p. 3)" in prompt and "[2] (handbook.pdf, p. 1)" in prompt
    assert "Question: Does it apply to renters?" in prompt
    assert "User: What is the fee?" in prompt
    assert used == ["handbook.pdf::p3::c1", "handbook.pdf::p1::c2"]


def test_context_block_respects_budget():
    block, used = build_context_block(RESULTS, max_chars=150)
    assert len(used) == 1 and "[2]" not in block


def test_ollama_unreachable_returns_error_not_exception():
    gen = OllamaGenerator("http://127.0.0.1:9", "llama3.2:3b", timeout=1)
    answer = gen.generate("question", RESULTS)
    assert not answer.ok and "Ollama request failed" in answer.error
    ok, detail = OllamaGenerator.is_available("http://127.0.0.1:9")
    assert not ok and "not reachable" in detail


def test_strip_citations():
    assert strip_citations("Fee is 5 dollars [a.pdf, p. 2].  Next [b.pdf, p. 10]") == "Fee is 5 dollars . Next"
