"""Cheap, deterministic answer-grounding check.

Each answer sentence is considered *supported* when enough of its content
words (and word bigrams) appear in the retrieved context.  This is not a
substitute for human review, but it reliably flags hallucinated sentences and
lets the evaluation script measure groundedness without an LLM judge.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from rag.chunking import split_sentences
from rag.citations import STOPWORDS

_WORD = re.compile(r"[a-z0-9]+")
_CITATION = re.compile(r"\[[^\[\]]*?,\s*p\.\s*\d+\]")  # "[doc.pdf, p. 3]"

ABSTAIN_PHRASES = (
    "couldn't find", "could not find", "no relevant", "not found in the", "does not contain",
    "doesn't contain", "not mentioned", "no information", "unable to find", "cannot find",
)


@dataclass
class GroundingReport:
    score: float                       # supported sentences / checked sentences (1.0 when nothing to check)
    supported: list[str] = field(default_factory=list)
    unsupported: list[str] = field(default_factory=list)
    abstained: bool = False

    @property
    def checked(self) -> int:
        return len(self.supported) + len(self.unsupported)

    @property
    def label(self) -> str:
        if self.abstained:
            return "abstained (no evidence)"
        if self.checked == 0:
            return "nothing to check"
        return f"{self.score:.0%} of answer sentences supported by sources"


def content_tokens(text: str) -> list[str]:
    return [t for t in _WORD.findall(text.lower()) if t not in STOPWORDS]


def is_abstention(text: str) -> bool:
    lowered = text.lower()
    return any(p in lowered for p in ABSTAIN_PHRASES)


def sentence_support(sentence: str, context_tokens: set[str], context_bigrams: set[tuple[str, str]],
                     min_coverage: float = 0.6) -> bool:
    tokens = content_tokens(sentence)
    if len(tokens) < 2:
        return True  # too short to judge (e.g. "Yes.")
    unigram_cov = sum(1 for t in tokens if t in context_tokens) / len(tokens)
    bigrams = list(zip(tokens, tokens[1:]))
    bigram_cov = (sum(1 for b in bigrams if b in context_bigrams) / len(bigrams)) if bigrams else 0.0
    return unigram_cov >= min_coverage or bigram_cov >= 0.5


def grounding_score(answer: str, contexts: list[str], min_coverage: float = 0.6) -> GroundingReport:
    if not answer or not answer.strip():
        return GroundingReport(score=1.0)
    if is_abstention(answer):
        return GroundingReport(score=1.0, abstained=True)

    joined_tokens = [t for ctx in contexts for t in content_tokens(ctx)]
    context_tokens = set(joined_tokens)
    context_bigrams: set[tuple[str, str]] = set()
    for ctx in contexts:
        toks = content_tokens(ctx)
        context_bigrams.update(zip(toks, toks[1:]))

    cleaned = _CITATION.sub("", answer)
    report = GroundingReport(score=1.0)
    for sentence in split_sentences(cleaned):
        if len(content_tokens(sentence)) < 2:
            continue
        if sentence_support(sentence, context_tokens, context_bigrams, min_coverage):
            report.supported.append(sentence)
        else:
            report.unsupported.append(sentence)
    if report.checked:
        report.score = len(report.supported) / report.checked
    return report
