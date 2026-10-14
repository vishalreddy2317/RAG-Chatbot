"""Turn follow-up questions into self-contained retrieval queries.

Example: "What is the enrollment fee?" -> "Does it apply to businesses too?"
The second question is rewritten for retrieval as
"What is the enrollment fee? Does it apply to businesses too?" so the vector
search still finds the right passages.  Chat history is also passed to LLM
generators so the model understands the pronoun.
"""

from __future__ import annotations

import re

_WORD = re.compile(r"[a-z0-9']+")

FOLLOWUP_HINTS = {
    "it", "its", "this", "that", "these", "those", "they", "them", "their", "he", "she", "his",
    "her", "there", "one", "same", "also", "else", "more", "again", "above", "previous", "earlier",
}


def previous_user_questions(history: list[dict] | None, limit: int = 1) -> list[str]:
    if not history:
        return []
    questions = [m.get("content", "") for m in history if m.get("role") == "user" and m.get("content")]
    return questions[-limit:]


def is_followup(question: str) -> bool:
    tokens = _WORD.findall(question.lower())
    if not tokens:
        return False
    return len(tokens) <= 4 or any(t in FOLLOWUP_HINTS for t in tokens)


def rewrite_for_retrieval(question: str, history: list[dict] | None) -> str:
    """Return a retrieval query that includes the previous question when
    ``question`` looks like a follow-up.  Otherwise return ``question`` unchanged."""
    question = question.strip()
    previous = previous_user_questions(history, limit=1)
    if previous and is_followup(question):
        return f"{previous[0].strip()} {question}"
    return question
