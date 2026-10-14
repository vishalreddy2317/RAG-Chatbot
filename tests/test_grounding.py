from rag.generators import NO_EVIDENCE_MESSAGE
from rag.grounding import grounding_score, is_abstention

CONTEXT = [
    "Members receive a bill credit at the community solar credit rate, which was set at 11.7 cents per "
    "kilowatt-hour for 2024. Subscription payments are billed on the 5th of each month."
]


def test_copied_answer_is_fully_supported():
    report = grounding_score("The credit rate was set at 11.7 cents per kilowatt-hour for 2024. [orion.pdf, p. 3]", CONTEXT)
    assert report.score == 1.0
    assert report.unsupported == []
    assert "100%" in report.label


def test_fabricated_sentence_is_flagged():
    answer = ("The credit rate was set at 11.7 cents per kilowatt-hour for 2024. "
              "Members also receive a free electric vehicle charger every winter.")
    report = grounding_score(answer, CONTEXT)
    assert report.score == 0.5
    assert len(report.unsupported) == 1
    assert "electric vehicle" in report.unsupported[0]


def test_abstention_is_not_penalised():
    report = grounding_score(NO_EVIDENCE_MESSAGE, CONTEXT)
    assert report.abstained and report.score == 1.0
    assert is_abstention("I couldn't find that in the uploaded documents.")
    assert not is_abstention("The fee is 5 dollars.")


def test_empty_answer_and_empty_context():
    assert grounding_score("", CONTEXT).score == 1.0
    report = grounding_score("Totally invented claim about something else entirely.", [])
    assert report.score == 0.0
