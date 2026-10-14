"""Evaluate retrieval quality, answer accuracy, grounding and abstention.

Usage:
    python evaluation/evaluate.py                 # extractive mode (no key needed)
    python evaluation/evaluate.py --mode ollama   # local LLM (if Ollama is running)
    python evaluation/evaluate.py --mode hosted   # hosted API (if a key is configured)

Metrics
-------
* Recall@k   - an expected (document, page) appears in the top-k retrieved chunks (k = 1, 3, 5)
* MRR        - mean reciprocal rank of the first expected page
* Doc@1      - the top chunk comes from the expected document (multi-document disambiguation)
* Keyword hit- the answer text contains at least one expected keyword (answer correctness proxy)
* Grounding  - share of answer sentences supported by the retrieved passages (rag.grounding)
* Abstain    - share of unanswerable questions where the assistant declined to answer
* Follow-ups - Recall@3 for the second turn with vs. without conversation history

Results are written to evaluation/results/results_<mode>.json and .md.  Nothing is fabricated:
every number comes from running the pipeline on the sample PDFs in this repository.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import statistics
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from rag.config import Settings  # noqa: E402
from rag.generators import describe_modes, strip_citations  # noqa: E402
from rag.pipeline import RAGPipeline  # noqa: E402

K_VALUES = (1, 3, 5)


def page_hits(result, doc: str, pages: list[str | int]) -> list[bool]:
    wanted = {(doc, int(p)) for p in pages}
    return [(c.doc_name, c.page_number) in wanted for c in result.citations]


def first_rank(hits: list[bool]) -> int | None:
    for i, hit in enumerate(hits, start=1):
        if hit:
            return i
    return None


def keyword_hit(answer: str, keywords: list[str]) -> bool:
    text = " ".join(strip_citations(answer).lower().split())
    return any(" ".join(k.lower().split()) in text for k in keywords)


def pct(values: list[bool | float]) -> float:
    return round(100.0 * (sum(values) / len(values)), 1) if values else 0.0


def evaluate(mode: str, top_k: int, questions_path: Path, out_dir: Path) -> dict:
    spec = json.loads(questions_path.read_text(encoding="utf-8"))
    settings = Settings.from_env()
    settings.top_k = max(top_k, max(K_VALUES))
    pipeline = RAGPipeline(settings, answer_mode=mode)
    print(f"answer mode: {pipeline.answer_mode} ({pipeline.generator.describe()})")

    for rel in spec["documents"]:
        report = pipeline.ingest(ROOT / rel)
        print(f"ingested {report.doc_name}: {report.num_pages} pages, {report.num_chunks} chunks ({report.status.value})")

    # ---------------------------------------------------------------- answerable
    rows = []
    for item in spec["answerable"]:
        res = pipeline.ask(item["question"], top_k=max(K_VALUES))
        hits = page_hits(res, item["doc"], item["pages"])
        rank = first_rank(hits)
        top = res.citations[0] if res.citations else None
        rows.append({
            "id": item["id"], "question": item["question"], "expected": f"{item['doc']} p.{'/'.join(map(str, item['pages']))}",
            "top1": f"{top.doc_name} p.{top.page_number} ({top.score:.2f})" if top else "-",
            "doc_at_1": bool(top and top.doc_name == item["doc"]),
            **{f"recall@{k}": any(hits[:k]) for k in K_VALUES},
            "rank": rank, "rr": (1.0 / rank) if rank else 0.0,
            "has_evidence": res.has_evidence,
            "keyword_hit": res.has_evidence and keyword_hit(res.answer, item["keywords"]),
            "grounding": res.grounding.score if res.grounding else None,
            "seconds": res.seconds, "answer": res.answer,
        })
        flag = "✓" if rows[-1]["keyword_hit"] else "✗"
        print(f"  {item['id']} R@1={int(rows[-1]['recall@1'])} R@3={int(rows[-1]['recall@3'])} kw={flag} {item['question'][:60]}")

    # -------------------------------------------------------------- unanswerable
    abstain_rows = []
    for item in spec["unanswerable"]:
        res = pipeline.ask(item["question"], top_k=max(K_VALUES))
        abstained = (not res.has_evidence) or (res.grounding is not None and res.grounding.abstained)
        abstain_rows.append({"id": item["id"], "question": item["question"], "category": item.get("category", ""),
                             "best_score": round(res.retrieval.best_score, 3) if res.retrieval else 0.0,
                             "abstained": abstained, "answer": res.answer})
        print(f"  {item['id']} abstain={'✓' if abstained else '✗'} best={abstain_rows[-1]['best_score']:.2f} {item['question'][:60]}")

    # ----------------------------------------------------------------- followups
    followup_rows = []
    for item in spec["followups"]:
        first, second = item["turns"]
        r1 = pipeline.ask(first, top_k=max(K_VALUES))
        history = [{"role": "user", "content": first}, {"role": "assistant", "content": r1.answer}]
        with_hist = pipeline.ask(second, history=history, top_k=max(K_VALUES))
        without = pipeline.ask(second, top_k=max(K_VALUES))
        followup_rows.append({
            "id": item["id"], "turns": item["turns"], "retrieval_query": with_hist.retrieval_query,
            "recall@3_with_history": any(page_hits(with_hist, item["doc"], item["pages"])[:3]),
            "recall@3_without_history": any(page_hits(without, item["doc"], item["pages"])[:3]),
            "keyword_hit": with_hist.has_evidence and keyword_hit(with_hist.answer, item["keywords"]),
            "answer": with_hist.answer,
        })
        print(f"  {item['id']} R@3 with={int(followup_rows[-1]['recall@3_with_history'])} "
              f"without={int(followup_rows[-1]['recall@3_without_history'])} kw={'✓' if followup_rows[-1]['keyword_hit'] else '✗'}")

    # ------------------------------------------------------------------ summary
    grounded = [r["grounding"] for r in rows if r["grounding"] is not None]
    off_topic = [r for r in abstain_rows if r["category"] == "off-topic"]
    in_domain = [r for r in abstain_rows if r["category"] != "off-topic"]
    summary = {
        "mode": pipeline.answer_mode, "generator": pipeline.generator.describe(),
        "embedding_model": settings.embedding_model, "chunk_size": settings.chunk_size,
        "chunk_overlap": settings.chunk_overlap, "score_threshold": settings.score_threshold,
        "strong_score": settings.strong_score,
        "n_answerable": len(rows), "n_unanswerable": len(abstain_rows), "n_followups": len(followup_rows),
        **{f"recall@{k}": pct([r[f"recall@{k}"] for r in rows]) for k in K_VALUES},
        "mrr": round(statistics.mean(r["rr"] for r in rows), 3) if rows else 0.0,
        "doc@1": pct([r["doc_at_1"] for r in rows]),
        "answered": pct([r["has_evidence"] for r in rows]),
        "keyword_hit": pct([r["keyword_hit"] for r in rows]),
        "grounding_mean": round(statistics.mean(grounded), 3) if grounded else None,
        "fully_grounded": pct([g >= 0.999 for g in grounded]),
        "abstain_overall": pct([r["abstained"] for r in abstain_rows]),
        "abstain_off_topic": pct([r["abstained"] for r in off_topic]),
        "abstain_in_domain": pct([r["abstained"] for r in in_domain]),
        "followup_recall@3_with_history": pct([r["recall@3_with_history"] for r in followup_rows]),
        "followup_recall@3_without_history": pct([r["recall@3_without_history"] for r in followup_rows]),
        "followup_keyword_hit": pct([r["keyword_hit"] for r in followup_rows]),
        "mean_seconds": round(statistics.mean(r["seconds"] for r in rows), 3) if rows else None,
        "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "python": platform.python_version(), "platform": platform.platform(),
    }

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"results_{pipeline.answer_mode}.json").write_text(
        json.dumps({"summary": summary, "answerable": rows, "unanswerable": abstain_rows, "followups": followup_rows}, indent=2),
        encoding="utf-8")
    (out_dir / f"results_{pipeline.answer_mode}.md").write_text(render_markdown(summary, rows, abstain_rows, followup_rows), encoding="utf-8")
    return summary


def render_markdown(summary: dict, rows: list[dict], abstain_rows: list[dict], followup_rows: list[dict]) -> str:
    lines = [f"# Evaluation results — mode `{summary['mode']}`", "",
             f"Generated {summary['timestamp']} · {summary['generator']} · embeddings `{summary['embedding_model']}` · "
             f"chunk {summary['chunk_size']}/{summary['chunk_overlap']} · threshold {summary['score_threshold']}", "",
             "## Summary", "", "| Metric | Value |", "|---|---|",
             f"| Answerable questions | {summary['n_answerable']} |",
             f"| Recall@1 / @3 / @5 (expected page retrieved) | {summary['recall@1']}% / {summary['recall@3']}% / {summary['recall@5']}% |",
             f"| MRR | {summary['mrr']} |",
             f"| Correct document at rank 1 | {summary['doc@1']}% |",
             f"| Answered (evidence found) | {summary['answered']}% |",
             f"| Keyword hit (answer contains expected fact) | {summary['keyword_hit']}% |",
             f"| Mean grounding score / fully grounded answers | {summary['grounding_mean']} / {summary['fully_grounded']}% |",
             f"| Abstained on unanswerable: overall / off-topic / in-domain | {summary['abstain_overall']}% / {summary['abstain_off_topic']}% / {summary['abstain_in_domain']}% |",
             f"| Follow-up Recall@3 with / without history | {summary['followup_recall@3_with_history']}% / {summary['followup_recall@3_without_history']}% |",
             f"| Follow-up keyword hit | {summary['followup_keyword_hit']}% |",
             f"| Mean latency per question | {summary['mean_seconds']} s |", "",
             "## Answerable questions", "", "| id | question | expected | top-1 (score) | R@1 | R@3 | keyword | grounding |", "|---|---|---|---|---|---|---|---|"]
    for r in rows:
        g = f"{r['grounding']:.2f}" if r["grounding"] is not None else "-"
        lines.append(f"| {r['id']} | {r['question']} | {r['expected']} | {r['top1']} | {'✓' if r['recall@1'] else '✗'} | "
                     f"{'✓' if r['recall@3'] else '✗'} | {'✓' if r['keyword_hit'] else '✗'} | {g} |")
    lines += ["", "## Unanswerable questions (should abstain)", "", "| id | category | question | best score | abstained |", "|---|---|---|---|---|"]
    for r in abstain_rows:
        lines.append(f"| {r['id']} | {r['category']} | {r['question']} | {r['best_score']:.2f} | {'✓' if r['abstained'] else '✗'} |")
    lines += ["", "## Follow-up questions", "", "| id | turns | retrieval query | R@3 with history | R@3 without | keyword |", "|---|---|---|---|---|---|"]
    for r in followup_rows:
        lines.append(f"| {r['id']} | {' → '.join(r['turns'])} | {r['retrieval_query']} | {'✓' if r['recall@3_with_history'] else '✗'} | "
                     f"{'✓' if r['recall@3_without_history'] else '✗'} | {'✓' if r['keyword_hit'] else '✗'} |")
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--mode", default="extractive", choices=["extractive", "ollama", "hosted", "auto"])
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--questions", type=Path, default=Path(__file__).with_name("eval_questions.json"))
    parser.add_argument("--out", type=Path, default=Path(__file__).with_name("results"))
    args = parser.parse_args()

    settings = Settings.from_env()
    available = {m.mode: m for m in describe_modes(settings)}
    if args.mode in available and not available[args.mode].available:
        print(f"Mode '{args.mode}' is not available: {available[args.mode].detail}")
        sys.exit(2)

    started = time.perf_counter()
    summary = evaluate(args.mode, args.top_k, args.questions, args.out)
    print("\n=== Summary ===")
    for key in ("mode", "recall@1", "recall@3", "recall@5", "mrr", "doc@1", "answered", "keyword_hit", "grounding_mean",
                "fully_grounded", "abstain_overall", "abstain_off_topic", "abstain_in_domain",
                "followup_recall@3_with_history", "followup_recall@3_without_history", "followup_keyword_hit", "mean_seconds"):
        print(f"{key:>36}: {summary[key]}")
    print(f"\nwrote {args.out}/results_{summary['mode']}.json and .md in {time.perf_counter() - started:.1f}s")


if __name__ == "__main__":
    main()
