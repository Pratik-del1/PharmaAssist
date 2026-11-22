"""Run the PharmaAssist evaluation set.

Modes:
  retrieval  Search the real approved knowledge base (application role, read-only). No model calls.
  recorded   Score previously saved answers (JSONL: {"id": ..., "result": {...AnswerResult...}}).
             No model calls.
  live       Generate answers with Groq. Needs --confirm-live-groq-calls and GROQ settings in .env.

Examples:
  python -m evaluation.run_eval --mode retrieval
  python -m evaluation.run_eval --mode recorded --answers evaluation/answers.jsonl
  python -m evaluation.run_eval --mode live --confirm-live-groq-calls --save-answers answers.jsonl \
      --save-diagnostics evaluation/results/live_diagnostics.jsonl

Diagnostics files (one record per case: HTTP status, finish reason, token usage, JSON and validation
outcomes) are git-ignored and never overwritten. They never contain prompts, source chunks or API keys.

Deterministic checks are reported under "deterministic". Model-judged checks are reported
separately and are never treated as ground truth.
"""
import argparse
import json
import sys
from pathlib import Path
from typing import Callable, Optional

from evaluation.metrics import score_answer, score_retrieval, summarize
from app.services.retrieval_service import RetrievedChunk

DATASET = Path(__file__).resolve().parent / "czarliv_eval.jsonl"
SYNTHETIC_CHUNK_ID_START = 900001   # never collides with real chunk ids in this small database

MODEL_JUDGED_NOT_RUN = {
    "status": "not run",
    "note": "No model judge is configured. Model-judged scores (for example faithfulness rated by an LLM) "
            "would be reported here separately and are not ground truth.",
}


def load_cases(path: Path = DATASET) -> list:
    with open(path, encoding="utf-8") as file:
        return [json.loads(line) for line in file if line.strip()]


def synthetic_chunks(case: dict) -> list:
    chunks = []
    for offset, source in enumerate(case.get("synthetic_chunks", [])):
        chunks.append(RetrievedChunk(
            chunk_id=SYNTHETIC_CHUNK_ID_START + offset, document_id=0, file_name=source["file_name"],
            document_type="pdf", source_type="product_leaflet", page_number=1, content=source["content"],
            product_name=source.get("product_name"), similarity=0.0))
    return chunks


def evaluate(cases: list, retrieve: Callable, answer: Optional[Callable] = None) -> dict:
    """retrieve(question) -> chunks; answer(case, chunks) -> AnswerResult (None = retrieval only)."""
    retrieval_scores, answer_scores, rows = [], [], []
    for case in cases:
        chunks = synthetic_chunks(case) if case["source_mode"] == "synthetic" else retrieve(case["question"])
        retrieval = score_retrieval(case, chunks)
        retrieval_scores.append(retrieval)
        row = {"id": case["id"], "category": case["category"], "retrieval": retrieval,
               "top_similarity": round(chunks[0].similarity, 4) if chunks and case["source_mode"] == "approved" else None}
        if answer is not None:
            result = answer(case, chunks)
            if result is None:
                row["answer"] = "missing"
            else:
                scored = score_answer(case, result, chunks)
                answer_scores.append(scored)
                row["answer"] = scored
        rows.append(row)
    return {
        "deterministic": summarize(retrieval_scores, answer_scores),
        "model_judged": MODEL_JUDGED_NOT_RUN,
        "cases": rows,
    }


def similarity_by_expectation(cases: list, rows: list) -> dict:
    """Top similarity per expected behaviour: information for choosing a threshold later, not a threshold."""
    groups = {}
    for case, row in zip(cases, rows):
        if row["top_similarity"] is not None:
            groups.setdefault(case["expected_behavior"], []).append(row["top_similarity"])
    return {behaviour: {"min": min(values), "max": max(values), "cases": len(values)}
            for behaviour, values in groups.items()}


def summarize_diagnostics(records: list) -> dict:
    """Counts and token statistics from per-case diagnostic records (live mode)."""
    def count(field):
        counts = {}
        for record in records:
            value = record.get(field)
            if value is not None:
                counts[str(value)] = counts.get(str(value), 0) + 1
        return counts

    token_stats = {}
    for name in ("prompt", "completion", "reasoning", "total"):
        values = [record["tokens"][name] for record in records
                  if isinstance(record.get("tokens"), dict) and isinstance(record["tokens"].get(name), int)]
        if values:
            token_stats[name] = {"min": min(values), "max": max(values),
                                 "mean": round(sum(values) / len(values), 1), "responses": len(values)}
    return {
        "records": len(records),
        "requests_made": sum(1 for record in records if record.get("request_made")),
        "outcomes": count("outcome"),
        "http_status": count("http_status"),
        "error_class": count("error_class"),
        "finish_reason": count("finish_reason"),
        "json_parse": count("json_parse"),
        "pydantic_validation": count("pydantic_validation"),
        "failure_reason": count("failure_reason"),
        # "length": Groq stopped at max_completion_tokens (Gemini, in older files, used "MAX_TOKENS").
        "token_limit_cases": [record["case_id"] for record in records
                              if record.get("finish_reason") in ("length", "MAX_TOKENS")],
        "tokens": token_stats,
    }


def write_jsonl(path: Path, items: list) -> None:
    with open(path, "w", encoding="utf-8") as file:
        for item in items:
            file.write(json.dumps(item, default=str) + "\n")


def write_report(path: Path, report: dict) -> None:
    path.write_text(json.dumps(report, indent=2, default=str))


def add_diagnostics(report: dict, records: list, path: Optional[Path]) -> None:
    """Summarise and save the diagnostic records. A failure is noted in the report, never raised."""
    problems = []
    try:
        report["diagnostics_summary"] = summarize_diagnostics(records)
    except Exception as error:
        problems.append(f"summary failed: {type(error).__name__}")
    if path:
        try:
            write_jsonl(path, records)
            report["diagnostics_file"] = str(path)
        except Exception as error:
            problems.append(f"writing {path.name} failed: {type(error).__name__}")
    if problems:
        report["diagnostics_error"] = "; ".join(problems)
        print("Warning: " + report["diagnostics_error"], file=sys.stderr)


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the PharmaAssist evaluation set.")
    parser.add_argument("--mode", choices=["retrieval", "recorded", "live"], required=True)
    parser.add_argument("--dataset", type=Path, default=DATASET)
    parser.add_argument("--answers", type=Path, help="Recorded answers (JSONL) for --mode recorded")
    parser.add_argument("--save-answers", type=Path, help="Where to save live answers (JSONL)")
    parser.add_argument("--output", type=Path, help="Write the full JSON report here")
    parser.add_argument("--save-diagnostics", type=Path,
                        help="Live mode only: write one diagnostic record per case (JSONL). Never overwrites.")
    parser.add_argument("--confirm-live-groq-calls", action="store_true",
                        help="Required for --mode live: sends questions and approved chunks to Groq")
    args = parser.parse_args()

    if args.mode == "live" and not args.confirm_live_groq_calls:
        parser.error("--mode live sends data to Groq; add --confirm-live-groq-calls to proceed")
    if args.mode == "recorded" and not args.answers:
        parser.error("--mode recorded needs --answers")
    if args.save_diagnostics and args.mode != "live":
        parser.error("--save-diagnostics is only available with --mode live")
    if args.save_diagnostics and args.save_diagnostics.exists():
        parser.error(f"{args.save_diagnostics} already exists; choose a new file name")

    from app.services.retrieval_service import get_retrieval_settings, open_retrieval_connection, search_chunks
    cases = load_cases(args.dataset)
    top_k, min_similarity = get_retrieval_settings()
    connection = open_retrieval_connection()
    try:
        known_products = [value for row in connection.execute(
            "SELECT product_id, product_name FROM products").fetchall() for value in row]

        def retrieve(question):
            return search_chunks(connection, question, top_k=top_k, min_similarity=min_similarity)

        answer = None
        if args.mode == "recorded":
            from app.services.answer_service import AnswerResult
            recorded = {}
            with open(args.answers, encoding="utf-8") as file:
                for line in file:
                    if line.strip():
                        item = json.loads(line)
                        recorded[item["id"]] = AnswerResult(**item["result"])
            answer = lambda case, chunks: recorded.get(case["id"])
        elif args.mode == "live":
            from app.services.answer_service import generate_answer, load_file_label_words
            file_label_words = load_file_label_words(connection)
            saved = []
            diagnostic_records = []

            def answer(case, chunks):
                case_records = []
                result = generate_answer(case["question"], chunks, known_products=known_products,
                                         diagnostics=case_records, file_label_words=file_label_words)
                saved.append({"id": case["id"], "result": result.model_dump()})
                diagnostic_records.append({"case_id": case["id"],
                                           **(case_records[0] if case_records else {"outcome": "not_recorded"})})
                return result

        report = evaluate(cases, retrieve, answer)
        report["settings"] = {"top_k": top_k, "min_similarity": min_similarity, "mode": args.mode}
        report["top_similarity_by_expected_behavior"] = similarity_by_expectation(cases, report["cases"])
    finally:
        connection.close()

    # Answers and the report are saved first, so a diagnostics failure can never lose them.
    if args.mode == "live" and args.save_answers:
        write_jsonl(args.save_answers, saved)
    if args.output:
        write_report(args.output, report)
    if args.mode == "live":
        add_diagnostics(report, diagnostic_records, args.save_diagnostics)
        if args.output:
            write_report(args.output, report)   # same file again, now with the diagnostics fields

    summary_keys = ["settings", "deterministic", "model_judged", "top_similarity_by_expected_behavior"]
    for key in ("diagnostics_summary", "diagnostics_file", "diagnostics_error"):
        if key in report:
            summary_keys.append(key)
    print(json.dumps({key: report[key] for key in summary_keys}, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
