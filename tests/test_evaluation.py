"""Tests for the evaluation dataset, the deterministic metrics and the runner. No model calls."""
import sys
from pathlib import Path

import pytest

import app.services.answer_service as answer_service
from app.services.answer_service import AnswerResult, Citation, generate_answer
from app.services.retrieval_service import RetrievedChunk
from evaluation import run_eval
from evaluation.metrics import (
    forbidden_hits,
    has_disclaimer,
    missing_terms,
    score_answer,
    summarize,
    unsupported_numbers,
    unsupported_reference_quantities,
    wrong_pairs,
)
from evaluation.run_eval import evaluate, load_cases

LEAFLET = Path(__file__).resolve().parents[1] / "data" / "raw" / "CZARLIV_SYRUP.pdf"
CASES = load_cases()
CASES_BY_ID = {case["id"]: case for case in CASES}

REQUIRED_CATEGORIES = {
    "answerable_ingredients", "answerable_strengths", "answerable_dosage", "answerable_storage",
    "qs_without_amount", "missing_reference_quantity", "strength_relationship", "dosage_interpretation",
    "storage_wording", "absent_contraindications", "absent_interactions", "absent_side_effects",
    "product_identity", "conflicting_evidence", "prompt_injection_in_source", "prompt_injection_in_question",
}


@pytest.fixture(autouse=True)
def no_real_groq(monkeypatch):
    def refuse(*args, **kwargs):
        raise AssertionError("tests must never create a real Groq client")
    monkeypatch.setattr(answer_service, "create_groq_client", refuse)
    monkeypatch.setattr(answer_service.groq, "Groq", refuse)


# ---------- the dataset ----------

def test_dataset_is_well_formed():
    ids = [case["id"] for case in CASES]
    assert len(ids) == len(set(ids)) == 31
    for case in CASES:
        assert case["source_mode"] in ("approved", "synthetic")
        assert case["expected_behavior"] in ("answer", "abstain", "either")
        assert case["question"].strip()
        if case["source_mode"] == "synthetic":
            assert case["synthetic_chunks"]
            assert all("fictional" in chunk["content"].lower() for chunk in case["synthetic_chunks"])
        else:
            assert "synthetic_chunks" not in case


def test_dataset_covers_every_required_category():
    assert REQUIRED_CATEGORIES <= {case["category"] for case in CASES}


@pytest.fixture(scope="module")
def leaflet_text():
    if not LEAFLET.exists():
        pytest.skip("data/raw/CZARLIV_SYRUP.pdf is not available on this machine")
    from app.services.pdf_extractor import extract_pdf_text
    from app.services.text_processing import clean_text
    return " ".join(clean_text(page["text"]) for page in extract_pdf_text(LEAFLET)["pages"])


def test_every_expectation_is_grounded_in_the_approved_leaflet(leaflet_text):
    """Expected evidence, preserved wording, required terms and strength pairs must all come from the source."""
    for case in CASES:
        if case["source_mode"] != "approved":
            continue
        for quote in case.get("expected_evidence", []) + case.get("must_preserve", []):
            assert not missing_terms(leaflet_text, [quote]), f"{case['id']}: '{quote}' is not in the leaflet"
        assert not missing_terms(leaflet_text, case.get("required_terms", [])), case["id"]
        assert wrong_pairs(leaflet_text, case.get("required_pairs", [])) == [], case["id"]


def test_forbidden_patterns_do_not_match_the_leaflet_itself(leaflet_text):
    # A forbidden pattern that matches the source would punish faithful quoting.
    for case in CASES:
        if case["source_mode"] == "approved":
            assert forbidden_hits(leaflet_text, case.get("forbidden", [])) == [], case["id"]


# ---------- deterministic metrics ----------

SOURCE = ("COMPOSITION :\nPicrorhiza Kurroa\nKutki\n200 mg\nEclipta Alba\nBhringraj\n100 mg\n"
          "Colour and Preservative\nQ.S\nDosage :\n10 ml twice a day or as directed\nby the physician\n"
          "Storage :\nStore below 25 degree C at\nroom temperature with max\ntemperature not exceeding 30\ndegree Celcius.")
CHUNKS = [RetrievedChunk(chunk_id=2, document_id=1, file_name="CZARLIV_SYRUP.pdf", document_type="pdf",
                         source_type="product_leaflet", page_number=1, content=SOURCE, similarity=0.6),
          RetrievedChunk(chunk_id=1, document_id=1, file_name="CZARLIV_SYRUP.pdf", document_type="pdf",
                         source_type="product_leaflet", page_number=1, content="CZARLIV SYRUP May be useful in",
                         similarity=0.7)]


def answered(text, cited=(2,)):
    return AnswerResult(answer=text, abstained=False, sources_used=len(cited),
                        citations=[Citation(chunk_id=c, document="CZARLIV_SYRUP.pdf", page=1, location="page 1")
                                   for c in cited])


def abstained(text="The approved product information does not contain enough information."):
    return AnswerResult(answer=text, abstained=True, sources_used=0, failure_reason="insufficient_evidence")


def test_strength_pairs_detect_a_wrong_relationship():
    assert wrong_pairs("Picrorhiza Kurroa (Kutki) 200 mg, Eclipta Alba (Bhringraj) 100 mg",
                       [["Picrorhiza Kurroa", "200 mg"], ["Eclipta Alba", "100 mg"]]) == []
    assert wrong_pairs("Eclipta Alba 200 mg", [["Eclipta Alba", "100 mg"]]) == [["Eclipta Alba", "100 mg", "200mg"]]
    assert wrong_pairs("Kutki is listed", [["Eclipta Alba", "100 mg"]])[0][2] == "name not found"


def test_reference_quantity_and_unsupported_numbers():
    assert unsupported_reference_quantities("Each 10 ml contains 200 mg of Kutki.", [SOURCE]) == ["Each 10 ml"]
    assert unsupported_reference_quantities("The dose is 10 ml twice a day.", [SOURCE]) == []
    assert unsupported_numbers("Kutki 300 mg [S1], page 1", [SOURCE]) == ["300"]


def test_disclaimers_and_forbidden_patterns():
    assert has_disclaimer("The leaflet does not state the reference quantity.", "reference_quantity")
    assert not has_disclaimer("Kutki 200 mg.", "reference_quantity")
    assert has_disclaimer("Listed as Q.S with no numerical amount.", "qs_no_amount")
    assert forbidden_hits("Store below 30 degree Celsius", ["Celsius", "re:\\bcures?\\b"]) == ["Celsius"]
    assert forbidden_hits("It cures diabetes", ["re:\\bcures?\\b"]) == ["re:\\bcures?\\b"]


def test_correct_storage_answer_scores_as_correct_with_full_citation_metrics():
    case = CASES_BY_ID["A04"]
    score = score_answer(case, answered(
        "The leaflet states: \"Store below 25 degree C at room temperature with max temperature not "
        "exceeding 30 degree Celcius.\" [S1]"), CHUNKS)
    assert score["correct"] and score["wording_preserved"]
    assert score["citation_precision"] == 1.0 and score["citation_coverage"] == 1.0
    assert score["unsupported_numbers"] == [] and score["forbidden_hits"] == []


def test_changed_storage_wording_is_flagged():
    score = score_answer(CASES_BY_ID["A04"], answered("Store below 30 degree Celsius."), CHUNKS)
    assert not score["correct"]
    assert score["wording_preserved"] is False
    assert "Celsius" in score["forbidden_hits"]


def test_citation_precision_counts_irrelevant_citations():
    case = CASES_BY_ID["A04"]
    score = score_answer(case, answered(
        "Store below 25 degree C at room temperature with max temperature not exceeding 30 degree Celcius.",
        cited=(2, 1)), CHUNKS)
    assert score["citation_precision"] == 0.5 and score["citation_coverage"] == 1.0


def test_invented_reference_quantity_counts_as_a_numerical_relationship_error():
    score = score_answer(CASES_BY_ID["A02"], answered("Each 10 ml contains Picrorhiza Kurroa 200 mg."), CHUNKS)
    assert not score["correct"]
    assert score["unsupported_reference_quantities"] == ["Each 10 ml"]
    summary = summarize([], [score])["answers"]
    assert summary["numerical_relationship_errors"] == 1 and summary["unsupported_claim_rate"] == 1.0


def test_abstention_scoring():
    assert score_answer(CASES_BY_ID["C01"], abstained(), CHUNKS)["correct"]
    wrong = score_answer(CASES_BY_ID["C01"], answered("Not for children under 12."), CHUNKS)
    assert not wrong["correct"]
    summary = summarize([], [wrong])["answers"]
    assert summary["answered_when_should_abstain"] == 1 and summary["abstention_accuracy"] == 0.0


def test_conflict_case_requires_a_reported_conflict():
    case = CASES_BY_ID["X01"]
    chunks = run_eval.synthetic_chunks(case)
    plain = abstained()
    assert not score_answer(case, plain, chunks)["correct"]
    with_conflict = plain.model_copy(update={"conflicts": ["25 vs 30 degree C"], "failure_reason": "conflicting_evidence"})
    assert score_answer(case, with_conflict, chunks)["correct"]


def test_either_cases_accept_a_safe_refusal_or_a_complete_answer():
    case = CASES_BY_ID["S01"]
    assert score_answer(case, abstained(), CHUNKS)["correct"]
    quoted = ("The leaflet states: \"Store below 25 degree C at room temperature with max temperature not "
              "exceeding 30 degree Celcius.\"")
    assert score_answer(case, answered(quoted), CHUNKS)["correct"]
    assert not score_answer(case, answered("Yes, " + quoted), CHUNKS)["correct"]


# ---------- the runner (no database, no model calls) ----------

def fake_retrieve(question):
    return CHUNKS


def test_runner_separates_deterministic_and_model_judged_results():
    report = evaluate(CASES, fake_retrieve, answer=lambda case, chunks: abstained())
    assert set(report) == {"deterministic", "model_judged", "cases"}
    assert report["model_judged"]["status"] == "not run"
    answers = report["deterministic"]["answers"]
    expected_correct = sum(1 for case in CASES if case["expected_behavior"] in ("abstain", "either")
                           and not case.get("expected_conflict"))
    assert answers["answer_correctness"] == round(expected_correct / len(CASES), 3)
    assert answers["abstained_when_should_answer"] == sum(1 for c in CASES if c["expected_behavior"] == "answer")
    assert answers["answered_when_should_abstain"] == 0


def test_runner_retrieval_metrics():
    report = evaluate(CASES, fake_retrieve)
    retrieval = report["deterministic"]["retrieval"]
    evidence_cases = [c for c in CASES if c["source_mode"] == "approved" and c.get("expected_evidence")]
    assert retrieval["cases"] == len(evidence_cases)
    # The fake chunks hold only part of the leaflet, so compute the expected hit rates from them.
    from evaluation.metrics import chunk_has_evidence
    hits_at_k = [any(chunk_has_evidence(chunk.content, c["expected_evidence"]) for chunk in CHUNKS) for c in evidence_cases]
    hits_at_1 = [chunk_has_evidence(CHUNKS[0].content, c["expected_evidence"]) for c in evidence_cases]
    assert retrieval["hit_at_k"] == round(sum(hits_at_k) / len(evidence_cases), 3)
    assert retrieval["hit_at_1"] == round(sum(hits_at_1) / len(evidence_cases), 3)
    assert "answers" not in report["deterministic"]


def test_runner_with_the_real_answer_service_and_a_fake_model():
    class FakeCompletions:
        calls = 0

        def create(self, **kwargs):
            FakeCompletions.calls += 1
            raise TimeoutError("timed out")

    class FakeChat:
        completions = FakeCompletions()

    class FakeClient:
        chat = FakeChat()

    def answer(case, chunks):
        return generate_answer(case["question"], chunks, known_products=["CZARLIV", "REPOB"],
                               client=FakeClient(), model="test-model")

    report = evaluate([CASES_BY_ID["P01"], CASES_BY_ID["A04"]], fake_retrieve, answer)
    rows = {row["id"]: row["answer"] for row in report["cases"]}
    assert rows["P01"]["failure_reason"] == "no_approved_source_for_product" and rows["P01"]["correct"]
    assert rows["A04"]["failure_reason"] == "model_unavailable" and not rows["A04"]["correct"]
    assert FakeCompletions.calls == 1   # REPOB never reached the model


def test_live_mode_requires_explicit_confirmation(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["run_eval", "--mode", "live"])
    with pytest.raises(SystemExit) as caught:
        run_eval.main()
    assert caught.value.code == 2


def test_recorded_mode_requires_an_answers_file(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["run_eval", "--mode", "recorded"])
    with pytest.raises(SystemExit):
        run_eval.main()


# ---------- diagnostics in the runner (no database, no model calls) ----------

import json  # noqa: E402

from evaluation.run_eval import summarize_diagnostics, write_jsonl  # noqa: E402


def test_diagnostics_summary_counts_outcomes_and_tokens():
    records = [
        {"case_id": "A01", "request_made": True, "outcome": "response", "finish_reason": "length",
         "json_parse": "invalid_json", "pydantic_validation": "failed", "failure_reason": "invalid_model_output",
         "tokens": {"prompt": 900, "completion": 1700, "reasoning": 340, "total": 2600, "cached": None}},
        {"case_id": "A03", "request_made": True, "outcome": "response", "finish_reason": "stop",
         "json_parse": "ok", "pydantic_validation": "ok", "failure_reason": None,
         "tokens": {"prompt": 880, "completion": 150, "reasoning": 200, "total": 1030, "cached": None}},
        {"case_id": "A05", "request_made": True, "outcome": "api_error", "http_status": 429, "error_class": "ClientError",
         "json_parse": "not_run", "pydantic_validation": "not_run", "failure_reason": "model_unavailable"},
        {"case_id": "P01", "request_made": False, "outcome": "skipped_product", "json_parse": "not_run",
         "pydantic_validation": "not_run", "failure_reason": "no_approved_source_for_product"},
    ]
    summary = summarize_diagnostics(records)
    assert summary["records"] == 4 and summary["requests_made"] == 3
    assert summary["outcomes"] == {"response": 2, "api_error": 1, "skipped_product": 1}
    assert summary["http_status"] == {"429": 1} and summary["error_class"] == {"ClientError": 1}
    assert summary["finish_reason"] == {"length": 1, "stop": 1}
    assert summary["token_limit_cases"] == ["A01"]
    assert summary["tokens"]["reasoning"] == {"min": 200, "max": 340, "mean": 270.0, "responses": 2}
    assert summary["failure_reason"]["model_unavailable"] == 1


def test_diagnostics_summary_handles_missing_metadata():
    summary = summarize_diagnostics([{"case_id": "X", "request_made": True, "outcome": "timeout",
                                      "tokens": {"prompt": None, "completion": None, "reasoning": None, "total": None}}])
    assert summary["tokens"] == {} and summary["finish_reason"] == {} and summary["outcomes"] == {"timeout": 1}


def test_diagnostics_file_has_one_record_per_case(tmp_path):
    class TimeoutClient:
        def __init__(self):
            self.chat = self
            self.completions = self
            self.calls = 0

        def create(self, **kwargs):
            self.calls += 1
            raise TimeoutError("timed out")

    client = TimeoutClient()
    records = []

    def answer(case, chunks):
        case_records = []
        result = generate_answer(case["question"], chunks, known_products=["CZARLIV", "REPOB"],
                                 client=client, model="test-model", diagnostics=case_records)
        records.append({"case_id": case["id"], **case_records[0]})
        return result

    evaluate([CASES_BY_ID["P01"], CASES_BY_ID["A04"]], fake_retrieve, answer)
    path = tmp_path / "live_diagnostics.jsonl"
    write_jsonl(path, records)
    lines = [json.loads(line) for line in path.read_text().splitlines()]
    assert [line["case_id"] for line in lines] == ["P01", "A04"]
    assert lines[0]["outcome"] == "skipped_product" and not lines[0]["request_made"]
    assert lines[1]["outcome"] == "timeout" and lines[1]["request_made"]
    assert client.calls == 1
    assert summarize_diagnostics(lines)["requests_made"] == 1


def test_save_diagnostics_is_only_allowed_in_live_mode(monkeypatch, tmp_path):
    monkeypatch.setattr(sys, "argv", ["run_eval", "--mode", "retrieval", "--save-diagnostics", str(tmp_path / "d.jsonl")])
    with pytest.raises(SystemExit) as caught:
        run_eval.main()
    assert caught.value.code == 2


def test_save_diagnostics_never_overwrites_an_existing_file(monkeypatch, tmp_path):
    existing = tmp_path / "existing_diagnostics.jsonl"
    existing.write_text("keep me\n")
    monkeypatch.setattr(sys, "argv", ["run_eval", "--mode", "live", "--confirm-live-groq-calls",
                                      "--save-diagnostics", str(existing)])
    with pytest.raises(SystemExit) as caught:
        run_eval.main()
    assert caught.value.code == 2
    assert existing.read_text() == "keep me\n"


# ---------- diagnostics failures never lose answers or the report ----------

def run_live_main(monkeypatch, tmp_path, diagnostics_path, client=None):
    """Run main() in live mode on P01 and A04 with fake retrieval and a fake (timing-out by default) Groq client."""
    import app.services.retrieval_service as retrieval_service

    class FakeConnection:
        def execute(self, sql):
            return self

        def fetchall(self):
            return [("CZARLIV", "CZARLIV SYRUP"), ("REPOB", "REPOB")]

        def close(self):
            pass

    class TimeoutClient:
        def __init__(self):
            self.chat = self
            self.completions = self
            self.calls = 0

        def create(self, **kwargs):
            self.calls += 1
            raise TimeoutError("timed out")

    client = client or TimeoutClient()
    real_generate_answer = answer_service.generate_answer
    monkeypatch.setattr(answer_service, "generate_answer", lambda question, chunks, **kwargs: real_generate_answer(
        question, chunks, client=client, model="test-model", **kwargs))
    monkeypatch.setattr(retrieval_service, "get_retrieval_settings", lambda: (5, None))
    monkeypatch.setattr(retrieval_service, "open_retrieval_connection", lambda: FakeConnection())
    monkeypatch.setattr(answer_service, "load_file_label_words", lambda connection: {})
    monkeypatch.setattr(retrieval_service, "search_chunks", lambda connection, question, **kwargs: fake_retrieve(question))

    dataset = tmp_path / "cases.jsonl"
    write_jsonl(dataset, [CASES_BY_ID["P01"], CASES_BY_ID["A04"]])
    answers_path, report_path = tmp_path / "answers.jsonl", tmp_path / "report.json"
    monkeypatch.setattr(sys, "argv", [
        "run_eval", "--mode", "live", "--confirm-live-groq-calls", "--dataset", str(dataset),
        "--save-answers", str(answers_path), "--output", str(report_path),
        "--save-diagnostics", str(diagnostics_path)])
    assert run_eval.main() == 0
    answers = [json.loads(line) for line in answers_path.read_text().splitlines()]
    report = json.loads(report_path.read_text())
    return answers, report, client


def test_a_failing_diagnostics_summary_keeps_answers_and_report(monkeypatch, tmp_path):
    def broken(records):
        raise RuntimeError("summary bug")
    monkeypatch.setattr(run_eval, "summarize_diagnostics", broken)
    diagnostics_path = tmp_path / "live_diagnostics.jsonl"
    answers, report, client = run_live_main(monkeypatch, tmp_path, diagnostics_path)
    assert [item["id"] for item in answers] == ["P01", "A04"]
    assert report["deterministic"]["answers"]["cases"] == 2
    assert report["diagnostics_error"] == "summary failed: RuntimeError"
    assert "diagnostics_summary" not in report
    assert len(diagnostics_path.read_text().splitlines()) == 2    # the records themselves are still saved
    assert client.calls == 1                                       # no additional model request


def test_a_failing_diagnostics_write_keeps_answers_and_report(monkeypatch, tmp_path):
    diagnostics_path = tmp_path / "missing_folder" / "live_diagnostics.jsonl"   # cannot be created
    answers, report, client = run_live_main(monkeypatch, tmp_path, diagnostics_path)
    assert [item["id"] for item in answers] == ["P01", "A04"]
    assert report["deterministic"]["answers"]["cases"] == 2
    assert report["diagnostics_error"] == "writing live_diagnostics.jsonl failed: FileNotFoundError"
    assert "diagnostics_file" not in report                        # never reported as saved
    assert report["diagnostics_summary"]["requests_made"] == 1
    assert not diagnostics_path.exists()
    assert client.calls == 1


def test_diagnostics_are_saved_and_reported_when_nothing_fails(monkeypatch, tmp_path):
    diagnostics_path = tmp_path / "live_diagnostics.jsonl"
    answers, report, client = run_live_main(monkeypatch, tmp_path, diagnostics_path)
    assert report["diagnostics_file"] == str(diagnostics_path) and "diagnostics_error" not in report
    assert report["diagnostics_summary"]["outcomes"] == {"skipped_product": 1, "timeout": 1}
    assert client.calls == 1


STORAGE_QUOTE = "Store below 25 degree C at\nroom temperature with max\ntemperature not exceeding 30\ndegree Celcius."
STORAGE_ANSWER = ("Store below 25 degree C at room temperature with max temperature not exceeding "
                  "30 degree Celcius [S1].")


class AnsweringClient:
    """A fake Groq client that returns one valid, grounded A04 storage answer."""
    def __init__(self):
        self.chat = self
        self.completions = self
        self.calls = 0

    def create(self, **kwargs):
        from groq.types.chat import ChatCompletion
        self.calls += 1
        content = json.dumps({"answer": STORAGE_ANSWER, "abstained": False, "cited_source_ids": ["S1"],
                              "facts": [{"statement": "Storage", "source_id": "S1", "quote": STORAGE_QUOTE}],
                              "missing_information": [], "conflicts": []})
        return ChatCompletion.model_validate({
            "id": "chatcmpl-1", "object": "chat.completion", "created": 1, "model": "test-model",
            "choices": [{"index": 0, "finish_reason": "stop", "message": {"role": "assistant", "content": content}}]})


def test_saved_answers_keep_the_validated_evidence_by_case_id(monkeypatch, tmp_path):
    answers, report, client = run_live_main(monkeypatch, tmp_path, tmp_path / "diagnostics.jsonl",
                                            client=AnsweringClient())
    saved = {item["id"]: item["result"] for item in answers}
    assert saved["A04"]["failure_reason"] is None
    assert saved["A04"]["evidence"] == [{"statement": "Storage", "source_id": "S1", "quote": STORAGE_QUOTE,
                                         "chunk_id": 2, "document": "CZARLIV_SYRUP.pdf"}]
    assert saved["P01"]["abstained"] and saved["P01"]["evidence"] == []     # refused before Groq
    assert "evidence" not in json.dumps(report)                              # not duplicated in the report
    diagnostics = [json.loads(line) for line in (tmp_path / "diagnostics.jsonl").read_text().splitlines()]
    assert all("evidence" not in record and "raw_text_excerpt" not in record for record in diagnostics)
    assert client.calls == 1


def test_evidence_does_not_change_scoring_and_old_answer_files_still_load():
    old_item = {"answer": STORAGE_ANSWER, "abstained": False, "sources_used": 1, "missing_information": [],
                "conflicts": [], "failure_reason": None,
                "citations": [{"chunk_id": 2, "document": "CZARLIV_SYRUP.pdf", "page": 1, "location": "page 1"}]}
    without_evidence = AnswerResult(**old_item)                              # saved before this change
    assert without_evidence.evidence == []
    with_evidence = AnswerResult(**old_item, evidence=[{"statement": "Storage", "source_id": "S1",
                                                        "quote": STORAGE_QUOTE, "chunk_id": 2,
                                                        "document": "CZARLIV_SYRUP.pdf"}])
    case = CASES_BY_ID["A04"]
    assert score_answer(case, with_evidence, CHUNKS) == score_answer(case, without_evidence, CHUNKS)


def test_token_limit_cases_also_recognise_older_gemini_records():
    summary = summarize_diagnostics([{"case_id": "A07", "finish_reason": "MAX_TOKENS"},
                                     {"case_id": "A08", "finish_reason": "length"},
                                     {"case_id": "A09", "finish_reason": "stop"}])
    assert summary["token_limit_cases"] == ["A07", "A08"]


def test_live_mode_requires_the_groq_confirmation_flag(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["run_eval", "--mode", "live", "--confirm-live-gemini-calls"])
    with pytest.raises(SystemExit) as caught:   # the old Gemini flag is no longer accepted
        run_eval.main()
    assert caught.value.code == 2
