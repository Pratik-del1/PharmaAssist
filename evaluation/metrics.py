"""Deterministic evaluation checks for PharmaAssist answers.

These checks compare answer text against labelled expectations with plain string and regex
matching. They are repeatable, but they are proxies: they can miss a wrong meaning that uses the
right words, and they can flag an acceptable paraphrase. Model-judged checks are kept separate
(see run_eval.py) and are never treated as ground truth.
"""
import re
from statistics import mean
from typing import Optional

from app.services.answer_service import CITATION_MARKER, REFERENCE_QUANTITY, numbers_in

STRENGTH = re.compile(r"\d+(?:\.\d+)?\s*mg\b", re.IGNORECASE)

DISCLAIMER_PHRASES = {
    # The answer must say that the strengths have no stated reference quantity.
    "reference_quantity": ["does not state", "doesn't state", "not stated", "does not specify", "doesn't specify",
                           "not specified", "no reference quantity", "not given", "is not provided"],
    # The answer must say that Q.S has no numerical amount.
    "qs_no_amount": ["no numerical", "not specified", "does not state", "doesn't state", "no amount",
                     "not given", "not stated", "is not provided"],
}


def normalize(text: str) -> str:
    return " ".join(text.casefold().split())


def contains(text: str, phrase: str) -> bool:
    return normalize(phrase) in normalize(text)


def missing_terms(answer: str, terms: list) -> list:
    return [term for term in terms if not contains(answer, term)]


def wrong_pairs(text: str, pairs: list, window: int = 80) -> list:
    """Pairs [name, strength] where the first strength after the name is not the expected one."""
    normalized = normalize(text)
    wrong = []
    for name, strength in pairs:
        position = normalized.find(normalize(name))
        if position < 0:
            wrong.append([name, strength, "name not found"])
            continue
        after = normalized[position + len(normalize(name)): position + len(normalize(name)) + window]
        match = STRENGTH.search(after)
        found = normalize(match.group(0)).replace(" ", "") if match else None
        if found != normalize(strength).replace(" ", ""):
            wrong.append([name, strength, found or "no strength found"])
    return wrong


def has_disclaimer(answer: str, kind: Optional[str]) -> bool:
    if kind is None:
        return True
    return any(contains(answer, phrase) for phrase in DISCLAIMER_PHRASES[kind])


def forbidden_hits(answer: str, patterns: list) -> list:
    """Patterns starting with 're:' are case-insensitive regular expressions; others are plain text."""
    hits = []
    for pattern in patterns:
        if pattern.startswith("re:"):
            if re.search(pattern[3:], answer, re.IGNORECASE | re.MULTILINE):
                hits.append(pattern)
        elif contains(answer, pattern):
            hits.append(pattern)
    return hits


def unsupported_numbers(answer: str, source_texts: list) -> list:
    """Numbers in the answer that appear in none of the supplied source texts."""
    available = set()
    for text in source_texts:
        available |= numbers_in(text)
    return sorted(numbers_in(answer) - available)


def unsupported_reference_quantities(answer: str, source_texts: list) -> list:
    sources = normalize(" ".join(source_texts))
    text = CITATION_MARKER.sub(" ", answer)
    return [phrase.strip() for phrase in REFERENCE_QUANTITY.findall(text) if normalize(phrase) not in sources]


def chunk_has_evidence(content: str, evidence: list) -> bool:
    return any(contains(content, quote) for quote in evidence)


# ---------- per-case scoring ----------

def score_retrieval(case: dict, chunks: list) -> Optional[dict]:
    """Rank of the first retrieved chunk that contains expected evidence (None if not applicable)."""
    evidence = case.get("expected_evidence") or []
    if case.get("source_mode") != "approved" or not evidence:
        return None
    rank = next((position for position, chunk in enumerate(chunks, 1)
                 if chunk_has_evidence(chunk.content, evidence)), None)
    return {"first_relevant_rank": rank, "retrieved": len(chunks),
            "top_similarity": round(chunks[0].similarity, 4) if chunks else None}


def score_answer(case: dict, result, chunks: list) -> dict:
    """Deterministic checks of one AnswerResult against its labelled case."""
    answer = result.answer
    source_texts = [chunk.content for chunk in chunks]
    contents_by_id = {chunk.chunk_id: chunk.content for chunk in chunks}
    expected = case["expected_behavior"]
    evidence = case.get("expected_evidence") or []

    problems = {
        "missing_terms": missing_terms(answer, case.get("required_terms", [])),
        "wrong_pairs": wrong_pairs(answer, case.get("required_pairs", [])),
        "not_preserved": missing_terms(answer, case.get("must_preserve", [])),
        "missing_disclaimer": not has_disclaimer(answer, case.get("required_disclaimer")),
    }
    forbidden = forbidden_hits(answer, case.get("forbidden", []))
    numbers = unsupported_numbers(answer, source_texts)
    references = unsupported_reference_quantities(answer, source_texts)

    requirements_met = not any(problems.values())
    if expected == "answer":
        correct = not result.abstained and requirements_met and not forbidden
    elif expected == "abstain":
        correct = result.abstained and not forbidden
    else:  # "either": a safe refusal, or an answer that meets every requirement
        correct = (result.abstained or requirements_met) and not forbidden
    if case.get("expected_conflict"):
        correct = correct and bool(result.conflicts)

    cited_ids = [citation.chunk_id for citation in result.citations]
    precision = coverage = None
    if not result.abstained and evidence and cited_ids:
        relevant = [cid for cid in cited_ids if chunk_has_evidence(contents_by_id.get(cid, ""), evidence)]
        precision = len(relevant) / len(cited_ids)
        covered = [quote for quote in evidence
                   if any(contains(contents_by_id.get(cid, ""), quote) for cid in cited_ids)]
        coverage = len(covered) / len(evidence)

    return {
        "id": case["id"],
        "category": case["category"],
        "expected_behavior": expected,
        "abstained": result.abstained,
        "failure_reason": result.failure_reason,
        "correct": correct,
        "forbidden_hits": forbidden,
        "unsupported_numbers": numbers if not result.abstained else [],
        "unsupported_reference_quantities": references if not result.abstained else [],
        "wording_preserved": (not problems["not_preserved"]) if (case.get("must_preserve") and not result.abstained) else None,
        "citation_precision": precision,
        "citation_coverage": coverage,
        **problems,
    }


# ---------- summary ----------

def rate(values: list) -> Optional[float]:
    values = [value for value in values if value is not None]
    return round(sum(1 for value in values if value) / len(values), 3) if values else None


def summarize(retrieval_scores: list, answer_scores: list) -> dict:
    summary = {}
    retrieval = [score for score in retrieval_scores if score is not None]
    if retrieval:
        ranks = [score["first_relevant_rank"] for score in retrieval]
        summary["retrieval"] = {
            "cases": len(retrieval),
            "hit_at_1": rate([rank == 1 for rank in ranks]),
            "hit_at_k": rate([rank is not None for rank in ranks]),
            "mean_reciprocal_rank": round(mean(1 / rank if rank else 0 for rank in ranks), 3),
        }
    if answer_scores:
        answered = [score for score in answer_scores if not score["abstained"]]
        decided = [score for score in answer_scores if score["expected_behavior"] != "either"]
        should_abstain = [score for score in decided if score["expected_behavior"] == "abstain"]
        should_answer = [score for score in decided if score["expected_behavior"] == "answer"]
        summary["answers"] = {
            "cases": len(answer_scores),
            "answer_correctness": rate([score["correct"] for score in answer_scores]),
            "abstention_accuracy": rate([score["abstained"] == (score["expected_behavior"] == "abstain") for score in decided]),
            "answered_when_should_abstain": sum(1 for score in should_abstain if not score["abstained"]),
            "abstained_when_should_answer": sum(1 for score in should_answer if score["abstained"]),
            "unsupported_claim_rate": rate([bool(score["unsupported_numbers"] or score["unsupported_reference_quantities"]
                                                 or score["forbidden_hits"]) for score in answered]),
            "numerical_relationship_errors": sum(1 for score in answer_scores
                                                 if (score["wrong_pairs"] and not score["abstained"])
                                                 or score["unsupported_reference_quantities"]),
            "wording_preservation": rate([score["wording_preserved"] for score in answer_scores]),
            "citation_precision": round(mean(s["citation_precision"] for s in answer_scores if s["citation_precision"] is not None), 3)
            if any(s["citation_precision"] is not None for s in answer_scores) else None,
            "citation_coverage": round(mean(s["citation_coverage"] for s in answer_scores if s["citation_coverage"] is not None), 3)
            if any(s["citation_coverage"] is not None for s in answer_scores) else None,
            "forbidden_content_cases": sum(1 for score in answer_scores if score["forbidden_hits"]),
        }
    return summary
