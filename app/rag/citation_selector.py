"""General-purpose evidence selection for retrieved RAG chunks."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any


_TOKEN_PATTERN = re.compile(r"[a-zA-Z][a-zA-Z0-9-]*|\d+(?:,\d{3})*(?:\.\d+)?")
_NUMBER_PATTERN = re.compile(
    r"(?<!\w)(?:[$€£]\s*)?(\d[\d,]*(?:\.\d+)?)(?:\s*)"
    r"(million|billion|thousand|millions|billions|thousands|bn|m|k|%)?"
    r"(?=$|[^\w])",
    re.IGNORECASE,
)
_STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "did", "does",
    "for", "from", "had", "has", "have", "how", "in", "is", "it", "of",
    "on", "or", "the", "to", "was", "were", "what", "when", "where", "which",
    "who", "with", "would", "company", "report", "annual",
}
_UNIT_MULTIPLIERS = {
    "billion": 1_000_000_000,
    "bn": 1_000_000_000,
    "million": 1_000_000,
    "m": 1_000_000,
    "thousand": 1_000,
    "k": 1_000,
}


@dataclass(frozen=True)
class EvidenceScoreConfig:
    """Weights and threshold used by :func:`select_citations`."""

    retrieval_weight: float = 0.30
    question_relevance_weight: float = 0.25
    answer_relevance_weight: float = 0.25
    numerical_weight: float = 0.15
    period_weight: float = 0.05
    min_relevance_threshold: float = 0.35


DEFAULT_CONFIG = EvidenceScoreConfig()


def _tokens(text: str) -> set[str]:
    return {
        token.lower()
        for token in _TOKEN_PATTERN.findall(text)
        if token.lower() not in _STOPWORDS and len(token) > 1
    }


def _extract_entities(text: str) -> set[str]:
    """Return normalized meaningful terms for backwards-compatible inspection."""
    return _tokens(text)


def _normalize_unit(unit: str | None) -> str:
    unit = (unit or "").lower().rstrip("s")
    return {"bn": "billion", "m": "million", "k": "thousand"}.get(unit, unit)


def _extract_numbers_with_context(text: str) -> dict[str, str]:
    """Map normalized numbers-with-units to surrounding text snippets."""
    values: dict[str, str] = {}
    for match in _NUMBER_PATTERN.finditer(text):
        raw_value, raw_unit = match.groups()
        value = float(raw_value.replace(",", ""))
        unit = _normalize_unit(raw_unit)
        key = f"{value:g}:{unit}"
        start = max(0, match.start() - 90)
        end = min(len(text), match.end() + 90)
        values[key] = text[start:end]
    return values


def _numbers_equivalent(key1: str, key2: str) -> bool:
    """Return whether two numeric keys represent the same quantity."""
    try:
        value1, unit1 = (key1.split(":", 1) + [""])[:2]
        value2, unit2 = (key2.split(":", 1) + [""])[:2]
        base1 = float(value1) * _UNIT_MULTIPLIERS.get(unit1, 1)
        base2 = float(value2) * _UNIT_MULTIPLIERS.get(unit2, 1)
    except ValueError:
        return False

    return abs(base1 - base2) / max(abs(base1), abs(base2), 1.0) < 0.01


def _compute_retrieval_score(chunk: dict[str, Any]) -> float:
    """Read the best available retrieval signal without requiring every field."""
    for field in ("final_score", "reranker_score", "rrf_score", "score", "vector_score"):
        value = chunk.get(field)
        if value is not None:
            try:
                return max(0.0, min(float(value), 1.0))
            except (TypeError, ValueError):
                continue

    value = chunk.get("bm25_score")
    try:
        return max(0.0, min(float(value) / 20.0, 1.0))
    except (TypeError, ValueError):
        return 0.0


def _compute_answer_relevance(answer: str, chunk_text: str) -> float:
    """Measure coverage of meaningful answer terms by an evidence chunk."""
    answer_terms = _tokens(answer)
    if not answer_terms:
        return 0.0
    return len(answer_terms & _tokens(chunk_text)) / len(answer_terms)


def _compute_question_relevance(question: str, chunk_text: str) -> float:
    """Measure coverage of meaningful question terms by an evidence chunk."""
    question_terms = _tokens(question)
    if not question_terms:
        return 0.0
    return len(question_terms & _tokens(chunk_text)) / len(question_terms)


def _compute_numerical_consistency(answer: str, chunk_text: str) -> float:
    """Score answer-number support, including equivalent million/billion values."""
    answer_numbers = _extract_numbers_with_context(answer)
    if not answer_numbers:
        return 0.5

    chunk_numbers = _extract_numbers_with_context(chunk_text)
    if not chunk_numbers:
        return 0.0

    matches = 0
    for answer_key in answer_numbers:
        if any(_numbers_equivalent(answer_key, key) for key in chunk_numbers):
            matches += 1
    return matches / len(answer_numbers)


def _compute_period_consistency(question: str, answer: str, chunk_text: str) -> float:
    """Prefer a chunk that contains the reporting period requested or answered."""
    periods = {
        token for token in _tokens(f"{question} {answer}")
        if re.fullmatch(r"(?:19|20)\d{2}", token)
    }
    if not periods:
        return 1.0
    return 1.0 if periods & _tokens(chunk_text) else 0.5


def _apply_contextual_penalty(question: str, chunk_text: str, answer: str) -> float:
    """Penalize known mismatched contexts without document-specific rules."""
    question_lower = question.lower()
    text = chunk_text.lower()

    if any(term in question_lower for term in ("sales", "revenue", "income")):
        if ("sale of" in text and any(word in text for word in ("facility", "plant", "asset"))) or "divest" in text:
            return 0.15
    if any(term in question_lower for term in ("employee", "workforce", "staff")):
        if "remuneration" in text or ("board" in text and "director" in text):
            return 0.20
    if any(term in question_lower for term in ("country", "countries", "nation")):
        if "customer" in text or ("product" in text and "operation" not in text):
            return 0.35
    return 1.0


def _build_selected_chunk(chunk: dict[str, Any], evidence_score: float, signals: dict[str, float]) -> dict[str, Any]:
    """Copy original metadata and attach the evidence score without mutating input."""
    selected = dict(chunk)
    selected["source"] = chunk.get("source", "Unknown source")
    selected["page"] = chunk.get("page")
    selected["text"] = chunk.get("text", "")
    selected["final_score"] = evidence_score
    selected["evidence_signals"] = signals
    return selected


def select_citations(
    question: str,
    answer: str,
    retrieved_chunks: list[dict[str, Any]],
    max_sources: int = 3,
    min_relevance_threshold: float | None = None,
    config: EvidenceScoreConfig | None = None,
) -> list[dict[str, Any]]:
    """Select direct, de-duplicated evidence without document-specific rules."""
    if not question or not answer or not retrieved_chunks or max_sources <= 0:
        return []

    cfg = config or DEFAULT_CONFIG
    threshold = cfg.min_relevance_threshold if min_relevance_threshold is None else min_relevance_threshold
    candidates: list[dict[str, Any]] = []
    seen_pages: set[tuple[str, Any]] = set()

    for chunk in retrieved_chunks:
        text = str(chunk.get("text") or "").strip()
        source = str(chunk.get("source") or "Unknown source")
        page_key = (source, chunk.get("page"))
        if not text or page_key in seen_pages:
            continue

        retrieval = _compute_retrieval_score(chunk)
        question_relevance = _compute_question_relevance(question, text)
        answer_relevance = _compute_answer_relevance(answer, text)
        numerical = _compute_numerical_consistency(answer, text)
        period = _compute_period_consistency(question, answer, text)
        penalty = _apply_contextual_penalty(question, text, answer)
        evidence_score = penalty * (
            cfg.retrieval_weight * retrieval
            + cfg.question_relevance_weight * question_relevance
            + cfg.answer_relevance_weight * answer_relevance
            + cfg.numerical_weight * numerical
            + cfg.period_weight * period
        )

        seen_pages.add(page_key)
        if evidence_score < threshold:
            continue

        signals = {
            "retrieval": retrieval,
            "question_relevance": question_relevance,
            "answer_relevance": answer_relevance,
            "numerical_consistency": numerical,
            "period_consistency": period,
            "context_penalty": penalty,
        }
        candidates.append(_build_selected_chunk(chunk, evidence_score, signals))

    candidates.sort(key=lambda item: item["final_score"], reverse=True)
    return candidates[:max_sources]
