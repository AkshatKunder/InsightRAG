"""Evaluate ranked retrieval results against benchmark ground truth.

This module only *scores* results. It does not create retrievers, load models
or rerank anything: callers pass the ranked results for each question (or a
``retrieve_fn`` that returns them), and their order is kept exactly as given.

Relevance (see data/evaluation/retrieval_evaluation.md for the full rationale):

* Evidence unit = a distinct ``(source, page)`` pair. The ranked chunks are
  collapsed into a ranking of distinct units (each page at its first
  occurrence, order otherwise preserved; a chunk with unusable metadata is its
  own non-relevant unit). **K counts positions in that ranking**, so "top K"
  means the first K distinct pages, and all metrics use the same unit.
* A retrieved chunk is relevant when its source file name equals the
  benchmark's ``source_document`` and its page is one of the question's
  *relevant pages*: ``expected_pages`` plus every page listed in an
  ``expected_facts[].pages`` entry (the benchmark lists every page where a
  fact can be found, so a retriever is credited for any of them).
* Recall@K is fact-level: the share of ``expected_facts`` with at least one of
  their pages in the top K. Page recall (share of ``expected_pages`` retrieved)
  is reported alongside it.
* Precision@K = relevant units in the top K / min(K, number of distinct units).
* Reciprocal rank = 1 / position of the first relevant unit (0 if none).

Unanswerable questions have no evidence and are excluded from every average.
"""

from __future__ import annotations

import json
import numbers
from collections.abc import Callable, Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path, PurePath

from app.evaluation.retrieval_metrics import (
    deduplicate_ranking,
    first_relevant_rank,
    group_recall_at_k,
    mean_reciprocal_rank,
    normalise_k_values,
    precision_at_k,
    recall_at_k,
    reciprocal_rank,
)


DEFAULT_K_VALUES = (1, 3, 5, 10)
METRIC_NAMES = ("recall", "page_recall", "precision")

# Score-like fields copied from retriever output into the detailed results.
_SCORE_FIELDS = (
    "score",
    "rrf_score",
    "bm25_score",
    "bm25_rank",
    "vector_score",
    "vector_rank",
    "reranker_score",
    "final_score",
)
_PREVIEW_CHARS = 200

EvidenceKey = tuple[str, int]


# ---------------------------------------------------------------------------
# Benchmark loading
# ---------------------------------------------------------------------------


def load_benchmark(path: str | Path) -> dict:
    """Load a benchmark JSON file and check the fields the evaluator relies on."""

    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(data, dict) or not isinstance(data.get("questions"), list):
        raise ValueError("Benchmark must be a JSON object with a 'questions' list")
    if not isinstance(data.get("source_document"), str) or not data["source_document"].strip():
        raise ValueError("Benchmark must name its 'source_document'")
    return data


def split_questions(benchmark: Mapping) -> tuple[list[dict], list[dict]]:
    """Return (answerable, unanswerable) questions, preserving benchmark order."""

    answerable, unanswerable = [], []
    for question in benchmark["questions"]:
        (answerable if question.get("answerable") is True else unanswerable).append(question)
    return answerable, unanswerable


# ---------------------------------------------------------------------------
# Ground truth and relevance matching
# ---------------------------------------------------------------------------


def _is_page(value) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value >= 1


def evidence_key(result) -> EvidenceKey | None:
    """Map one retrieved result to ``(source file name, page)``.

    Returns None when the result has no usable source or page (missing,
    wrong type, page < 1). Such a result keeps its rank but is never relevant.
    """

    if not isinstance(result, Mapping):
        return None
    source, page = result.get("source"), result.get("page")
    if not isinstance(source, str) or not source.strip() or not _is_page(page):
        return None
    # Compare file names only, so "annual_report.pdf" and
    # "data/documents/annual_report.pdf" refer to the same document.
    return PurePath(source.replace("\\", "/")).name, page


def fact_page_groups(question: Mapping) -> list[tuple[str, list[int]]]:
    """Evidence groups for fact-level recall: ``[(label, pages), ...]``.

    Each expected fact is one group satisfied by any of its pages. A question
    without ``expected_facts`` falls back to one group per expected page.
    """

    groups = []
    for index, fact in enumerate(question.get("expected_facts") or []):
        pages = sorted({page for page in fact.get("pages", []) if _is_page(page)})
        if pages:
            label = fact.get("metric") or f"fact_{index + 1}"
            qualifiers = [str(fact[field]) for field in ("segment", "period") if fact.get(field)]
            if qualifiers:
                label = f"{label} ({', '.join(qualifiers)})"
            groups.append((label, pages))
    if not groups:
        groups = [(f"page_{page}", [page]) for page in sorted(set(question.get("expected_pages", [])))]
    return groups


def relevant_pages(question: Mapping) -> list[int]:
    """All pages that hold evidence for the question (expected + fact pages)."""

    pages = {page for page in question.get("expected_pages", []) if _is_page(page)}
    for _, group_pages in fact_page_groups(question):
        pages.update(group_pages)
    return sorted(pages)


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------


def _round(value: float) -> float:
    return round(value, 4)


def evaluate_question(
    question: Mapping,
    results: Sequence[Mapping],
    k_values: Sequence[int],
    source_document: str,
) -> dict:
    """Score one answerable question's ranked results at every K."""

    if question.get("answerable") is not True:
        raise ValueError(f"{question.get('id')}: only answerable questions can be scored")
    if not isinstance(results, Sequence) or isinstance(results, (str, bytes)):
        raise TypeError(f"{question.get('id')}: results must be a list of result dicts")

    k_values = normalise_k_values(k_values)
    source_name = PurePath(source_document.replace("\\", "/")).name

    expected_keys = {(source_name, page) for page in question.get("expected_pages", []) if _is_page(page)}
    relevant_page_list = relevant_pages(question)
    relevant_keys = {(source_name, page) for page in relevant_page_list}
    groups = fact_page_groups(question)
    group_keys = [{(source_name, page) for page in pages} for _, pages in groups]

    keys = [evidence_key(result) for result in results]

    # Ranking of distinct evidence units: the unit of K for every metric.
    ranking = deduplicate_ranking(keys)

    retrieved = []
    seen: set[EvidenceKey] = set()
    unit_rank = 0
    for chunk_rank, (result, key) in enumerate(zip(results, keys), start=1):
        duplicate = key is not None and key in seen
        if not duplicate:
            unit_rank += 1
            if key is not None:
                seen.add(key)
        entry = {
            "chunk_rank": chunk_rank,
            # Position in the distinct-unit ranking (what K counts); None for a duplicate chunk.
            "unit_rank": None if duplicate else unit_rank,
            "source": result.get("source") if isinstance(result, Mapping) else None,
            "page": result.get("page") if isinstance(result, Mapping) else None,
            "relevant": key is not None and key in relevant_keys,
            "duplicate": duplicate,
            "invalid_metadata": key is None,
        }
        if isinstance(result, Mapping):
            for field in _SCORE_FIELDS:
                value = result.get(field)
                # numbers.Real also covers numpy scalars (the reranker returns float32).
                if isinstance(value, numbers.Real) and not isinstance(value, bool):
                    entry[field] = float(value) if field.endswith("score") else int(value)
            text = result.get("text")
            if isinstance(text, str):
                entry["text_preview"] = " ".join(text.split())[:_PREVIEW_CHARS]
        retrieved.append(entry)

    rank = first_relevant_rank(keys, relevant_keys)
    metrics = {name: {} for name in METRIC_NAMES}
    for k in k_values:
        metrics["recall"][str(k)] = _round(group_recall_at_k(keys, group_keys, k))
        metrics["page_recall"][str(k)] = _round(recall_at_k(keys, expected_keys, k))
        metrics["precision"][str(k)] = _round(precision_at_k(keys, relevant_keys, k))

    found = {key for key in keys if key is not None}
    return {
        "id": question["id"],
        "category": question.get("category"),
        "question": question.get("question"),
        "expected_pages": sorted(page for _, page in expected_keys),
        "relevant_pages": relevant_page_list,
        "expected_facts": [
            {"label": label, "pages": pages, "retrieved": bool(keys_ & found)}
            for (label, pages), keys_ in zip(groups, group_keys)
        ],
        "num_results": len(results),
        "num_distinct_units": len(ranking),
        "num_duplicate_chunks": sum(1 for entry in retrieved if entry["duplicate"]),
        "num_invalid_metadata": sum(1 for key in keys if key is None),
        "retrieved_pages": [entry["page"] for entry in retrieved],
        "ranked_pages": [key[1] if key is not None else None for key in ranking],
        "relevant_pages_retrieved": [key[1] for key in ranking if key is not None and key in relevant_keys],
        "first_relevant_rank": rank,
        "reciprocal_rank": _round(reciprocal_rank(keys, relevant_keys)),
        "metrics": metrics,
        "retrieved": retrieved,
    }


def _mean(values: Sequence[float]) -> float:
    return _round(sum(values) / len(values)) if values else 0.0


def _aggregate(question_results: Sequence[Mapping], k_values: Sequence[int]) -> dict:
    aggregate = {
        "num_questions": len(question_results),
        "mrr": _round(mean_reciprocal_rank(r["reciprocal_rank"] for r in question_results)),
    }
    for name in METRIC_NAMES:
        aggregate[name] = {
            str(k): _mean([r["metrics"][name][str(k)] for r in question_results]) for k in k_values
        }
    return aggregate


def evaluate_results(
    benchmark: Mapping,
    results_by_id: Mapping[str, Sequence[Mapping]],
    k_values: Sequence[int] = DEFAULT_K_VALUES,
    config: Mapping | None = None,
) -> dict:
    """Score precomputed ranked results for every answerable benchmark question.

    ``results_by_id`` maps question id -> ranked results (dicts with at least
    ``source`` and ``page``). Every answerable question must be present;
    entries for unanswerable questions are ignored.

    Returns ``{"summary": {...}, "questions": [...]}``.
    """

    k_values = normalise_k_values(k_values)
    answerable, unanswerable = split_questions(benchmark)
    missing = [q["id"] for q in answerable if q["id"] not in results_by_id]
    if missing:
        raise KeyError(f"No retrieval results for answerable questions: {', '.join(missing)}")

    source_document = benchmark["source_document"]
    question_results = [
        evaluate_question(question, results_by_id[question["id"]], k_values, source_document)
        for question in answerable
    ]

    categories = list(dict.fromkeys(r["category"] for r in question_results))
    max_k = max(k_values)
    summary = {
        "benchmark_name": benchmark.get("benchmark_name"),
        "benchmark_version": benchmark.get("version"),
        "source_document": source_document,
        "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "config": dict(config or {}),
        "k_values": k_values,
        "num_questions_total": len(benchmark["questions"]),
        "num_answerable": len(answerable),
        "num_unanswerable": len(unanswerable),
        "num_evaluated": len(question_results),
        "excluded_unanswerable_ids": [q["id"] for q in unanswerable],
        "overall": _aggregate(question_results, k_values),
        "by_category": {
            category: _aggregate([r for r in question_results if r["category"] == category], k_values)
            for category in categories
        },
        "questions_with_no_relevant_result": [r["id"] for r in question_results if r["first_relevant_rank"] is None],
        "questions_with_fewer_than_max_k_units": [
            r["id"] for r in question_results if r["num_distinct_units"] < max_k
        ],
        "num_duplicate_chunks": sum(r["num_duplicate_chunks"] for r in question_results),
        "num_results_with_invalid_metadata": sum(r["num_invalid_metadata"] for r in question_results),
        "notes": (
            "Metrics cover answerable questions only; unanswerable questions are excluded "
            "and these numbers say nothing about abstention. recall = fact-level recall; "
            "page_recall = share of expected_pages retrieved; precision = relevant pages in "
            "top K / min(K, distinct pages returned). K counts distinct (source, page) units "
            "in first-occurrence order, not raw chunks; relevance is page-level."
        ),
    }
    return {"summary": summary, "questions": question_results}


def run_retrieval(
    benchmark: Mapping,
    retrieve_fn: Callable[[str], Sequence[Mapping]],
    progress: Callable[[str], None] | None = None,
) -> dict[str, list[dict]]:
    """Call ``retrieve_fn(question_text)`` for each answerable question.

    The returned lists are copied as-is; their order is the ranking that gets scored.
    """

    answerable, _ = split_questions(benchmark)
    results_by_id = {}
    for index, question in enumerate(answerable, start=1):
        if progress:
            progress(f"[{index}/{len(answerable)}] {question['id']}: {question['question']}")
        results = retrieve_fn(question["question"])
        if results is None:
            results = []
        results_by_id[question["id"]] = list(results)
    return results_by_id


def evaluate_benchmark(
    benchmark: Mapping,
    retrieve_fn: Callable[[str], Sequence[Mapping]],
    k_values: Sequence[int] = DEFAULT_K_VALUES,
    config: Mapping | None = None,
    progress: Callable[[str], None] | None = None,
) -> dict:
    """Run ``retrieve_fn`` on every answerable question and score the results."""

    k_values = normalise_k_values(k_values)  # fail fast before any retrieval
    results_by_id = run_retrieval(benchmark, retrieve_fn, progress)
    return evaluate_results(benchmark, results_by_id, k_values, config)
