"""Run the production question-answering pipeline on benchmark questions.

The steps and settings are those of ``app/main.py``:

1. hybrid retrieval, top 20 pages from 50 candidates per retriever;
2. cross-encoder reranking, top 5 (the generator's context);
3. answer generation with Ollama ``llama3.2:3b``;
4. citation selection (max 3 sources);
5. claim-level citation validation (``SUPPORTED`` / ``UNSUPPORTED`` /
   ``INSUFFICIENT_EVIDENCE`` / ``ABSTAINED``).

Components are injected, so the runner is testable with fakes. A failure in
any step is recorded on that question (``error``) and never skipped.
The generator is used exactly as in production (no temperature or seed
override); repeat runs measure how much its sampling changes the results.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

RETRIEVE_TOP_K = 20
RETRIEVE_CANDIDATE_K = 50
CONTEXT_SIZE = 5
MAX_CITATIONS = 3


@dataclass
class PipelineComponents:
    retrieve: Callable[[str], Sequence[Mapping]]
    rerank: Callable[[str, list], Sequence[Mapping]]
    generate: Callable[[str, list], str]
    select_citations: Callable[[str, str, list], Sequence[Mapping]]
    validate: Callable[[str, list], Mapping]
    description: dict = field(default_factory=dict)


def _chunk(item: Mapping, keep_text: bool = True) -> dict:
    out = {"source": item.get("source"), "page": item.get("page")}
    for key in ("score", "reranker_score", "final_score"):
        value = item.get(key)
        if value is not None:
            out[key] = float(value)
    if keep_text:
        out["text"] = item.get("text", "")
    return out


def _validation(result: Mapping) -> dict:
    claims = [
        {
            "claim_text": claim.get("claim_text"),
            "status": claim.get("status"),
            "source": claim.get("source"),
            "page": claim.get("page"),
            "explanation": claim.get("explanation"),
        }
        for claim in result.get("claims", [])
    ]
    return {
        "overall_status": result.get("overall_status"),
        "is_valid": result.get("is_valid"),
        "reason": result.get("reason"),
        "claims": claims,
        "unsupported_numbers": list(result.get("unsupported_numbers", [])),
    }


def run_question(question: str, components: PipelineComponents) -> dict:
    record = {"answer": None, "context": [], "citations": [], "validation": None, "error": None, "timings": {}}
    try:
        start = time.perf_counter()
        candidates = list(components.retrieve(question))
        context = list(components.rerank(question, candidates))
        record["context"] = [_chunk(item) for item in context]
        record["timings"]["retrieval_s"] = round(time.perf_counter() - start, 3)

        start = time.perf_counter()
        answer = components.generate(question, context)
        record["answer"] = answer
        record["timings"]["generation_s"] = round(time.perf_counter() - start, 3)

        citations = list(components.select_citations(question, answer, context))
        record["citations"] = [_chunk(item) for item in citations]
        record["validation"] = _validation(components.validate(answer, citations))
    except Exception as error:  # recorded explicitly, never skipped
        logger.exception("pipeline failed for %r", question)
        record["error"] = f"{type(error).__name__}: {error}"
    return record


def run_benchmark(
    benchmark: Mapping,
    components: PipelineComponents,
    repeats: int = 1,
    progress: Callable[[str], None] | None = None,
) -> dict:
    """Answer every benchmark question ``repeats`` times. Returns a JSON-serialisable record."""

    if repeats < 1:
        raise ValueError("repeats must be >= 1")
    questions = benchmark["questions"]
    runs = []
    for run_index in range(1, repeats + 1):
        records = []
        for index, question in enumerate(questions, start=1):
            if progress:
                progress(f"run {run_index}/{repeats} [{index}/{len(questions)}] {question['id']}")
            record = run_question(question["question"], components)
            records.append({"id": question["id"], "question": question["question"], **record})
        runs.append({"run": run_index, "questions": records})
    return {
        "config": {
            "repeats": repeats,
            "retrieve_top_k": RETRIEVE_TOP_K,
            "retrieve_candidate_k": RETRIEVE_CANDIDATE_K,
            "context_size": CONTEXT_SIZE,
            "max_citations": MAX_CITATIONS,
            **components.description,
        },
        "runs": runs,
        "errors": [
            {"run": run["run"], "id": record["id"], "error": record["error"]}
            for run in runs
            for record in run["questions"]
            if record["error"]
        ],
    }
