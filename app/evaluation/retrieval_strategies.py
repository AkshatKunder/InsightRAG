"""Run several retrieval strategies on benchmark questions with a shared, controlled setup.

Strategies (names used throughout the results):

=====================  =========================================================
``bm25``               BM25 over the chunked PDF, query passed through the
                       project's ``expand_query`` (as the hybrid retriever does)
``vector``             Qdrant vector search (bge-small), same expanded query
``hybrid``             ``HybridRetriever.search``: weighted RRF of BM25 + vector
                       with page diversity (the production first stage)
``<base>+rerank``      the base strategy's pool reordered by the production
                       ``Reranker`` (0.7 * normalised first-stage score
                       + 0.3 * normalised cross-encoder score)
``<base>+ce``          the same pool ordered by the cross-encoder score alone
                       (diagnostic; taken from the same reranker call)
=====================  =========================================================

Controls that keep the comparison fair:

* One query per question for all base strategies: BM25 and vector receive the
  same expanded query the hybrid retriever builds internally, so the only
  difference between them and ``hybrid`` is the fusion.
* Every strategy is reduced to a **pool of the top ``pool_size`` distinct
  (source, page) units** (first chunk per page kept, order preserved). This is
  the same unit the metrics use, so all strategies are scored at the same
  depth. BM25 and vector fetch more chunks (doubling from ``candidate_k`` up to
  ``max_candidate_k``) until the pool is full, because several of their chunks
  can come from one page. The hybrid retriever is page-diverse already and
  is asked for ``pool_size`` pages from ``candidate_k`` candidates per
  retriever, exactly as ``app/main.py`` does.
* Reranked variants reorder exactly the base pool (identical candidates), so
  before/after differences come from reordering only.

Model-backed components are injected as callables (``RetrievalComponents``),
so everything here runs with fakes in unit tests.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import PurePath

logger = logging.getLogger(__name__)

BASE_STRATEGIES = ("bm25", "vector", "hybrid")
RERANK_SUFFIX = "+rerank"
CE_SUFFIX = "+ce"

DEFAULT_CANDIDATE_K = 50
DEFAULT_MAX_CANDIDATE_K = 400
DEFAULT_POOL_SIZE = 20

_KEEP_FIELDS = (
    "source", "page", "score", "rrf_score", "bm25_score", "bm25_rank",
    "vector_score", "vector_rank", "reranker_score", "final_score",
)
_PREVIEW_CHARS = 200


def strategy_names(include_rerank: bool = True) -> list[str]:
    names = list(BASE_STRATEGIES)
    if include_rerank:
        names += [base + RERANK_SUFFIX for base in BASE_STRATEGIES]
        names += [base + CE_SUFFIX for base in BASE_STRATEGIES]
    return names


@dataclass
class RetrievalComponents:
    """Callables wrapping the project's retrievers (real or fake)."""

    bm25_search: Callable[[str, int], Sequence[Mapping]]
    vector_search: Callable[[str, int], Sequence[Mapping]]
    hybrid_search: Callable[[str, int, int], Sequence[Mapping]]
    expand_query: Callable[[str], str] = lambda query: query
    rerank: Callable[[str, list, int], Sequence[Mapping]] | None = None
    description: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Page collapsing
# ---------------------------------------------------------------------------


def _page_key(result) -> tuple | None:
    if not isinstance(result, Mapping):
        return None
    source, page = result.get("source"), result.get("page")
    if not isinstance(source, str) or not source.strip():
        return None
    if isinstance(page, bool) or not isinstance(page, int) or page < 1:
        return None
    return PurePath(source.replace("\\", "/")).name, page


def collapse_to_pages(results: Sequence[Mapping], limit: int | None = None) -> list[dict]:
    """Keep the first result per (source, page), preserving order; truncate to ``limit`` units.

    Results with unusable metadata are kept (each as its own unit) so that
    they still occupy a rank, matching the metric semantics.
    """

    seen = set()
    pool = []
    for result in results:
        key = _page_key(result)
        if key is not None:
            if key in seen:
                continue
            seen.add(key)
        pool.append(dict(result) if isinstance(result, Mapping) else {"invalid": repr(result)})
        if limit is not None and len(pool) >= limit:
            break
    return pool


def _distinct_units(results: Sequence[Mapping]) -> int:
    return len(collapse_to_pages(results))


def _slim(result: Mapping, preview: bool = True) -> dict:
    item = {key: result[key] for key in _KEEP_FIELDS if key in result}
    for key, value in list(item.items()):
        if hasattr(value, "item"):  # numpy scalar
            item[key] = value.item()
    text = result.get("text")
    if preview and isinstance(text, str):
        item["text_preview"] = " ".join(text.split())[:_PREVIEW_CHARS]
    return item


# ---------------------------------------------------------------------------
# Pool construction
# ---------------------------------------------------------------------------


def _fetch_page_pool(search, query, candidate_k, max_candidate_k, pool_size) -> tuple[list[dict], dict]:
    """Fetch chunks, doubling the depth until ``pool_size`` distinct pages are found."""

    depth = candidate_k
    while True:
        chunks = list(search(query, depth) or [])
        distinct = _distinct_units(chunks)
        exhausted = len(chunks) < depth  # retriever returned everything it has
        if distinct >= pool_size or exhausted or depth >= max_candidate_k:
            break
        depth = min(depth * 2, max_candidate_k)
    pool = collapse_to_pages(chunks, pool_size)
    return pool, {
        "chunks_fetched": len(chunks),
        "chunk_depth_requested": depth,
        "distinct_pages_available": distinct,
        "pool_size": len(pool),
    }


def build_base_pools(
    question: str,
    components: RetrievalComponents,
    candidate_k: int = DEFAULT_CANDIDATE_K,
    pool_size: int = DEFAULT_POOL_SIZE,
    max_candidate_k: int = DEFAULT_MAX_CANDIDATE_K,
) -> dict[str, dict]:
    """Return ``{strategy: {"results": [...], "diagnostics": {...}, "error": str|None}}`` for the base strategies."""

    expanded = components.expand_query(question)
    runs: dict[str, dict] = {}
    for name, fetch in (
        ("bm25", lambda: _fetch_page_pool(components.bm25_search, expanded, candidate_k, max_candidate_k, pool_size)),
        ("vector", lambda: _fetch_page_pool(components.vector_search, expanded, candidate_k, max_candidate_k, pool_size)),
        ("hybrid", lambda: _hybrid_pool(components, question, candidate_k, pool_size)),
    ):
        try:
            pool, diagnostics = fetch()
            runs[name] = {"results": pool, "diagnostics": diagnostics, "error": None}
        except Exception as error:  # recorded, never silently skipped
            logger.exception("%s retrieval failed for %r", name, question)
            runs[name] = {"results": [], "diagnostics": {}, "error": f"{type(error).__name__}: {error}"}
    for run in runs.values():
        run["diagnostics"]["expanded_query"] = expanded
    return runs


def _hybrid_pool(components, question, candidate_k, pool_size):
    results = list(components.hybrid_search(question, pool_size, candidate_k) or [])
    pool = collapse_to_pages(results, pool_size)
    return pool, {
        "chunks_fetched": len(results),
        "candidate_k_per_retriever": candidate_k,
        "distinct_pages_available": _distinct_units(results),
        "pool_size": len(pool),
    }


def rerank_pool(question: str, pool: list[dict], rerank) -> tuple[list[dict], list[dict]]:
    """Reorder an existing pool. Returns (production blend order, cross-encoder-only order).

    The pool is passed to the reranker unchanged and fully (top_k = pool size),
    so both orders contain exactly the pool's candidates.
    """

    if not pool:
        return [], []
    reranked = [dict(item) for item in rerank(question, [dict(item) for item in pool], len(pool))]
    if len(reranked) != len(pool):
        raise ValueError(f"Reranker returned {len(reranked)} results for a pool of {len(pool)}")
    blend_position = {id(item): index for index, item in enumerate(reranked)}
    ce_order = sorted(
        reranked,
        key=lambda item: (-float(item.get("reranker_score", float("-inf"))), blend_position[id(item)]),
    )
    return reranked, ce_order


def run_question(
    question: str,
    components: RetrievalComponents,
    include_rerank: bool = True,
    candidate_k: int = DEFAULT_CANDIDATE_K,
    pool_size: int = DEFAULT_POOL_SIZE,
    max_candidate_k: int = DEFAULT_MAX_CANDIDATE_K,
) -> dict[str, dict]:
    """All strategies for one question: base pools plus (optionally) reranked orders."""

    runs = build_base_pools(question, components, candidate_k, pool_size, max_candidate_k)
    if include_rerank:
        if components.rerank is None:
            raise ValueError("include_rerank=True needs a rerank component")
        for base in BASE_STRATEGIES:
            base_run = runs[base]
            blend_name, ce_name = base + RERANK_SUFFIX, base + CE_SUFFIX
            if base_run["error"]:
                error = f"base strategy failed: {base_run['error']}"
                runs[blend_name] = {"results": [], "diagnostics": {}, "error": error}
                runs[ce_name] = {"results": [], "diagnostics": {}, "error": error}
                continue
            try:
                blend, ce = rerank_pool(question, base_run["results"], components.rerank)
                diagnostics = {"pool_size": len(blend), "reranked_from": base}
                runs[blend_name] = {"results": blend, "diagnostics": diagnostics, "error": None}
                runs[ce_name] = {"results": ce, "diagnostics": dict(diagnostics), "error": None}
            except Exception as error:
                logger.exception("rerank of %s failed for %r", base, question)
                message = f"{type(error).__name__}: {error}"
                runs[blend_name] = {"results": [], "diagnostics": {}, "error": message}
                runs[ce_name] = {"results": [], "diagnostics": {}, "error": message}
    return runs


def run_benchmark(
    benchmark: Mapping,
    components: RetrievalComponents,
    include_rerank: bool = True,
    candidate_k: int = DEFAULT_CANDIDATE_K,
    pool_size: int = DEFAULT_POOL_SIZE,
    max_candidate_k: int = DEFAULT_MAX_CANDIDATE_K,
    progress: Callable[[str], None] | None = None,
) -> dict:
    """Run every strategy on every benchmark question (answerable and unanswerable).

    Returns a JSON-serialisable record of the raw rankings, so results can be
    re-scored later without models (see ``rankings_by_strategy``).
    """

    questions = benchmark["questions"]
    records = []
    for index, question in enumerate(questions, start=1):
        if progress:
            progress(f"[{index}/{len(questions)}] {question['id']}: {question['question']}")
        runs = run_question(question["question"], components, include_rerank, candidate_k, pool_size, max_candidate_k)
        records.append({
            "id": question["id"],
            "category": question.get("category"),
            "answerable": question.get("answerable") is True,
            "question": question["question"],
            "strategies": {
                name: {
                    # Reranked variants reorder the base pool, so their text previews would be duplicates.
                    "results": [_slim(item, preview=name in BASE_STRATEGIES) for item in run["results"]],
                    "diagnostics": run["diagnostics"],
                    "error": run["error"],
                }
                for name, run in runs.items()
            },
        })
    names = strategy_names(include_rerank)
    return {
        "strategies": names,
        "config": {
            "candidate_k": candidate_k,
            "max_candidate_k": max_candidate_k,
            "pool_size": pool_size,
            "include_rerank": include_rerank,
            **components.description,
        },
        "questions": records,
        "errors": [
            {"id": record["id"], "strategy": name, "error": run["error"]}
            for record in records
            for name, run in record["strategies"].items()
            if run["error"]
        ],
    }


def rankings_by_strategy(runs: Mapping) -> dict[str, dict[str, list[dict]]]:
    """``{strategy: {question_id: ranked results}}`` from a ``run_benchmark`` record."""

    rankings: dict[str, dict[str, list[dict]]] = {name: {} for name in runs["strategies"]}
    for record in runs["questions"]:
        for name in runs["strategies"]:
            rankings[name][record["id"]] = list(record["strategies"].get(name, {}).get("results", []))
    return rankings


def pool_shortfalls(runs: Mapping, required: int) -> list[dict]:
    """Strategy/question pairs whose pool holds fewer than ``required`` distinct units."""

    return [
        {"id": record["id"], "strategy": name, "pool_size": len(run["results"])}
        for record in runs["questions"]
        for name, run in record["strategies"].items()
        if len(run["results"]) < required
    ]


def unanswerable_retrieval(benchmark: Mapping, runs: Mapping, k: int) -> list[dict]:
    """What each strategy retrieved for unanswerable questions (reported separately, not scored).

    Records the top-``k`` pages, which of the question's ``distractor_pages``
    (near-miss content) appear there, and the highest cross-encoder score when
    the strategy was reranked.
    """

    by_id = {q["id"]: q for q in benchmark["questions"]}
    report = []
    for record in runs["questions"]:
        question = by_id.get(record["id"], {})
        if question.get("answerable") is True:
            continue
        distractors = set(question.get("distractor_pages") or [])
        per_strategy = {}
        for name, run in record["strategies"].items():
            pages = [item.get("page") for item in run["results"][:k]]
            ce_scores = [item["reranker_score"] for item in run["results"] if isinstance(item.get("reranker_score"), (int, float))]
            per_strategy[name] = {
                "top_pages": pages,
                "distractor_pages_in_top_k": [page for page in pages if page in distractors],
                "max_reranker_score": max(ce_scores) if ce_scores else None,
                "error": run["error"],
            }
        report.append({
            "id": record["id"],
            "question": record["question"],
            "distractor_pages": sorted(distractors),
            "strategies": per_strategy,
        })
    return report
