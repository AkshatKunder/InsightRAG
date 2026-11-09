"""Read-only, staged diagnostic for total-sales retrieval.

The ``all`` mode uses exactly one production ``HybridRetriever`` and one
``Reranker``. For environments with short command windows, run the individual
stages instead; JSON handoff files contain only retrieved diagnostic results,
never embeddings or Qdrant data.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


QUESTION = "What were ArcelorMittal's total sales in 2025?"
TRACKED_PAGES = (95, 101, 203, 204)
AUTHORITATIVE_TOTAL_SALES_PAGE = 95
TOP_TO_PRINT = 10
HYBRID_TOP_K = 20
CANDIDATE_K = 50
DEFAULT_RESULTS_DIR = PROJECT_ROOT / "scripts" / "diagnostic_results"


def _score(value: Any) -> str:
    return "-" if value is None else f"{float(value):.8f}"


def _print_result(rank: int, result: dict[str, Any]) -> None:
    print(f"\nRank: {rank}")
    print(f"Page: {result.get('page')}")
    print(f"Source: {result.get('source')}")
    print(f"BM25 rank / score: {result.get('bm25_rank', '-')} / {_score(result.get('bm25_score'))}")
    print(f"Vector rank / score: {result.get('vector_rank', '-')} / {_score(result.get('vector_score'))}")
    print(f"RRF score: {_score(result.get('rrf_score'))}")
    print(f"Raw reranker score: {_score(result.get('reranker_score'))}")
    print(f"Final combined score: {_score(result.get('final_score'))}")
    print(f"Public score: {_score(result.get('score'))}")
    print("Text:")
    print(result.get("text", ""))


def _page_rank(results: list[dict[str, Any]], page: int) -> int | None:
    return next((rank for rank, result in enumerate(results, start=1) if result.get("page") == page), None)


def _print_stage(name: str, results: list[dict[str, Any]]) -> None:
    print("\n" + "=" * 100)
    print(name)
    print("=" * 100)
    print(f"Returned results: {len(results)}; printing first {min(TOP_TO_PRINT, len(results))}.")
    for rank, result in enumerate(results[:TOP_TO_PRINT], start=1):
        _print_result(rank, result)

    print("\nTracked-page positions in this returned stage:")
    for page in TRACKED_PAGES:
        rank = _page_rank(results, page)
        print(f"  Page {page}: {'rank ' + str(rank) if rank is not None else 'not returned'}")

    tracked = [result for result in results if result.get("page") in TRACKED_PAGES]
    if tracked:
        print("\nTRACKED PAGES PRESENT IN THIS STAGE; complete chunk text follows:")
        for result in tracked:
            print("-" * 100)
            print(f"Page {result.get('page')}")
            print(result.get("text", ""))


def _print_comparison(stages: list[tuple[str, list[dict[str, Any]]]]) -> None:
    print("\n" + "=" * 100)
    print("TRACKED-PAGE COMPARISON")
    print("=" * 100)
    for page in TRACKED_PAGES:
        positions = [f"{name}: {(_page_rank(results, page) or 'not returned')}" for name, results in stages]
        print(f"Page {page} -> " + "; ".join(positions))

    page_95_positions = [(name, _page_rank(results, AUTHORITATIVE_TOTAL_SALES_PAGE)) for name, results in stages]
    first_loss = next(
        (
            name for (_, previous_rank), (name, rank) in zip(page_95_positions, page_95_positions[1:])
            if previous_rank is not None and rank is None
        ),
        None,
    )
    if first_loss:
        print(f"\nFirst returned stage that loses page 95: {first_loss}.")
    elif page_95_positions[0][1] is None:
        print("\nPage 95 is absent from the first inspected stage; no loss stage can be identified.")
    else:
        print("\nPage 95 remains in every inspected returned stage.")

    for name, results in stages:
        page_95_rank, page_203_rank = _page_rank(results, 95), _page_rank(results, 203)
        if page_95_rank is not None and page_203_rank is not None and page_203_rank < page_95_rank:
            print(f"Page 203 first ranks ahead of page 95 in: {name} (page 203 rank {page_203_rank}; page 95 rank {page_95_rank}).")
            break
    else:
        print("Page 203 does not rank ahead of page 95 in a stage where both are returned.")


def _result_path(results_dir: Path, stage: str) -> Path:
    return results_dir / f"{stage}.json"


def _json_safe(value: Any) -> Any:
    """Recursively convert NumPy-like scalar values to standard JSON values."""
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    if isinstance(value, tuple):
        return [_json_safe(item) for item in value]

    item = getattr(value, "item", None)
    if callable(item):
        try:
            return _json_safe(item())
        except ValueError:
            # Non-scalar array-like values are not diagnostic score values;
            # leave them unchanged so json.dumps() reports unsupported data.
            return value
    return value


def _save_results(results_dir: Path, stage: str, results: list[dict[str, Any]]) -> None:
    results_dir.mkdir(parents=True, exist_ok=True)
    path = _result_path(results_dir, stage)
    path.write_text(json.dumps(_json_safe(results), indent=2), encoding="utf-8")
    print(f"\nSaved {stage} results to {path}")


def _load_results(results_dir: Path, stage: str) -> list[dict[str, Any]]:
    path = _result_path(results_dir, stage)
    if not path.exists():
        raise FileNotFoundError(f"Missing {path}. Run the '{stage}' stage first.")
    return json.loads(path.read_text(encoding="utf-8"))


def _close_hybrid(hybrid_retriever: Any | None) -> None:
    if hybrid_retriever is not None:
        hybrid_retriever.vector_retriever.vector_store.close()


def _run_bm25_stage(results_dir: Path) -> None:
    """Build production BM25 only; no embeddings, Qdrant, or model initialization."""
    from app.rag.bm25_retriever import BM25Retriever
    from app.rag.chunker import chunk_pages
    from app.rag.loader import load_pdf

    pages = load_pdf("data/documents/annual_report.pdf")
    results = BM25Retriever(chunk_pages(pages)).search(QUESTION, top_k=CANDIDATE_K)
    _print_stage("Independent BM25 retrieval", results)
    _save_results(results_dir, "bm25", results)


def _run_vector_stage(results_dir: Path) -> None:
    """Query the existing Qdrant collection only; no PDF load or BM25 build."""
    from app.rag.retriever import Retriever

    retriever = None
    try:
        retriever = Retriever()
        results = retriever.search(QUESTION, top_k=CANDIDATE_K)
        _print_stage("Independent vector retrieval", results)
        _save_results(results_dir, "vector", results)
    finally:
        if retriever is not None:
            retriever.vector_store.close()


def _run_hybrid_stage(results_dir: Path) -> None:
    """Use one HybridRetriever and its existing BM25/vector components."""
    from app.rag.hybrid_retriever import HybridRetriever

    hybrid_retriever = None
    try:
        hybrid_retriever = HybridRetriever()
        results = hybrid_retriever.search(QUESTION, top_k=HYBRID_TOP_K, candidate_k=CANDIDATE_K)
        _print_stage("Hybrid retrieval", results)
        _save_results(results_dir, "hybrid", results)
    finally:
        _close_hybrid(hybrid_retriever)


def _run_rerank_stage(results_dir: Path) -> None:
    """Rerank saved hybrid results; no PDF, Qdrant, or embedding initialization."""
    from app.rag.reranker import Reranker

    hybrid_results = _load_results(results_dir, "hybrid")
    results = Reranker().rerank(QUESTION, hybrid_results, top_k=TOP_TO_PRINT)
    _print_stage("Cross-encoder reranking", results)
    print("\n" + "=" * 100)
    print("TOP 5 RERANKED CHUNKS (FULL TEXT)")
    print("=" * 100)
    for rank, result in enumerate(results[:5], start=1):
        _print_result(rank, result)
    _save_results(results_dir, "reranked", results)


def _run_comparison_stage(results_dir: Path) -> None:
    stages = [
        ("Independent BM25", _load_results(results_dir, "bm25")),
        ("Independent vector", _load_results(results_dir, "vector")),
        ("Hybrid", _load_results(results_dir, "hybrid")),
        ("Reranked", _load_results(results_dir, "reranked")),
    ]
    _print_comparison(stages)


def _run_all(results_dir: Path) -> None:
    """Full diagnostic using one HybridRetriever, preserving production behavior."""
    from app.rag.hybrid_retriever import HybridRetriever
    from app.rag.reranker import Reranker

    hybrid_retriever = None
    try:
        hybrid_retriever = HybridRetriever()
        reranker = Reranker()
        bm25_results = hybrid_retriever.bm25.search(QUESTION, top_k=CANDIDATE_K)
        vector_results = hybrid_retriever.vector_retriever.search(QUESTION, top_k=CANDIDATE_K)
        hybrid_results = hybrid_retriever.search(QUESTION, top_k=HYBRID_TOP_K, candidate_k=CANDIDATE_K)
        reranked_results = reranker.rerank(QUESTION, hybrid_results, top_k=TOP_TO_PRINT)
        for stage, results, name in [
            ("bm25", bm25_results, "Independent BM25 retrieval"),
            ("vector", vector_results, "Independent vector retrieval"),
            ("hybrid", hybrid_results, "Hybrid retrieval"),
            ("reranked", reranked_results, "Cross-encoder reranking"),
        ]:
            _print_stage(name, results)
            _save_results(results_dir, stage, results)
        _print_comparison([
            ("Independent BM25", bm25_results),
            ("Independent vector", vector_results),
            ("Hybrid", hybrid_results),
            ("Reranked", reranked_results),
        ])
    finally:
        _close_hybrid(hybrid_retriever)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("all", "bm25", "vector", "hybrid", "rerank", "compare"), default="all")
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS_DIR)
    args = parser.parse_args()
    print(f"Question: {QUESTION}\nStage: {args.stage}")
    {
        "all": _run_all,
        "bm25": _run_bm25_stage,
        "vector": _run_vector_stage,
        "hybrid": _run_hybrid_stage,
        "rerank": _run_rerank_stage,
        "compare": _run_comparison_stage,
    }[args.stage](args.results_dir)


if __name__ == "__main__":
    main()
