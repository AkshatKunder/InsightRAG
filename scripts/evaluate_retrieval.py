"""Evaluate InsightRAG retrieval on the benchmark with Recall@K, Precision@K and MRR.

Usage (from the project root, with the Qdrant index built by ``python -m app.ingest``):

    python scripts/evaluate_retrieval.py
    python scripts/evaluate_retrieval.py --k 1 3 5 10 --output-dir data/evaluation/results

The retrieval stage is the one the assistant uses (app/main.py): hybrid
retrieval (BM25 + vector, weighted RRF, page diversity) for the top 20
candidates from 50 per retriever, then cross-encoder reranking. The reranker
keeps the top max(K) results instead of the assistant's 5. Reranking sorts all
candidates before cutting the list, so the first 5 results here are exactly
the 5 chunks the assistant passes to the LLM.

Ollama, answer generation and citation validation are not used. Only the
answerable questions are retrieved and scored.

Writes:
    <output-dir>/retrieval_evaluation.json          per-question detail
    <output-dir>/retrieval_evaluation_summary.json  averages, counts, config
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.evaluation.retrieval_evaluator import (  # noqa: E402
    DEFAULT_K_VALUES,
    evaluate_benchmark,
    load_benchmark,
)
from app.evaluation.retrieval_metrics import normalise_k_values  # noqa: E402


DEFAULT_BENCHMARK = PROJECT_ROOT / "data" / "evaluation" / "benchmark_v1.json"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "data" / "evaluation" / "results"
DEFAULT_PDF = PROJECT_ROOT / "data" / "documents" / "annual_report.pdf"

# Same settings as app/main.py.
HYBRID_TOP_K = 20
HYBRID_CANDIDATE_K = 50

DETAIL_FILE = "retrieval_evaluation.json"
SUMMARY_FILE = "retrieval_evaluation_summary.json"


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--benchmark", type=Path, default=DEFAULT_BENCHMARK)
    parser.add_argument("--pdf", type=Path, default=DEFAULT_PDF, help="Source PDF used to build the BM25 index")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--k",
        type=int,
        nargs="+",
        default=list(DEFAULT_K_VALUES),
        help="K values to report (default: %(default)s)",
    )
    args = parser.parse_args(argv)
    try:
        args.k = normalise_k_values(args.k)
    except ValueError as error:
        parser.error(str(error))
    return args


def build_pipeline_retriever(pdf_path: Path, depth: int):
    """Create the assistant's retrieval stage. Returns (retrieve_fn, close_fn, config)."""

    # Imported here so the module can be imported (and tested) without models.
    from app.rag.hybrid_retriever import (
        BM25_WEIGHT,
        RRF_RANK_CONSTANT,
        VECTOR_WEIGHT,
        HybridRetriever,
    )
    from app.rag.reranker import MODEL_NAME as RERANKER_MODEL
    from app.rag.reranker import Reranker

    hybrid_top_k = max(HYBRID_TOP_K, depth)
    retriever = HybridRetriever(pdf_path=str(pdf_path))
    reranker = Reranker()

    def retrieve(question: str):
        candidates = retriever.search(question, top_k=hybrid_top_k, candidate_k=HYBRID_CANDIDATE_K)
        return reranker.rerank(question, candidates, top_k=depth)

    def close():
        retriever.vector_retriever.vector_store.close()

    config = {
        "strategy": "hybrid_rrf+rerank",
        "description": "Assistant retrieval stage (app/main.py): hybrid BM25+vector RRF with page diversity, then cross-encoder rerank",
        "hybrid_top_k": hybrid_top_k,
        "hybrid_candidate_k": HYBRID_CANDIDATE_K,
        "bm25_weight": BM25_WEIGHT,
        "vector_weight": VECTOR_WEIGHT,
        "rrf_rank_constant": RRF_RANK_CONSTANT,
        "reranker_model": RERANKER_MODEL,
        "rerank_top_k": depth,
        "assistant_context_size": 5,
        "pdf": pdf_path.name,
    }
    return retrieve, close, config


def write_reports(report: dict, output_dir: Path) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    detail_path = output_dir / DETAIL_FILE
    summary_path = output_dir / SUMMARY_FILE
    detail = {"summary": report["summary"], "questions": report["questions"]}
    detail_path.write_text(json.dumps(detail, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    summary_path.write_text(json.dumps(report["summary"], indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return detail_path, summary_path


def format_summary(report: dict) -> str:
    summary = report["summary"]
    ks = [str(k) for k in summary["k_values"]]
    overall = summary["overall"]
    width = 14 + 9 * len(ks)

    lines = [
        "=" * width,
        "Retrieval evaluation",
        "=" * width,
        f"Benchmark:  {summary['benchmark_name']} ({summary['source_document']})",
        f"Strategy:   {summary['config'].get('strategy', 'n/a')}",
        f"Questions:  {summary['num_questions_total']} total, {summary['num_answerable']} answerable "
        f"(evaluated: {summary['num_evaluated']}), {summary['num_unanswerable']} unanswerable (excluded)",
        "",
        f"{'Metric':<14}" + "".join(f"{'@' + k:>9}" for k in ks),
        "-" * width,
    ]
    for name, label in (("recall", "Recall"), ("page_recall", "Page recall"), ("precision", "Precision")):
        lines.append(f"{label:<14}" + "".join(f"{overall[name][k]:>9.3f}" for k in ks))
    lines.append(f"{'MRR':<14}{overall['mrr']:>9.3f}")

    lines += ["", f"By category (Recall@K, MRR):", f"{'Category':<14}" + "".join(f"{'R@' + k:>9}" for k in ks) + f"{'MRR':>9}{'n':>4}"]
    for category, values in summary["by_category"].items():
        lines.append(
            f"{category:<14}"
            + "".join(f"{values['recall'][k]:>9.3f}" for k in ks)
            + f"{values['mrr']:>9.3f}{values['num_questions']:>4}"
        )

    lines += ["", "Per question (first relevant rank, relevant pages vs retrieved pages):"]
    for question in report["questions"]:
        rank = question["first_relevant_rank"]
        top = question["retrieved_pages"][: int(ks[-1])]
        lines.append(
            f"  {question['id']} {question['category']:<12} rank={rank if rank is not None else '-':<3} "
            f"relevant={question['relevant_pages']} retrieved={top}"
        )

    misses = summary["questions_with_no_relevant_result"]
    lines += ["", f"No relevant result in top {ks[-1]}: {', '.join(misses) if misses else 'none'}"]
    if summary["num_results_with_invalid_metadata"]:
        lines.append(f"Results with invalid source/page metadata: {summary['num_results_with_invalid_metadata']}")
    lines.append("Note: unanswerable questions are excluded; these metrics do not measure abstention.")
    return "\n".join(lines)


def main(argv=None, retriever_factory=build_pipeline_retriever) -> int:
    args = parse_args(argv)
    benchmark = load_benchmark(args.benchmark)

    # The vector store opens data/qdrant relative to the working directory.
    os.chdir(PROJECT_ROOT)

    retrieve, close, config = retriever_factory(args.pdf, max(args.k))
    try:
        report = evaluate_benchmark(
            benchmark,
            retrieve,
            args.k,
            config={**config, "benchmark": args.benchmark.name},
            progress=print,
        )
    finally:
        close()

    detail_path, summary_path = write_reports(report, args.output_dir)
    print()
    print(format_summary(report))
    print()
    print(f"Saved: {detail_path}")
    print(f"Saved: {summary_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
