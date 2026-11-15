"""Phase 2 Tasks 3-4: compare BM25, vector and hybrid retrieval, before and after reranking.

Usage (from the project root, with the Qdrant index built):

    python scripts/compare_retrieval.py
    python scripts/compare_retrieval.py --runs data/evaluation/results/retrieval_runs.json   # re-score, no models

Every strategy runs once per question (all 25 questions) on one shared
HybridRetriever instance, so BM25, vector and hybrid use the same chunks,
index and expanded query. Each strategy is reduced to a pool of its top
``--pool-size`` distinct pages; reranked variants reorder exactly that pool.
See app/evaluation/retrieval_strategies.py for the controls.

Rankings are saved to ``retrieval_runs.json`` and scored against benchmark v2
(primary) and v1 (baseline). Outputs:

    results/retrieval_runs.json                raw rankings for every strategy and question
    results/strategy_comparison.json           per-strategy, per-question scores (v2 and v1)
    results/strategy_comparison_summary.json   overall/category tables, failures, unanswerable
    results/reranker_evaluation.json           before/after reranking, per question (v2)
    results/reranker_evaluation_summary.json   counts, deltas, hit rates (v2 and v1)
    retrieval_strategy_comparison.md           readable report (+ charts/)
    reranker_evaluation.md                     readable report (+ charts/)
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from datetime import datetime, timezone
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.evaluation.comparison import best_strategy, compare_reports, strategy_table  # noqa: E402
from app.evaluation.reporting import fmt, md_table, signed, svg_grouped_bars, write_text  # noqa: E402
from app.evaluation.retrieval_evaluator import (  # noqa: E402
    DEFAULT_K_VALUES,
    evaluate_results,
    load_benchmark,
    split_questions,
)
from app.evaluation.retrieval_metrics import normalise_k_values  # noqa: E402
from app.evaluation.retrieval_strategies import (  # noqa: E402
    BASE_STRATEGIES,
    CE_SUFFIX,
    DEFAULT_CANDIDATE_K,
    DEFAULT_MAX_CANDIDATE_K,
    DEFAULT_POOL_SIZE,
    RERANK_SUFFIX,
    RetrievalComponents,
    pool_shortfalls,
    rankings_by_strategy,
    run_benchmark,
    unanswerable_retrieval,
)

log = logging.getLogger("compare_retrieval")

EVAL_DIR = PROJECT_ROOT / "data" / "evaluation"
DEFAULT_BENCHMARK = EVAL_DIR / "benchmark_v2.json"
DEFAULT_BASELINE = EVAL_DIR / "benchmark_v1.json"
DEFAULT_PDF = PROJECT_ROOT / "data" / "documents" / "annual_report.pdf"

LABELS = {
    "bm25": "BM25",
    "vector": "Vector",
    "hybrid": "Hybrid (RRF)",
    "bm25+rerank": "BM25 + rerank",
    "vector+rerank": "Vector + rerank",
    "hybrid+rerank": "Hybrid + rerank (production)",
    "bm25+ce": "BM25 + CE only",
    "vector+ce": "Vector + CE only",
    "hybrid+ce": "Hybrid + CE only",
}


# ---------------------------------------------------------------------------
# Model-backed components
# ---------------------------------------------------------------------------


def build_components(pdf_path: Path, include_rerank: bool) -> tuple[RetrievalComponents, callable]:
    from app.rag.hybrid_retriever import BM25_WEIGHT, RRF_RANK_CONSTANT, VECTOR_WEIGHT, HybridRetriever
    from app.rag.query_expander import expand_query

    retriever = HybridRetriever(pdf_path=str(pdf_path))
    description = {
        "pdf": pdf_path.name,
        "num_chunks": len(retriever.chunks),
        "embedding_model": getattr(retriever.vector_retriever.embedding_model, "model_name", "BAAI/bge-small-en-v1.5"),
        "vector_collection": "insightrag_documents",
        "bm25_weight": BM25_WEIGHT,
        "vector_weight": VECTOR_WEIGHT,
        "rrf_rank_constant": RRF_RANK_CONSTANT,
        "query_expansion": "app.rag.query_expander.expand_query (all base strategies)",
    }
    rerank = None
    if include_rerank:
        from app.rag.reranker import MODEL_NAME, Reranker

        reranker = Reranker()
        rerank = lambda question, docs, k: reranker.rerank(question, docs, top_k=k)  # noqa: E731
        description["reranker_model"] = MODEL_NAME
        description["rerank_formula"] = "0.7 * minmax(first-stage score) + 0.3 * minmax(cross-encoder score) (app/rag/reranker.py)"

    components = RetrievalComponents(
        bm25_search=lambda query, k: retriever.bm25.search(query, top_k=k),
        vector_search=lambda query, k: retriever.vector_retriever.search(query, top_k=k),
        hybrid_search=lambda query, top_k, candidate_k: retriever.search(query, top_k=top_k, candidate_k=candidate_k),
        expand_query=expand_query,
        rerank=rerank,
        description=description,
    )
    return components, retriever.vector_retriever.vector_store.close


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------


def score_runs(benchmark: dict, runs: dict, k_values: list[int], version: str) -> dict[str, dict]:
    rankings = rankings_by_strategy(runs)
    answerable, _ = split_questions(benchmark)
    ids = {q["id"] for q in answerable}
    return {
        name: evaluate_results(
            benchmark,
            {qid: results for qid, results in by_id.items() if qid in ids},
            k_values,
            config={"strategy": name, "benchmark_version": version, **runs["config"]},
        )
        for name, by_id in rankings.items()
    }


def _check_same_questions(primary: dict, baseline: dict) -> None:
    a = [(q["id"], q["question"], q.get("answerable")) for q in primary["questions"]]
    b = [(q["id"], q["question"], q.get("answerable")) for q in baseline["questions"]]
    if a != b:
        raise SystemExit("Primary and baseline benchmarks must contain the same questions")


def _compact_question_matrix(reports: dict[str, dict], strategies: list[str]) -> list[dict]:
    first = reports[strategies[0]]["questions"]
    rows = []
    for index, question in enumerate(first):
        rows.append({
            "id": question["id"],
            "category": question["category"],
            "relevant_pages": question["relevant_pages"],
            "first_relevant_rank": {s: reports[s]["questions"][index]["first_relevant_rank"] for s in strategies},
            "recall_at_3": {s: reports[s]["questions"][index]["metrics"]["recall"]["3"] for s in strategies},
        })
    return rows


def _failures(reports: dict[str, dict], strategies: list[str], k: int) -> dict[str, list[str]]:
    return {
        s: [q["id"] for q in reports[s]["questions"] if q["first_relevant_rank"] is None or q["first_relevant_rank"] > k]
        for s in strategies
    }


def _incomplete(reports: dict[str, dict], strategies: list[str], k: int) -> dict[str, list[str]]:
    """Questions where some expected fact is still missing from the top k."""

    return {s: [q["id"] for q in reports[s]["questions"] if q["metrics"]["recall"][str(k)] < 1.0] for s in strategies}


# ---------------------------------------------------------------------------
# Reports
# ---------------------------------------------------------------------------


def _metric_rows(table: dict, strategies: list[str], k_values: list[int]) -> list[list]:
    rows = []
    for s in strategies:
        t = table[s]
        rows.append(
            [LABELS.get(s, s)]
            + [t["recall"][str(k)] for k in k_values]
            + [t["page_recall"][str(k)] for k in k_values]
            + [t["precision"][str(k)] for k in k_values]
            + [t["mrr"]]
        )
    return rows


def _metric_headers(k_values: list[int]) -> list[str]:
    return (
        ["Strategy"]
        + [f"R@{k}" for k in k_values]
        + [f"PR@{k}" for k in k_values]
        + [f"P@{k}" for k in k_values]
        + ["MRR"]
    )


def render_strategy_report(data: dict, chart_dir: Path, rel_chart: str) -> str:
    k_values = data["k_values"]
    base = list(BASE_STRATEGIES)
    v2 = data["tables"]["v2"]
    v1 = data["tables"]["v1"]

    chart = svg_grouped_bars(
        "Fact recall@K by strategy (benchmark v2, 22 answerable questions)",
        [f"K={k}" for k in k_values],
        {LABELS[s]: [v2[s]["recall"][str(k)] for k in k_values] for s in base},
        y_label="Recall (per fact)",
    )
    write_text(chart_dir / "strategy_recall.svg", chart)
    categories = list(data["by_category_v2"][base[0]])
    chart_cat = svg_grouped_bars(
        "MRR by category and strategy (benchmark v2)",
        categories,
        {LABELS[s]: [data["by_category_v2"][s][c]["mrr"] for c in categories] for s in base},
        y_label="MRR",
    )
    write_text(chart_dir / "strategy_mrr_by_category.svg", chart_cat)

    lines = [
        "# Retrieval strategy comparison: BM25 vs vector vs hybrid",
        "",
        "_Generated by `scripts/compare_retrieval.py`. Do not edit by hand; re-run the script._",
        "",
        f"- Generated: {data['timestamp']}",
        f"- Primary benchmark: `{data['benchmarks']['v2']}`; baseline: `{data['benchmarks']['v1']}`",
        f"- Questions: {data['counts']['answerable']} answerable scored, {data['counts']['unanswerable']} unanswerable reported separately",
        f"- K values: {', '.join(map(str, k_values))} (K counts distinct (source, page) units)",
        "",
        "## Setup (measured configuration)",
        "",
        md_table(["Setting", "Value"], [[k, json.dumps(v) if not isinstance(v, str) else v] for k, v in data["config"].items()], ["l", "l"]),
        "",
        "Every strategy is reduced to its top "
        f"{data['config']['pool_size']} distinct pages before scoring. BM25 and vector fetch more chunks "
        f"(from {data['config']['candidate_k']} up to {data['config']['max_candidate_k']}) until that many distinct "
        "pages are available; hybrid is page-diverse and is called exactly as in `app/main.py`.",
        f"Pools with fewer than {max(k_values)} distinct pages: "
        + (", ".join(f"{p['id']}/{p['strategy']} ({p['pool_size']})" for p in data["pool_shortfalls"]) or "none")
        + ". Errors: " + (", ".join(f"{e['id']}/{e['strategy']}: {e['error']}" for e in data["errors"]) or "none") + ".",
        "",
        "## Overall results (benchmark v2, measured)",
        "",
        "R = fact recall, PR = page recall, P = precision, all at K distinct pages.",
        "",
        md_table(_metric_headers(k_values), _metric_rows(v2, base, k_values)),
        "",
        f"![Fact recall by strategy]({rel_chart}/strategy_recall.svg)",
        "",
        "Hit rate@K (share of questions with a relevant page in the top K):",
        "",
        md_table(["Strategy"] + [f"Hit@{k}" for k in k_values], [[LABELS[s]] + [v2[s]["hit_rate"][str(k)] for k in k_values] for s in base]),
        "",
        "## Best strategy per metric (calculated from the table above)",
        "",
        md_table(
            ["Metric", "Best"],
            [["MRR", ", ".join(LABELS[s] for s in best_strategy({s: v2[s] for s in base}, "mrr"))]]
            + [[f"Recall@{k}", ", ".join(LABELS[s] for s in best_strategy({s: v2[s] for s in base}, "recall", k))] for k in k_values]
            + [[f"Precision@{k}", ", ".join(LABELS[s] for s in best_strategy({s: v2[s] for s in base}, "precision", k))] for k in k_values],
            ["l", "l"],
        ),
        "",
        "## Category-level results (benchmark v2)",
        "",
    ]
    cat_rows = []
    for c in categories:
        row = [c, data["by_category_v2"][base[0]][c]["num_questions"]]
        for s in base:
            cat = data["by_category_v2"][s][c]
            row += [cat["recall"]["3"], cat["mrr"]]
        cat_rows.append(row)
    lines += [
        md_table(["Category", "n"] + [f"{LABELS[s]} {m}" for s in base for m in ("R@3", "MRR")], cat_rows),
        "",
        f"![MRR by category]({rel_chart}/strategy_mrr_by_category.svg)",
        "",
        "## Per-question first relevant rank (benchmark v2)",
        "",
        "Rank of the first relevant distinct page (— = none in the pool of "
        f"{data['config']['pool_size']}).",
        "",
        md_table(
            ["ID", "Category", "Relevant pages"] + [LABELS[s] for s in base],
            [[row["id"], row["category"], ", ".join(map(str, row["relevant_pages"]))]
             + [row["first_relevant_rank"][s] for s in base] for row in data["question_matrix_v2"]],
            ["l", "l", "l"] + ["r"] * len(base),
        ),
        "",
        "## Where each strategy fails (calculated)",
        "",
        md_table(
            ["Strategy", "No relevant page in top 3", "No relevant page in top 10", "A fact still missing at K=10"],
            [[LABELS[s], ", ".join(data["failures_top3"][s]) or "none", ", ".join(data["failures_top10"][s]) or "none",
              ", ".join(data["incomplete_top10"][s]) or "none"] for s in base],
            ["l", "l", "l", "l"],
        ),
        "",
        "## Benchmark correction impact (v1 → v2)",
        "",
        "v2 adds page 3 as alternative evidence for Q017/Q018 (see `benchmark_v2_changes.md`). Same rankings, scored twice:",
        "",
        md_table(
            ["Strategy", "MRR v1", "MRR v2", "R@3 v1", "R@3 v2", "P@3 v1", "P@3 v2"],
            [[LABELS[s], v1[s]["mrr"], v2[s]["mrr"], v1[s]["recall"]["3"], v2[s]["recall"]["3"],
              v1[s]["precision"]["3"], v2[s]["precision"]["3"]] for s in data["strategies"]],
        ),
        "",
        "Overall results on v1 (for comparison with the Task 2 baseline):",
        "",
        md_table(_metric_headers(k_values), _metric_rows(v1, data["strategies"], k_values)),
        "",
        f"Task 2 baseline reproduction: {data['task2_baseline_check']}",
        "",
        "## Unanswerable questions (retrieval only, not scored)",
        "",
        "These questions have no evidence pages. The table shows whether near-miss "
        "(distractor) pages, which hold related but non-answering figures, reach the top 5.",
        "",
        md_table(
            ["ID", "Distractor pages"] + [LABELS[s] for s in base],
            [[u["id"], ", ".join(map(str, u["distractor_pages"]))]
             + [", ".join(map(str, u["strategies"][s]["distractor_pages_in_top_k"])) or "none" for s in base]
             for u in data["unanswerable"]],
            ["l", "l"] + ["l"] * len(base),
        ),
        "",
        "Retrieval always returns pages; whether the system abstains is decided "
        "after generation and is evaluated in the answer-quality report.",
    ]
    return "\n".join(lines)


def _reranker_rows(comparisons: dict, k_values: list[int]) -> list[list]:
    rows = []
    for name, comp in comparisons.items():
        b, a = comp["overall_before"], comp["overall_after"]
        rows.append([
            name, b["mrr"], a["mrr"], signed(comp["deltas"]["mrr"]),
            comp["hit_rate"]["1"]["before"], comp["hit_rate"]["1"]["after"],
            comp["hit_rate"]["3"]["before"], comp["hit_rate"]["3"]["after"],
            b["recall"]["3"], a["recall"]["3"],
            b["precision"]["3"], a["precision"]["3"],
            f"{comp['counts']['improved']}/{comp['counts']['degraded']}/{comp['counts']['unchanged']}",
        ])
    return rows


def render_reranker_report(data: dict, chart_dir: Path, rel_chart: str) -> str:
    k_values = data["k_values"]
    v2 = data["comparisons"]["v2"]
    blend = {base: v2[f"{base} -> {base}{RERANK_SUFFIX}"] for base in BASE_STRATEGIES}
    chart = svg_grouped_bars(
        "MRR before and after reranking (benchmark v2)",
        [LABELS[b] for b in BASE_STRATEGIES],
        {
            "Before": [blend[b]["overall_before"]["mrr"] for b in BASE_STRATEGIES],
            "After (production rerank)": [blend[b]["overall_after"]["mrr"] for b in BASE_STRATEGIES],
            "Cross-encoder only": [v2[f"{b} -> {b}{CE_SUFFIX}"]["overall_after"]["mrr"] for b in BASE_STRATEGIES],
        },
        y_label="MRR",
    )
    write_text(chart_dir / "reranker_mrr.svg", chart)

    headers = ["Comparison", "MRR before", "MRR after", "Δ MRR", "Hit@1 before", "Hit@1 after",
               "Hit@3 before", "Hit@3 after", "R@3 before", "R@3 after", "P@3 before", "P@3 after",
               "Improved/degraded/unchanged"]
    lines = [
        "# Reranker evaluation: before vs after cross-encoder reranking",
        "",
        "_Generated by `scripts/compare_retrieval.py`. Do not edit by hand; re-run the script._",
        "",
        f"- Generated: {data['timestamp']}",
        f"- Primary benchmark: `{data['benchmarks']['v2']}` (v1 deltas at the end)",
        f"- Reranker: `{data['config'].get('reranker_model')}`; production formula: {data['config'].get('rerank_formula')}",
        f"- Each reranked list is the base strategy's own pool of {data['config']['pool_size']} distinct pages, "
        "reordered (identical candidates before and after).",
        "- `+rerank` = production blend (what the assistant uses). `+ce` = the same pool sorted by the "
        "cross-encoder score alone (diagnostic, same model call).",
        "",
        "## Overall (benchmark v2, measured)",
        "",
        md_table(headers, _reranker_rows(v2, k_values), ["l"] + ["r"] * (len(headers) - 1)),
        "",
        f"![MRR before and after reranking]({rel_chart}/reranker_mrr.svg)",
        "",
        "Verdict per question: reciprocal rank up = improved, down = degraded; ties are decided by "
        "summed fact recall over all K; otherwise unchanged.",
        "",
        "## Does reranking improve top-1 and top-3 retrieval? (calculated)",
        "",
    ]
    for name, comp in v2.items():
        h1, h3 = comp["hit_rate"]["1"], comp["hit_rate"]["3"]
        lines.append(
            f"- **{name}**: Hit@1 {fmt(h1['before'])} → {fmt(h1['after'])} ({signed(h1['delta'])}), "
            f"Hit@3 {fmt(h3['before'])} → {fmt(h3['after'])} ({signed(h3['delta'])}), "
            f"R@1 {signed(comp['deltas']['recall']['1'])}, R@3 {signed(comp['deltas']['recall']['3'])}."
        )
    lines += ["", "## Per-question changes (benchmark v2)", ""]
    for name, comp in v2.items():
        lines += [
            f"### {name}",
            "",
            f"Improved: {', '.join(comp['questions_by_verdict']['improved']) or 'none'}  ",
            f"Degraded: {', '.join(comp['questions_by_verdict']['degraded']) or 'none'}  ",
            f"Unchanged: {', '.join(comp['questions_by_verdict']['unchanged']) or 'none'}  ",
            f"Relevant evidence moved down: {', '.join(comp['questions_with_relevant_moved_down']) or 'none'}",
            "",
            md_table(
                ["ID", "Category", "First rel. rank before", "after", "RR Δ", "Verdict", "Relevant pages moved down (before→after)"],
                [[c["id"], c["category"], c["first_relevant_rank_before"], c["first_relevant_rank_after"],
                  signed(c["reciprocal_rank_delta"]), c["verdict"],
                  ", ".join(f"p{m['page']}: {m['rank_before']}→{m['rank_after'] if m['rank_after'] is not None else 'out'}"
                            for m in c["relevant_moved_down"]) or ""]
                 for c in comp["per_question"]],
                ["l", "l", "r", "r", "r", "l", "l"],
            ),
            "",
        ]
    lines += ["## Category-level MRR change (benchmark v2, production rerank)", ""]
    categories = list(blend["hybrid"]["by_category"])
    lines += [
        md_table(
            ["Category"] + [f"{LABELS[b]} before → after" for b in BASE_STRATEGIES],
            [[c] + [f"{fmt(blend[b]['by_category'][c]['mrr_before'])} → {fmt(blend[b]['by_category'][c]['mrr_after'])}"
                    for b in BASE_STRATEGIES] for c in categories],
            ["l"] + ["r"] * len(BASE_STRATEGIES),
        ),
        "",
        "## Benchmark v1 (baseline) deltas",
        "",
        md_table(headers, _reranker_rows(data["comparisons_v1_summary"], k_values), ["l"] + ["r"] * (len(headers) - 1)),
    ]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--benchmark", type=Path, default=DEFAULT_BENCHMARK, help="Primary benchmark (default v2)")
    parser.add_argument("--baseline-benchmark", type=Path, default=DEFAULT_BASELINE, help="Baseline benchmark (default v1)")
    parser.add_argument("--pdf", type=Path, default=DEFAULT_PDF)
    parser.add_argument("--k", type=int, nargs="+", default=list(DEFAULT_K_VALUES))
    parser.add_argument("--candidate-k", type=int, default=DEFAULT_CANDIDATE_K)
    parser.add_argument("--max-candidate-k", type=int, default=DEFAULT_MAX_CANDIDATE_K)
    parser.add_argument("--pool-size", type=int, default=DEFAULT_POOL_SIZE)
    parser.add_argument("--output-dir", type=Path, default=EVAL_DIR / "results")
    parser.add_argument("--report-dir", type=Path, default=EVAL_DIR)
    parser.add_argument("--runs", type=Path, help="Re-score an existing retrieval_runs.json instead of running models")
    args = parser.parse_args(argv)
    try:
        args.k = normalise_k_values(args.k)
    except ValueError as error:
        parser.error(str(error))
    if args.pool_size < max(args.k):
        parser.error(f"--pool-size ({args.pool_size}) must be at least max K ({max(args.k)})")
    return args


def _dump(path: Path, data) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return path


def main(argv=None, components_factory=build_components) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    args = parse_args(argv)
    primary = load_benchmark(args.benchmark)
    baseline = load_benchmark(args.baseline_benchmark)
    _check_same_questions(primary, baseline)
    os.chdir(PROJECT_ROOT)  # Qdrant path and HybridRetriever defaults are relative

    if args.runs:
        runs = json.loads(args.runs.read_text(encoding="utf-8"))
        log.info("Re-scoring saved rankings from %s", args.runs)
    else:
        components, close = components_factory(args.pdf, True)
        try:
            runs = run_benchmark(primary, components, True, args.candidate_k, args.pool_size, args.max_candidate_k, progress=log.info)
        finally:
            close()
        runs["timestamp"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
        _dump(args.output_dir / "retrieval_runs.json", runs)

    k_values = args.k
    strategies = runs["strategies"]
    base = [s for s in BASE_STRATEGIES if s in strategies]
    reports = {
        "v2": score_runs(primary, runs, k_values, primary.get("version", "v2")),
        "v1": score_runs(baseline, runs, k_values, baseline.get("version", "v1")),
    }
    tables = {v: strategy_table(reports[v], k_values) for v in reports}
    answerable, unanswerable = split_questions(primary)
    timestamp = datetime.now(timezone.utc).isoformat(timespec="seconds")
    common = {
        "timestamp": timestamp,
        "runs_timestamp": runs.get("timestamp"),
        "benchmarks": {"v2": args.benchmark.name, "v1": args.baseline_benchmark.name},
        "k_values": k_values,
        "config": runs["config"],
        "counts": {"answerable": len(answerable), "unanswerable": len(unanswerable)},
    }

    # Task 2 consistency check: hybrid+rerank on v1 should reproduce the Task 2 baseline.
    task2 = EVAL_DIR / "results" / "retrieval_evaluation_summary.json"
    check = "Task 2 summary not found."
    if task2.is_file() and "hybrid+rerank" in tables["v1"]:
        old = json.loads(task2.read_text(encoding="utf-8"))["overall"]
        new = tables["v1"]["hybrid+rerank"]
        same = old["mrr"] == new["mrr"] and all(old[m] == new[m] for m in ("recall", "page_recall", "precision"))
        check = (
            f"hybrid+rerank on v1 gives MRR {fmt(new['mrr'])} vs Task 2 {fmt(old['mrr'])}; "
            + ("all recall/page-recall/precision values identical." if same else "values DIFFER (see JSON).")
        )

    strategy_data = {
        **common,
        "strategies": strategies,
        "tables": tables,
        "by_category_v2": {s: reports["v2"][s]["summary"]["by_category"] for s in strategies},
        "question_matrix_v2": _compact_question_matrix(reports["v2"], strategies),
        "failures_top3": _failures(reports["v2"], strategies, 3),
        "failures_top10": _failures(reports["v2"], strategies, 10),
        "incomplete_top10": _incomplete(reports["v2"], strategies, 10),
        "pool_shortfalls": pool_shortfalls(runs, max(k_values)),
        "errors": runs.get("errors", []),
        "unanswerable": unanswerable_retrieval(primary, runs, 5),
        "task2_baseline_check": check,
    }
    _dump(args.output_dir / "strategy_comparison_summary.json", strategy_data)
    # Per-question records without the ranked chunk list (that lives in retrieval_runs.json).
    _dump(args.output_dir / "strategy_comparison.json", {
        **common,
        "reports": {
            v: {
                s: {
                    "summary": reports[v][s]["summary"],
                    "questions": [{k: val for k, val in q.items() if k != "retrieved"} for q in reports[v][s]["questions"]],
                }
                for s in strategies
            }
            for v in reports
        },
    })

    comparisons = {}
    for version in ("v2", "v1"):
        comparisons[version] = {}
        for b in base:
            for suffix in (RERANK_SUFFIX, CE_SUFFIX):
                if b + suffix in strategies:
                    comparisons[version][f"{b} -> {b}{suffix}"] = compare_reports(reports[version][b], reports[version][b + suffix], k_values)
    reranker_data = {
        **common,
        "comparisons": {"v2": comparisons["v2"]},
        "comparisons_v1_summary": {
            name: {key: comp[key] for key in ("counts", "deltas", "hit_rate", "overall_before", "overall_after", "questions_by_verdict")}
            for name, comp in comparisons["v1"].items()
        },
    }
    _dump(args.output_dir / "reranker_evaluation.json", reranker_data)
    _dump(args.output_dir / "reranker_evaluation_summary.json", {
        **common,
        "v2": {name: {key: comp[key] for key in ("counts", "deltas", "hit_rate", "questions_by_verdict", "questions_with_relevant_moved_down")}
               for name, comp in comparisons["v2"].items()},
        "v1": {name: {key: comp[key] for key in ("counts", "deltas", "hit_rate", "questions_by_verdict")}
               for name, comp in comparisons["v1"].items()},
    })

    chart_dir = args.report_dir / "charts"
    write_text(args.report_dir / "retrieval_strategy_comparison.md", render_strategy_report(strategy_data, chart_dir, "charts"))
    if comparisons["v2"]:
        write_text(args.report_dir / "reranker_evaluation.md", render_reranker_report(reranker_data, chart_dir, "charts"))

    print()
    print(md_table(_metric_headers(k_values), _metric_rows(tables["v2"], strategies, k_values)))
    print()
    for name, comp in comparisons["v2"].items():
        print(f"{name}: MRR {signed(comp['deltas']['mrr'])}, improved/degraded/unchanged = "
              f"{comp['counts']['improved']}/{comp['counts']['degraded']}/{comp['counts']['unchanged']}")
    print(check)
    if runs.get("errors"):
        print(f"ERRORS: {runs['errors']}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
