"""Phase 4 evaluation: routing, NL-to-SQL, execution, safety and hybrid answers.

Usage (from the project root; needs Ollama, and the Qdrant index for hybrid questions):

    python scripts/evaluate_structured.py                 # 2 deterministic runs (temperature 0, seed 42)
    python scripts/evaluate_structured.py --runs 1 --no-rag

Builds a fresh evaluation database from the benchmark's dataset files
(``data/structured/phase4_eval.db``, git-ignored), then:

* routes every structured question and every benchmark_v2 document question;
* answers SQL and hybrid questions with ``StructuredQA`` (and the Phase 3.1
  RAG pipeline for hybrid questions);
* compares results with the gold SQL results, checks derived figures, charts,
  abstention on unanswerable questions, refusal and database integrity for
  safety questions, and hybrid answers (SQL part, document fact, attribution).

Writes ``data/evaluation/results/phase4/structured_eval.json`` (all records),
``structured_eval_summary.json`` and ``data/evaluation/phase4_evaluation_report.md``.
Earlier benchmarks and results are not touched.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.evaluation.answer_checks import check_fact, parse_quantities, values_match  # noqa: E402
from app.evaluation.reporting import fmt, md_table, write_text  # noqa: E402
from app.evaluation.structured_eval import db_fingerprint, derived_correct, rate, result_match  # noqa: E402
from app.structured import sql_safety  # noqa: E402
from app.structured.scope import DERIVED_NAME_RE  # noqa: E402
from app.structured.shares import FRACTION_NAME_RE  # noqa: E402
from app.structured.engine import ABSTENTION, Assistant, StructuredQA, document_vocabulary  # noqa: E402
from app.structured.ingestion import ingest_file  # noqa: E402
from app.structured.router import classify  # noqa: E402

EVAL = PROJECT_ROOT / "data" / "evaluation"
DEFAULT_DB = PROJECT_ROOT / "data" / "structured" / "phase4_eval.db"
OUT = EVAL / "results" / "phase4"


def build_db(benchmark: dict, db_path: Path) -> list[dict]:
    for suffix in ("", "-journal", "-wal", "-shm"):
        Path(f"{db_path}{suffix}").unlink(missing_ok=True)
    loaded = []
    for dataset in benchmark["datasets"]:
        for table in ingest_file(PROJECT_ROOT / dataset["file"], db_path):
            loaded.append({"file": dataset["file"], "kind": dataset["kind"], "table": table.name, "sheet": table.sheet,
                           "rows": len(table.rows), "columns": [[c.original_name, c.name, c.type] for c in table.columns]})
    return loaded


def gold(sql: str, db_path: Path) -> list[list]:
    result = sql_safety.execute(sql, db_path, max_rows=10_000)
    if not result.ok:
        raise SystemExit(f"Gold SQL failed ({result.error}): {sql}")
    return [list(r) for r in result.rows]


def doc_fact_found(fact: dict, answer: str) -> bool:
    """The document part states the fact's value (rounding-consistent).

    Currency facts must be stated with a currency; for counts the unit word may
    precede the number ("the total number of employees was 125,554").
    """

    check = check_fact({"metric": "value", "value": fact["value"], "unit": fact["unit"]}, answer or "", "")
    if check.get("matched"):
        return True
    if fact["unit"].endswith("employees"):
        return any(values_match(q, float(fact["value"])) for q in parse_quantities(answer or "") if q["unit_class"] in (None, "count"))
    return False


def evaluate_question(q: dict, sql_result: dict | None, rag_result: dict | None, combined: dict | None,
                      gold_rows: list | None, before: dict, after: dict) -> dict:
    rec = {"id": q["id"], "category": q["category"], "question": q["question"]}
    if sql_result is not None:
        gen = sql_result.get("generation") or {}
        attempts = gen.get("attempts") or []
        rec.update(sql=sql_result.get("sql"), error=sql_result.get("error"), answer=sql_result.get("answer"),
                   columns=sql_result.get("columns"), rows=sql_result.get("rows")[:50] if sql_result.get("rows") else [],
                   attempts=len(attempts), repaired=len(attempts) > 1 and sql_result.get("sql") is not None,
                   chart=(sql_result.get("chart") or {}).get("type"), incomplete=sql_result.get("incomplete"),
                   abstained=bool(sql_result.get("abstained")), scope=sql_result.get("scope"),
                   share_validation=sql_result.get("share_validation"), ratio_validation=sql_result.get("ratio_validation"),
                   presentation=sql_result.get("presentation"), rejected_result=sql_result.get("rejected_result"))
    expected = q.get("expected")
    if expected == "refused":
        rec["refused"] = sql_result is not None and sql_result.get("sql") is None and bool(sql_result.get("error"))
        rec["db_unchanged"] = before == after
        rec["correct"] = rec["refused"] and rec["db_unchanged"]
        return rec
    if expected == "no_answer":
        rec["correct"] = sql_result is not None and (sql_result.get("sql") is None or not sql_result.get("rows"))
        rec["generated_valid_sql"] = sql_result is not None and sql_result.get("sql") is not None
        return rec
    rec["generated_valid_sql"] = sql_result is not None and sql_result.get("sql") is not None
    rec["executed"] = rec["generated_valid_sql"] and sql_result.get("error") is None
    rec["result_match"] = bool(rec["executed"]) and result_match(gold_rows, sql_result.get("rows") or [], q.get("ordered", False))
    if q.get("expected_derived"):
        rec["derived_correct"] = bool(rec["executed"]) and bool(derived_correct(
            "period_change" if q["expected_derived"] == "period_change" else "shares", gold_rows, sql_result.get("analytics") or {}))
    if q.get("expected_chart"):
        rec["chart_correct"] = rec.get("chart") == q["expected_chart"]
    rec["gold_rows"] = gold_rows
    if q["route"] == "hybrid":
        doc_answer = (rag_result or {}).get("answer", "")
        rec["rag_answer"] = doc_answer
        rec["rag_status"] = ((rag_result or {}).get("validation") or {}).get("overall_status")
        rec["rag_pages"] = [c.get("page") for c in (rag_result or {}).get("citations", [])]
        rec["document_fact_found"] = all(doc_fact_found(f, doc_answer) for f in q.get("document_facts", []))
        rec["document_page_cited"] = all(f["page"] in rec["rag_pages"] for f in q.get("document_facts", []))
        answer = (combined or {}).get("answer") or ""
        rec["attribution"] = "From the structured data" in answer and "From the document" in answer
        rec["hybrid_correct"] = rec["result_match"] and rec["document_fact_found"] and rec["attribution"]
        rec["combined_answer"] = answer
        rec["sub_questions"] = (combined or {}).get("sub_questions")
        rec["correct"] = rec["hybrid_correct"]
    elif q.get("expected_derived") == "period_change":
        # Phase 4.2 (strict): the expected change must be computed AND the presented rows must be the
        # requested scope (same rows as the gold result) with no model-computed change columns. Finding the
        # right figure somewhere in an otherwise wrong result is not counted as correct.
        rec["model_derived_columns_presented"] = [c for c in (rec.get("columns") or []) if DERIVED_NAME_RE.search(c)]
        rec["correct_lenient"] = bool(rec.get("derived_correct"))
        rec["correct"] = (bool(rec.get("derived_correct")) and rec["result_match"]
                          and not rec["model_derived_columns_presented"])
    elif q.get("expected_derived") == "shares":
        # Phase 4.3 (strict): the shares must be right AND the presented rows must be the raw values of the
        # requested scope (same rows as the gold result) with no model-computed share/ratio columns.
        rec["model_fraction_columns_presented"] = [c for c in (rec.get("columns") or []) if FRACTION_NAME_RE.search(c)]
        rec["correct_lenient"] = bool(rec.get("derived_correct"))
        rec["correct"] = (bool(rec.get("derived_correct")) and rec["result_match"]
                          and not rec["model_fraction_columns_presented"])
    else:
        rec["correct"] = rec["result_match"]
    return rec


def summarise(records: list[dict], routing: list[dict]) -> dict:
    by = defaultdict(list)
    for r in records:
        by[r["category"]].append(r)
    answerable = [r for r in records if r["category"] not in ("unanswerable", "safety")]
    sql_q = [r for r in answerable if r["category"] != "hybrid"]
    hybrid = [r for r in records if r["category"] == "hybrid"]
    routes = defaultdict(list)
    for r in routing:
        routes[r["expected"]].append(r["predicted"] == r["expected"])
    return {
        "questions": len(records),
        "routing_accuracy": rate([r["predicted"] == r["expected"] for r in routing]),
        "routing_by_expected_route": {k: {"n": len(v), "accuracy": rate(v)} for k, v in routes.items()},
        "sql_generation_valid": rate([r.get("generated_valid_sql") for r in answerable]),
        "sql_repaired": sum(1 for r in answerable if r.get("repaired")),
        "execution_success": rate([r.get("executed") for r in answerable]),
        "execution_accuracy": rate([r.get("result_match") for r in answerable]),
        "execution_accuracy_sql_only": rate([r.get("result_match") for r in sql_q]),
        "derived_accuracy": rate([r.get("derived_correct") for r in records if "derived_correct" in r]),
        "chart_accuracy": rate([r.get("chart_correct") for r in records if "chart_correct" in r]),
        "unanswerable_abstention": rate([r["correct"] for r in by["unanswerable"]]),
        "unanswerable_valid_sql_generated": sum(1 for r in by["unanswerable"] if r.get("generated_valid_sql")),
        "safety_refused_and_db_unchanged": rate([r["correct"] for r in by["safety"]]),
        "hybrid_correct": rate([r.get("hybrid_correct") for r in hybrid]),
        "hybrid_sql_part": rate([r.get("result_match") for r in hybrid]),
        "hybrid_document_fact": rate([r.get("document_fact_found") for r in hybrid]),
        "hybrid_document_page_cited": rate([r.get("document_page_cited") for r in hybrid]),
        "hybrid_attribution": rate([r.get("attribution") for r in hybrid]),
        "by_category": {k: {"n": len(v), "correct": rate([r["correct"] for r in v])} for k, v in sorted(by.items())},
        "overall_correct": rate([r["correct"] for r in records]),
        # Phase 4.1 grading of period changes (derived figure only), for comparison with earlier results.
        "overall_correct_lenient": rate([r.get("correct_lenient", r["correct"]) for r in records]),
        "period_change_scope_correct": rate([r["correct"] for r in records if "model_derived_columns_presented" in r]),
        "share_raw_values_correct": rate([r["correct"] for r in records if "model_fraction_columns_presented" in r]),
    }


def render(data: dict) -> str:
    runs = data["runs"]
    s = [r["summary"] for r in runs]

    def across(key):
        values = [x[key] for x in s if x.get(key) is not None]
        if not values:
            return "—"
        return fmt(values[0]) if len(set(values)) == 1 else f"{fmt(sum(values) / len(values))} ({fmt(min(values))}–{fmt(max(values))})"

    first = runs[0]
    metric_rows = [[label, across(key)] for label, key in [
        ("Routing accuracy (all questions incl. 25 document questions)", "routing_accuracy"),
        ("Valid SQL generated (answerable SQL + hybrid)", "sql_generation_valid"),
        ("Execution success", "execution_success"),
        ("Execution accuracy (result = gold result)", "execution_accuracy"),
        ("Execution accuracy, SQL-only questions", "execution_accuracy_sql_only"),
        ("Derived figures correct (period change, shares)", "derived_accuracy"),
        ("Chart type as expected", "chart_accuracy"),
        ("Unanswerable: no result returned", "unanswerable_abstention"),
        ("Safety: refused and database unchanged", "safety_refused_and_db_unchanged"),
        ("Hybrid: fully correct (SQL part + document fact + attribution)", "hybrid_correct"),
        ("Hybrid: SQL part correct", "hybrid_sql_part"),
        ("Hybrid: document fact correct", "hybrid_document_fact"),
        ("Hybrid: document page cited", "hybrid_document_page_cited"),
        ("Overall correct (structured benchmark)", "overall_correct"),
        ("Overall correct, figure-only grading (period change and shares judged on the figures only)", "overall_correct_lenient"),
        ("Period change: figure AND scoped rows AND no model-computed columns", "period_change_scope_correct"),
        ("Shares: figures AND raw-value rows AND no model-computed share columns", "share_raw_values_correct"),
    ]]
    routing_rows = [[k, v["n"], fmt(v["accuracy"])] for k, v in first["summary"]["routing_by_expected_route"].items()]
    misrouted = [r for r in first["routing"] if r["predicted"] != r["expected"]]
    q_rows = []
    for rec in first["records"]:
        q_rows.append([rec["id"], rec["category"], "yes" if rec["correct"] else "no",
                       (rec.get("sql") or ("refused" if rec.get("refused") else (rec.get("error") or "")))[:150].replace("\n", " ").replace("|", "\\|"),
                       rec.get("chart") or "", (rec.get("error") or "")[:80]])
    cats = [[k, v["n"], fmt(v["correct"])] for k, v in first["summary"]["by_category"].items()]
    return "\n".join([
        "# InsightRAG Phase 4: structured data evaluation report",
        "",
        f"_Generated by `scripts/evaluate_structured.py` on {data['generated']}. Benchmark: "
        f"`{data['benchmark']}` ({data['questions']} structured questions) plus the 25 questions of "
        "`benchmark_v2.json` for routing. Earlier benchmarks and results are unchanged._",
        "",
        f"- LLM: {data['llm']} with options {json.dumps(data['generation_options'])}; {len(runs)} runs.",
        f"- Datasets: " + "; ".join(f"`{d['file']}` ({d['kind']}) -> `{d['table']}` ({d['rows']} rows)" for d in data["tables"]) + ".",
        "",
        "## 1. Metrics (measured; range across runs when they differ)",
        "",
        md_table(["Metric", "Value"], metric_rows, ["l", "r"]),
        "",
        "## 2. Routing (run 1)",
        "",
        md_table(["Expected route", "Questions", "Accuracy"], routing_rows, ["l", "r", "r"]),
        "",
        "Misrouted: " + (", ".join(f"{r['id']} (expected {r['expected']}, got {r['predicted']})" for r in misrouted) or "none") + ".",
        "",
        "## 3. By category (run 1)",
        "",
        md_table(["Category", "Questions", "Correct"], cats, ["l", "r", "r"]),
        "",
        "## 4. Per question (run 1)",
        "",
        md_table(["ID", "Category", "Correct", "Generated SQL / outcome", "Chart", "Error"], q_rows, ["l", "l", "l", "l", "l", "l"]),
        "",
        "## 5. Hybrid answers (run 1)",
        "",
        "\n\n".join(f"**{r['id']}**: {r['question']}\n\n```\n{r.get('combined_answer', '')}\n```" for r in first["records"] if r["category"] == "hybrid"),
        "",
        "## 6. Findings and limitations (qualitative)",
        "",
        data.get("findings") or "_No qualitative notes recorded._",
    ])


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--benchmark", type=Path, default=EVAL / "structured_benchmark_v1.json")
    parser.add_argument("--db", type=Path, default=DEFAULT_DB)
    parser.add_argument("--runs", type=int, default=2)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-rag", action="store_true", help="Skip the document pipeline for hybrid questions")
    parser.add_argument("--findings", type=Path, default=EVAL / "phase4_findings.md")
    parser.add_argument("--from-results", type=Path, help="Re-render the report from a saved structured_eval.json")
    parser.add_argument("--output-dir", type=Path, default=OUT, help="Where the JSON results are written")
    parser.add_argument("--report", type=Path, default=EVAL / "phase4_evaluation_report.md")
    args = parser.parse_args(argv)
    os.chdir(PROJECT_ROOT)
    findings = args.findings.read_text(encoding="utf-8").split("\n", 2)[-1].strip() if args.findings.is_file() else ""

    if args.from_results:
        data = json.loads(args.from_results.read_text(encoding="utf-8"))
        data["findings"] = findings
        write_text(args.report, render(data))
        print("Report re-rendered.")
        return 0

    from app.rag.chunker import chunk_pages
    from app.rag.generator import MODEL_NAME, complete
    from app.rag.loader import load_pdf

    benchmark = json.loads(args.benchmark.read_text(encoding="utf-8"))
    tables = build_db(benchmark, args.db)
    options = {"temperature": args.temperature, "seed": args.seed}
    structured = StructuredQA(args.db, complete=lambda prompt: complete(prompt, options)[0])
    vocab = document_vocabulary(chunk_pages(load_pdf(PROJECT_ROOT / "data" / "documents" / "annual_report.pdf")))
    rag = None
    if not args.no_rag:
        from app.rag.pipeline import build_pipeline

        rag = build_pipeline(generation_options=options)
    assistant = Assistant(structured, rag.answer if rag else None, vocab)

    doc_questions = json.loads((EVAL / "benchmark_v2.json").read_text(encoding="utf-8"))["questions"]
    gold_rows = {q["id"]: gold(q["gold_sql"], args.db) for q in benchmark["questions"] if q.get("gold_sql")}
    runs = []
    try:
        for run in range(1, args.runs + 1):
            routing = [{"id": q["id"], "expected": q["route"], "predicted": classify(q["question"], structured.schema, vocab)["route"]}
                       for q in benchmark["questions"]]
            routing += [{"id": q["id"], "expected": "document", "predicted": classify(q["question"], structured.schema, vocab)["route"]}
                        for q in doc_questions]
            records = []
            for q in benchmark["questions"]:
                print(f"run {run} {q['id']}", flush=True)
                before = db_fingerprint(args.db)
                sql_result = rag_result = combined = None
                if q["route"] == "hybrid":
                    combined = assistant.ask(q["question"])
                    sql_result, rag_result = combined.get("sql_result"), combined.get("rag_result")
                    if sql_result is None:  # misrouted: still score the SQL part
                        sql_result = structured.ask(q["question"])
                else:
                    sql_result = structured.ask(q["question"])
                after = db_fingerprint(args.db)
                records.append(evaluate_question(q, sql_result, rag_result, combined, gold_rows.get(q["id"]), before, after))
            runs.append({"run": run, "routing": routing, "records": records, "summary": summarise(records, routing)})
    finally:
        if rag is not None:
            rag.retriever.vector_retriever.vector_store.close()

    data = {"generated": datetime.now(timezone.utc).isoformat(timespec="seconds"), "benchmark": args.benchmark.name,
            "questions": len(benchmark["questions"]), "llm": MODEL_NAME, "generation_options": options,
            "tables": tables, "runs": runs}
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "structured_eval.json").write_text(json.dumps(data, indent=2, default=str) + "\n", encoding="utf-8")
    (args.output_dir / "structured_eval_summary.json").write_text(json.dumps(
        {k: data[k] for k in ("generated", "benchmark", "llm", "generation_options", "tables")}
        | {"runs": [{"run": r["run"], "summary": r["summary"]} for r in runs]}, indent=2, default=str) + "\n", encoding="utf-8")
    data["findings"] = findings
    write_text(args.report, render(data))
    for r in runs:
        print(f"run {r['run']}: " + json.dumps({k: v for k, v in r["summary"].items() if not isinstance(v, dict)}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
