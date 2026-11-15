"""End-to-end wiring of scripts/compare_retrieval.py with fake retrievers (no models, no Qdrant, no PDF)."""

import importlib.util
import json
import sys
from pathlib import Path

import pytest

from app.evaluation.reporting import md_table, svg_grouped_bars
from app.evaluation.retrieval_strategies import RetrievalComponents

ROOT = Path(__file__).resolve().parents[2]


def _cli():
    spec = importlib.util.spec_from_file_location("compare_retrieval", ROOT / "scripts" / "compare_retrieval.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _fake_factory(pdf_path, include_rerank):
    """BM25 finds the first expected page at rank 2; vector misses; hybrid hits at rank 1."""

    bench = json.loads((ROOT / "data" / "evaluation" / "benchmark_v2.json").read_text(encoding="utf-8"))
    by_text = {q["question"]: q for q in bench["questions"]}
    filler = list(range(200, 240))

    def ranking(query, first):
        q = by_text.get(query.replace(" expanded", ""))
        expected = (q or {}).get("expected_pages") or []
        pages = ([filler[0], expected[0]] if first == "second" else expected[:1]) if expected else []
        pages += [p for p in filler if p not in pages]
        return [{"source": "annual_report.pdf", "page": p, "text": f"t{p}", "score": 1.0 / (i + 1)} for i, p in enumerate(pages)]

    def rerank(question, docs, top_k):
        out = [{**d, "reranker_score": float(-i), "final_score": float(-i)} for i, d in enumerate(docs)]
        return out[:top_k]

    components = RetrievalComponents(
        bm25_search=lambda q, k: ranking(q, "second")[:k],
        vector_search=lambda q, k: [{"source": "annual_report.pdf", "page": p, "text": "v", "score": 0.5} for p in range(250, 250 + k)],
        hybrid_search=lambda q, top_k, cand: ranking(q + " expanded", "first")[:top_k],
        expand_query=lambda q: q + " expanded",
        rerank=rerank,
        description={"fake": True},
    )
    return components, lambda: None


def test_cli_runs_scores_and_writes_reports(tmp_path, capsys):
    cli = _cli()
    out, reports = tmp_path / "results", tmp_path / "reports"

    code = cli.main(["--output-dir", str(out), "--report-dir", str(reports), "--pool-size", "12"], components_factory=_fake_factory)

    assert code == 0
    summary = json.loads((out / "strategy_comparison_summary.json").read_text(encoding="utf-8"))
    tables = summary["tables"]["v2"]
    assert tables["hybrid"]["mrr"] == 1.0
    assert tables["bm25"]["mrr"] == 0.5
    assert tables["vector"]["mrr"] == 0.0
    assert summary["counts"] == {"answerable": 22, "unanswerable": 3}
    assert summary["config"]["pool_size"] == 12 and summary["config"]["fake"] is True
    assert [u["id"] for u in summary["unanswerable"]] == ["Q023", "Q024", "Q025"]
    assert summary["failures_top10"]["vector"] and not summary["failures_top10"]["hybrid"]

    reranker = json.loads((out / "reranker_evaluation_summary.json").read_text(encoding="utf-8"))
    # The fake reranker keeps the order, so nothing changes.
    assert reranker["v2"]["hybrid -> hybrid+rerank"]["counts"]["unchanged"] == 22

    runs = json.loads((out / "retrieval_runs.json").read_text(encoding="utf-8"))
    assert len(runs["questions"]) == 25
    detail = json.loads((out / "strategy_comparison.json").read_text(encoding="utf-8"))
    assert "retrieved" not in detail["reports"]["v2"]["bm25"]["questions"][0]

    text = (reports / "retrieval_strategy_comparison.md").read_text(encoding="utf-8")
    assert "BM25 vs vector vs hybrid" in text and "Q023" in text
    assert (reports / "reranker_evaluation.md").is_file()
    assert (reports / "charts" / "strategy_recall.svg").read_text(encoding="utf-8").startswith("<svg")

    # Re-scoring the saved runs without models gives identical tables.
    code = cli.main(["--runs", str(out / "retrieval_runs.json"), "--output-dir", str(tmp_path / "again"),
                     "--report-dir", str(tmp_path / "again_reports"), "--pool-size", "12"],
                    components_factory=lambda *a: pytest.fail("models must not load"))
    again = json.loads((tmp_path / "again" / "strategy_comparison_summary.json").read_text(encoding="utf-8"))
    assert code == 0 and again["tables"] == summary["tables"]


def test_cli_rejects_pool_smaller_than_k():
    with pytest.raises(SystemExit):
        _cli().parse_args(["--pool-size", "5", "--k", "1", "10"])


def test_md_table_and_svg_helpers():
    table = md_table(["A", "B"], [["x|y", 0.12345], ["z", None]])
    assert "x\\|y" in table and "0.123" in table and "—" in table
    svg = svg_grouped_bars("t", ["g1", "g2"], {"s1": [0.1, 0.2], "s2": [0.3, 1.5]})
    assert svg.count("<path") == 4 and "<title>g2 · s2: 1.500</title>" in svg
    with pytest.raises(ValueError):
        svg_grouped_bars("t", ["g1"], {"s1": [0.1, 0.2]})
