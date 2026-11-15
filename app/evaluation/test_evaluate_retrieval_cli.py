"""CLI wiring for scripts/evaluate_retrieval.py, with a fake retriever (no models, no Qdrant)."""

import importlib.util
import json
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[2]
BENCHMARK = ROOT / "data" / "evaluation" / "benchmark_v1.json"


def _cli():
    spec = importlib.util.spec_from_file_location("evaluate_retrieval", ROOT / "scripts" / "evaluate_retrieval.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_cli_writes_detail_and_summary(tmp_path, capsys, monkeypatch):
    cli = _cli()
    monkeypatch.chdir(tmp_path)
    benchmark = json.loads(BENCHMARK.read_text(encoding="utf-8"))
    by_text = {q["question"]: q for q in benchmark["questions"]}
    closed = []
    depths = []

    def fake_factory(pdf_path, depth):
        depths.append(depth)

        def retrieve(text):
            pages = [999] + by_text[text]["expected_pages"]
            return [{"source": "annual_report.pdf", "page": page, "text": "t", "score": 1.0} for page in pages]

        return retrieve, lambda: closed.append(True), {"strategy": "fake"}

    exit_code = cli.main(["--k", "5", "1", "--output-dir", str(tmp_path / "out")], retriever_factory=fake_factory)

    assert exit_code == 0
    assert closed == [True]
    assert depths == [5]

    summary = json.loads((tmp_path / "out" / "retrieval_evaluation_summary.json").read_text(encoding="utf-8"))
    detail = json.loads((tmp_path / "out" / "retrieval_evaluation.json").read_text(encoding="utf-8"))

    assert summary["k_values"] == [1, 5]
    assert summary["num_evaluated"] == 22
    assert summary["num_unanswerable"] == 3
    assert summary["config"]["strategy"] == "fake"
    assert summary["config"]["benchmark"] == "benchmark_v1.json"
    assert summary["overall"]["mrr"] == 0.5
    assert summary["overall"]["recall"]["1"] == 0.0
    assert len(detail["questions"]) == 22
    assert "Recall" in capsys.readouterr().out


def test_cli_rejects_invalid_k():
    cli = _cli()
    with pytest.raises(SystemExit):
        cli.parse_args(["--k", "0"])
