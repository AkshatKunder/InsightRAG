"""scripts/build_phase2_report.py: reads saved results only (no models)."""

import importlib.util
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "data" / "evaluation" / "results"


def _module():
    spec = importlib.util.spec_from_file_location("build_phase2_report", ROOT / "scripts" / "build_phase2_report.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_missing_results_are_reported(tmp_path):
    with pytest.raises(SystemExit, match="Missing result files"):
        _module().load_results(tmp_path)


def test_findings_sections_are_split(tmp_path):
    path = tmp_path / "findings.md"
    path.write_text("# Title\n\nintro\n\n## Executive summary\nA\nB\n\n## Recommendations\n- x\n", encoding="utf-8")
    sections = _module()._findings_sections(path)
    assert sections == {"Executive summary": "A\nB", "Recommendations": "- x"}


def test_benchmark_info_counts_excerpts():
    m = _module()
    assert m.benchmark_info(ROOT / "data" / "evaluation" / "benchmark_v1.json")["excerpts"] == 48
    assert m.benchmark_info(ROOT / "data" / "evaluation" / "benchmark_v2.json")["excerpts"] == 52


@pytest.mark.skipif(
    not all((RESULTS / name).is_file() for name in (
        "strategy_comparison_summary.json", "reranker_evaluation_summary.json", "answer_evaluation_summary.json",
        "numerical_evaluation.json", "abstention_evaluation.json", "retrieval_evaluation_summary.json")),
    reason="evaluation results not generated",
)
def test_report_builds_from_saved_results(tmp_path):
    m = _module()
    results = tmp_path / "results"
    results.mkdir()
    for name in m.REQUIRED.values():
        (results / name).write_text((RESULTS / name).read_text(encoding="utf-8"), encoding="utf-8")
    output = tmp_path / "report.md"

    assert m.main(["--results-dir", str(results), "--output", str(output), "--findings", str(tmp_path / "none.md")]) == 0

    text = output.read_text(encoding="utf-8")
    for heading in ("## 1. Executive summary", "## 4. Retrieval strategy comparison", "## 5. Reranker comparison",
                    "## 6. Answer correctness", "## 7. Citation quality", "## 8. Numerical accuracy",
                    "## 9. Abstention performance", "## 10. Category-level analysis", "## 11. Failure-case analysis",
                    "## 12. Limitations", "## 13. Recommendations", "## 14. Conclusions"):
        assert heading in text
    summary = json.loads((results / "phase2_summary.json").read_text(encoding="utf-8"))
    assert set(summary["retrieval"]) >= {"bm25", "vector", "hybrid", "hybrid+rerank"}
    assert 0 <= summary["answer_accuracy"] <= 1
