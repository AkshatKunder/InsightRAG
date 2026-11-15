"""Keep data/evaluation/benchmark_v1.json structurally valid.

Runs the structural checks from scripts/validate_benchmark.py without the PDF
(the PDF is local-only and not committed).
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
BENCHMARK = ROOT / "data" / "evaluation" / "benchmark_v1.json"


def _validator():
    spec = importlib.util.spec_from_file_location("validate_benchmark", ROOT / "scripts" / "validate_benchmark.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_benchmark_passes_structural_validation():
    validator = _validator()
    report = validator.Report()

    validator.validate(json.loads(BENCHMARK.read_text(encoding="utf-8")), report)

    assert report.errors == []
    assert report.checks and all(passed for _, passed in report.checks)


def test_validator_detects_structural_problems():
    validator = _validator()
    data = json.loads(BENCHMARK.read_text(encoding="utf-8"))
    data["questions"][1]["id"] = data["questions"][0]["id"]
    data["questions"][-1]["expected_answer"] = "$70 billion"
    report = validator.Report()

    validator.validate(data, report)

    assert any("duplicate id" in error for error in report.errors)
    assert any("ABSTAIN" in error for error in report.errors)
