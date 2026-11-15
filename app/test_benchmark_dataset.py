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


BENCHMARK_V2 = ROOT / "data" / "evaluation" / "benchmark_v2.json"


def test_benchmark_v2_passes_structural_validation():
    validator = _validator()
    report = validator.Report()

    validator.validate(json.loads(BENCHMARK_V2.read_text(encoding="utf-8")), report)

    assert report.errors == []
    assert report.checks and all(passed for _, passed in report.checks)


def test_benchmark_v2_differs_from_v1_only_in_q017_q018_page3():
    v1 = json.loads(BENCHMARK.read_text(encoding="utf-8"))
    v2 = json.loads(BENCHMARK_V2.read_text(encoding="utf-8"))

    changed = [a["id"] for a, b in zip(v1["questions"], v2["questions"]) if a != b]
    assert changed == ["Q017", "Q018"]
    for qid in changed:
        old = next(q for q in v1["questions"] if q["id"] == qid)
        new = next(q for q in v2["questions"] if q["id"] == qid)
        # Primary evidence unchanged; page 3 added only as an alternative fact page.
        assert new["expected_pages"] == old["expected_pages"] == [95]
        assert new["supporting_evidence"] == old["supporting_evidence"]
        assert [f["pages"] for f in new["expected_facts"]] == [[95, 3], [95, 3]]
        assert [f["value"] for f in new["expected_facts"]] == [f["value"] for f in old["expected_facts"]]
        assert {item["page"] for item in new["alternative_evidence"]} == {3}
        stripped = {k: v for k, v in new.items() if k not in ("expected_facts", "alternative_evidence")}
        assert stripped == {k: v for k, v in old.items() if k != "expected_facts"}


def test_validator_checks_alternative_evidence_pages():
    validator = _validator()
    data = json.loads(BENCHMARK_V2.read_text(encoding="utf-8"))
    q017 = next(q for q in data["questions"] if q["id"] == "Q017")
    q017["alternative_evidence"].append({"page": 200, "excerpt": "x"})
    report = validator.Report()

    validator.validate(data, report)

    assert any("alternative evidence page 200" in error for error in report.errors)


def test_validator_detects_structural_problems():
    validator = _validator()
    data = json.loads(BENCHMARK.read_text(encoding="utf-8"))
    data["questions"][1]["id"] = data["questions"][0]["id"]
    data["questions"][-1]["expected_answer"] = "$70 billion"
    report = validator.Report()

    validator.validate(data, report)

    assert any("duplicate id" in error for error in report.errors)
    assert any("ABSTAIN" in error for error in report.errors)
