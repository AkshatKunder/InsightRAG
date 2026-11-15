"""Validate the structure and consistency of an InsightRAG benchmark file.

Usage (from the project root):

    python scripts/validate_benchmark.py
    python scripts/validate_benchmark.py data/evaluation/benchmark_v1.json --pdf data/documents/annual_report.pdf
    python scripts/validate_benchmark.py --no-pdf

Structural checks always run. When the source PDF is available, every
evidence excerpt is also checked to appear verbatim (whitespace, quote style
and case normalised) on the page it cites. This confirms that cited text
exists where claimed; it does not judge whether the expected answer is
semantically correct.

Exit code 0 when every check passes, 1 otherwise.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BENCHMARK = PROJECT_ROOT / "data" / "evaluation" / "benchmark_v1.json"
DEFAULT_PDF = PROJECT_ROOT / "data" / "documents" / "annual_report.pdf"

VALID_CATEGORIES = {"factual", "numerical", "financial", "comparative", "multi_hop", "unanswerable"}
MIN_QUESTIONS, MAX_QUESTIONS = 20, 30
ID_PATTERN = re.compile(r"^Q\d{3}$")
ABSTAIN = "ABSTAIN"

_QUOTES = str.maketrans({"‘": "'", "’": "'", "“": '"', "”": '"'})


def normalise(text: str) -> str:
    return re.sub(r"\s+", " ", text.translate(_QUOTES)).strip().lower()


class Report:
    def __init__(self) -> None:
        self.errors: list[str] = []
        self.checks: list[tuple[str, bool]] = []

    def error(self, message: str) -> None:
        self.errors.append(message)

    def check(self, label: str, errors_before: int) -> None:
        self.checks.append((label, len(self.errors) == errors_before))


def _is_page(value, page_count: int) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and 1 <= value <= page_count


def validate(data: dict, report: Report, page_texts: dict[int, str] | None = None) -> None:
    questions = data.get("questions")
    if not isinstance(questions, list):
        report.error("'questions' must be a list")
        return
    page_count = data.get("document_pages")
    if not isinstance(page_count, int) or page_count < 1:
        report.error("'document_pages' must be a positive integer")
        return

    start = len(report.errors)
    if not MIN_QUESTIONS <= len(questions) <= MAX_QUESTIONS:
        report.error(f"expected {MIN_QUESTIONS}-{MAX_QUESTIONS} questions, found {len(questions)}")
    if data.get("total_questions") != len(questions):
        report.error(f"total_questions={data.get('total_questions')} but {len(questions)} questions present")
    report.check(f"Question count within {MIN_QUESTIONS}-{MAX_QUESTIONS} and matches total_questions", start)

    start = len(report.errors)
    ids = [q.get("id") for q in questions]
    for qid, count in Counter(ids).items():
        if count > 1:
            report.error(f"duplicate id {qid}")
    for qid in ids:
        if not isinstance(qid, str) or not ID_PATTERN.match(qid):
            report.error(f"invalid id {qid!r} (expected Q###)")
    report.check("All IDs unique and well-formed", start)

    start = len(report.errors)
    declared = set(data.get("categories", []))
    if declared != VALID_CATEGORIES:
        report.error(f"'categories' must be exactly {sorted(VALID_CATEGORIES)}")
    for q in questions:
        if q.get("category") not in VALID_CATEGORIES:
            report.error(f"{q.get('id')}: invalid category {q.get('category')!r}")
    actual_counts = Counter(q.get("category") for q in questions)
    if data.get("category_counts") and dict(actual_counts) != data["category_counts"]:
        report.error(f"category_counts {data['category_counts']} do not match actual {dict(actual_counts)}")
    report.check("All categories valid and category_counts consistent", start)

    start = len(report.errors)
    for q in questions:
        if not str(q.get("question") or "").strip():
            report.error(f"{q.get('id')}: empty question text")
    report.check("All questions have non-empty text", start)

    start = len(report.errors)
    for q in questions:
        qid, answerable, category = q.get("id"), q.get("answerable"), q.get("category")
        if not isinstance(answerable, bool):
            report.error(f"{qid}: 'answerable' must be true or false")
            continue
        if (category == "unanswerable") != (answerable is False):
            report.error(f"{qid}: category '{category}' inconsistent with answerable={answerable}")
        answer = str(q.get("expected_answer") or "").strip()
        if answerable:
            if not answer or answer == ABSTAIN:
                report.error(f"{qid}: answerable question needs a concrete expected_answer")
        else:
            if answer != ABSTAIN:
                report.error(f"{qid}: unanswerable question must have expected_answer '{ABSTAIN}'")
            if q.get("expected_pages"):
                report.error(f"{qid}: unanswerable question must have empty expected_pages")
            if not str(q.get("unanswerable_reason") or "").strip():
                report.error(f"{qid}: unanswerable question needs an unanswerable_reason")
    report.check("Answerability, expected answers and abstentions consistent", start)

    start = len(report.errors)
    for q in questions:
        if not q.get("answerable"):
            continue
        qid = q.get("id")
        pages = q.get("expected_pages")
        if not isinstance(pages, list) or not pages:
            report.error(f"{qid}: answerable question needs expected_pages")
            continue
        evidence = q.get("supporting_evidence")
        if not isinstance(evidence, list) or not evidence:
            report.error(f"{qid}: answerable question needs supporting_evidence")
            continue
        for item in evidence:
            if not str(item.get("excerpt") or "").strip():
                report.error(f"{qid}: empty evidence excerpt")
            if item.get("page") not in pages:
                report.error(f"{qid}: evidence page {item.get('page')} not listed in expected_pages {pages}")
        facts = q.get("expected_facts") or []
        if not facts:
            report.error(f"{qid}: answerable question needs expected_facts")
        for fact in facts:
            fact_pages = fact.get("pages") or []
            if not fact_pages:
                report.error(f"{qid}: fact '{fact.get('metric')}' has no pages")
            elif not set(fact_pages) & set(pages):
                report.error(f"{qid}: fact '{fact.get('metric')}' pages {fact_pages} not covered by expected_pages {pages}")
    report.check("Answerable questions have evidence, facts and covering expected_pages", start)

    start = len(report.errors)
    for q in questions:
        qid = q.get("id")
        all_pages = list(q.get("expected_pages") or []) + list(q.get("distractor_pages") or [])
        all_pages += [item.get("page") for key in ("supporting_evidence", "distractor_evidence") for item in q.get(key) or []]
        all_pages += [page for fact in q.get("expected_facts") or [] for page in fact.get("pages") or []]
        for page in all_pages:
            if not _is_page(page, page_count):
                report.error(f"{qid}: page {page!r} outside 1-{page_count}")
    report.check(f"All page numbers within 1-{page_count}", start)

    start = len(report.errors)
    for q in questions:
        if q.get("category") != "multi_hop":
            continue
        qid = q.get("id")
        pages = set(q.get("expected_pages") or [])
        facts = q.get("expected_facts") or []
        evidence_pages = {item.get("page") for item in q.get("supporting_evidence") or []}
        if len(pages) < 2:
            report.error(f"{qid}: multi_hop question needs at least two expected_pages")
        if len(facts) < 2:
            report.error(f"{qid}: multi_hop question needs at least two expected_facts")
        if len(evidence_pages) < 2:
            report.error(f"{qid}: multi_hop question needs evidence from at least two pages")
        for fact in facts:
            if not set(fact.get("pages") or []) & evidence_pages:
                report.error(f"{qid}: fact '{fact.get('metric')}' has no supporting excerpt on any of its pages")
    report.check("Multi-hop questions span multiple facts and pages", start)

    start = len(report.errors)
    for q in questions:
        if q.get("category") != "comparative":
            continue
        qid = q.get("id")
        periods = {fact.get("period") for fact in q.get("expected_facts") or []}
        if len(periods) < 2:
            report.error(f"{qid}: comparative question needs facts for at least two periods")
        comparison = q.get("expected_comparison") or {}
        if comparison.get("direction") not in {"increase", "decrease", "unchanged"}:
            report.error(f"{qid}: comparative question needs expected_comparison.direction")
    report.check("Comparative questions record both periods and a direction", start)

    if page_texts is None:
        return
    start = len(report.errors)
    checked = 0
    for q in questions:
        for key in ("supporting_evidence", "distractor_evidence"):
            for item in q.get(key) or []:
                page, excerpt = item.get("page"), str(item.get("excerpt") or "")
                checked += 1
                if normalise(excerpt) not in normalise(page_texts.get(page, "")):
                    report.error(f"{q.get('id')}: excerpt not found on page {page}: {excerpt[:70]!r}")
    report.check(f"All {checked} evidence excerpts found verbatim on their cited PDF pages", start)


def load_page_texts(pdf_path: Path) -> tuple[dict[int, str], int]:
    sys.path.insert(0, str(PROJECT_ROOT))
    from pypdf import PdfReader

    from app.rag.loader import load_pdf

    return {page["page"]: page["text"] for page in load_pdf(pdf_path)}, len(PdfReader(pdf_path).pages)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("benchmark", nargs="?", default=str(DEFAULT_BENCHMARK))
    parser.add_argument("--pdf", default=str(DEFAULT_PDF), help="Source PDF for excerpt verification")
    parser.add_argument("--no-pdf", action="store_true", help="Skip excerpt verification against the PDF")
    args = parser.parse_args(argv)

    report = Report()
    try:
        data = json.loads(Path(args.benchmark).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        print(f"FAIL  JSON could not be loaded: {exc}")
        return 1
    report.checks.append(("JSON loads without errors", True))

    page_texts = None
    pdf_path = Path(args.pdf)
    if args.no_pdf:
        pdf_note = "skipped (--no-pdf)"
    elif not pdf_path.is_file():
        pdf_note = f"skipped (PDF not found at {pdf_path})"
    else:
        page_texts, pdf_pages = load_page_texts(pdf_path)
        pdf_note = f"{pdf_path.name}, {pdf_pages} pages"
        if data.get("document_pages") != pdf_pages:
            report.error(f"document_pages={data.get('document_pages')} but the PDF has {pdf_pages} pages")
        report.checks.append(("document_pages matches the PDF", data.get("document_pages") == pdf_pages))

    validate(data, report, page_texts)

    questions = data.get("questions") or []
    answerable = sum(1 for q in questions if q.get("answerable") is True)
    print(f"Benchmark: {data.get('benchmark_name')}  ({args.benchmark})")
    print(f"Excerpt verification: {pdf_note}")
    print(f"Questions: {len(questions)}  answerable: {answerable}  unanswerable: {len(questions) - answerable}")
    print("By category: " + ", ".join(f"{cat}={count}" for cat, count in sorted(Counter(q.get('category') for q in questions).items())))
    print()
    for label, passed in report.checks:
        print(f"{'PASS' if passed else 'FAIL'}  {label}")
    if report.errors:
        print(f"\n{len(report.errors)} problem(s):")
        for message in report.errors:
            print(f"  - {message}")
        return 1
    print("\nAll checks passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
