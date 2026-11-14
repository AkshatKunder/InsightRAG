"""Tests for prompt construction and answer clean-up (no LLM calls)."""

from __future__ import annotations

import pytest

from app.rag import generator
from app.rag.generator import build_prompt, clean_answer


@pytest.mark.parametrize("raw, expected", [
    (
        "According to SOURCE 2, DOCUMENT: annual_report.pdf, PAGE: 95, and SOURCE 3, "
        "DOCUMENT: annual_report.pdf, PAGE: 191, the total sales of ArcelorMittal in 2025 were $61,352 million.",
        "The total sales of ArcelorMittal in 2025 were $61,352 million.",
    ),
    ("Based on excerpt 1, the company operates in 14 countries.", "The company operates in 14 countries."),
    ("Total sales were $61,352 million (Excerpt 2).", "Total sales were $61,352 million."),
    ("Total sales were $61,352 million [Source 1, Page 95].", "Total sales were $61,352 million."),
])
def test_clean_answer_strips_context_label_references(raw, expected):
    assert clean_answer(raw) == expected


@pytest.mark.parametrize("answer", [
    "ArcelorMittal had approximately 125,554 employees in 2025.",
    "Page count is not a metric used in the report.",
    "I could not find this information in the provided document.",
])
def test_clean_answer_leaves_ordinary_answers_unchanged(answer):
    assert clean_answer(answer) == answer


def test_prompt_omits_page_numbers_and_file_names():
    prompt = build_prompt(
        "What were total sales?",
        [{"text": "Total sales were 61,352.", "source": "annual_report.pdf", "page": 95}],
    )

    assert "Total sales were 61,352." in prompt
    assert "EXCERPT 1" in prompt
    assert "annual_report.pdf" not in prompt
    assert "PAGE" not in prompt


def test_generate_returns_cleaned_model_output(monkeypatch):
    sent = {}

    def fake_chat(model, messages):
        sent["model"], sent["messages"] = model, messages
        return {"message": {"content": "According to excerpt 1, total sales were $61,352 million."}}

    monkeypatch.setattr(generator.ollama, "chat", fake_chat)

    answer = generator.Generator().generate("What were total sales?", [{"text": "Total sales were $61,352 million."}])

    assert answer == "Total sales were $61,352 million."
    assert sent["model"] == generator.MODEL_NAME
    assert "What were total sales?" in sent["messages"][0]["content"]
