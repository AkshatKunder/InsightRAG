"""Unit tests for the Retriever-to-Generator orchestration.

This is deliberately a mocked pipeline test, not a local-Ollama integration
test. The helper mirrors the original standalone script's two-step flow.
"""

from __future__ import annotations

import pytest


def _run_rag_pipeline(question: str, retriever, generator, top_k: int = 5):
    """Mirror the existing script: retrieve evidence, then generate from it."""
    results = retriever.search(question, top_k=top_k)
    return generator.generate(question, results)


class _FakeRetriever:
    def __init__(self, results=None, error: Exception | None = None):
        self.results = [] if results is None else results
        self.error = error
        self.calls = []

    def search(self, question: str, top_k: int = 5):
        self.calls.append((question, top_k))
        if self.error:
            raise self.error
        return self.results


class _FakeGenerator:
    def __init__(self, answer: str = "", error: Exception | None = None):
        self.answer = answer
        self.error = error
        self.calls = []

    def generate(self, question: str, chunks):
        self.calls.append((question, chunks))
        if self.error:
            raise self.error
        return self.answer


def test_pipeline_forwards_question_and_top_k_to_retriever():
    retriever = _FakeRetriever()
    generator = _FakeGenerator(answer="No evidence found.")

    _run_rag_pipeline("How many employees?", retriever, generator, top_k=7)

    assert retriever.calls == [("How many employees?", 7)]


def test_pipeline_passes_original_question_and_retrieved_chunks_to_generator():
    chunks = [{"text": "The company employed 100 people.", "source": "report.pdf", "page": 4, "score": 0.9}]
    retriever = _FakeRetriever(results=chunks)
    generator = _FakeGenerator(answer="It employed 100 people.")

    _run_rag_pipeline("How many employees?", retriever, generator)

    assert generator.calls == [("How many employees?", chunks)]


def test_pipeline_returns_generated_answer_unchanged():
    retriever = _FakeRetriever(results=[])
    generator = _FakeGenerator(answer="The answer from the generator.")

    answer = _run_rag_pipeline("Question", retriever, generator)

    assert answer == "The answer from the generator."


def test_pipeline_handles_empty_retrieval_without_fabricating_evidence():
    retriever = _FakeRetriever(results=[])
    generator = _FakeGenerator(answer="I could not find this information in the provided document.")

    answer = _run_rag_pipeline("Unknown question", retriever, generator)

    assert answer == "I could not find this information in the provided document."
    assert generator.calls == [("Unknown question", [])]


def test_pipeline_preserves_retrieved_source_metadata():
    chunks = [
        {"text": "Total sales were 10 million.", "source": "annual.pdf", "page": 12, "score": 0.8},
        {"text": "Workforce was 100.", "source": "annual.pdf", "page": 13, "score": 0.7},
    ]
    retriever = _FakeRetriever(results=chunks)
    generator = _FakeGenerator(answer="Synthetic answer")

    _run_rag_pipeline("Question", retriever, generator)

    forwarded_chunks = generator.calls[0][1]
    assert forwarded_chunks is chunks
    assert [(chunk["source"], chunk["page"]) for chunk in forwarded_chunks] == [
        ("annual.pdf", 12),
        ("annual.pdf", 13),
    ]


def test_empty_generated_answer_is_returned_as_is():
    retriever = _FakeRetriever(results=[])
    generator = _FakeGenerator(answer="")

    assert _run_rag_pipeline("Question", retriever, generator) == ""


def test_retrieval_exception_propagates_without_calling_generator():
    retriever = _FakeRetriever(error=RuntimeError("retrieval failed"))
    generator = _FakeGenerator(answer="should not run")

    with pytest.raises(RuntimeError, match="retrieval failed"):
        _run_rag_pipeline("Question", retriever, generator)

    assert generator.calls == []


def test_generation_exception_propagates_after_retrieval():
    chunks = [{"text": "evidence", "source": "report.pdf", "page": 1, "score": 0.9}]
    retriever = _FakeRetriever(results=chunks)
    generator = _FakeGenerator(error=RuntimeError("generation failed"))

    with pytest.raises(RuntimeError, match="generation failed"):
        _run_rag_pipeline("Question", retriever, generator)

    assert retriever.calls == [("Question", 5)]
    assert generator.calls == [("Question", chunks)]
