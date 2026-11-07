"""Deterministic unit tests for page chunking."""

import pytest

from app.rag.chunker import chunk_pages


def _page(text: str, page: int = 1, source: str = "sample.pdf") -> dict[str, object]:
    return {"text": text, "source": source, "page": page}


def test_creates_chunks_from_a_sample_page():
    chunks = chunk_pages([_page("one two three four")], chunk_size=2, overlap=0)

    assert [chunk["text"] for chunk in chunks] == ["one two", "three four"]


def test_chunk_size_is_respected():
    chunks = chunk_pages([_page("one two three four five six seven")], chunk_size=3, overlap=0)

    assert [len(chunk["text"].split()) for chunk in chunks] == [3, 3, 1]


def test_consecutive_chunks_have_the_requested_overlap():
    chunks = chunk_pages([_page("one two three four five six")], chunk_size=4, overlap=2)

    assert chunks[0]["text"].split()[-2:] == chunks[1]["text"].split()[:2]
    assert [chunk["text"] for chunk in chunks] == ["one two three four", "three four five six", "five six"]


def test_preserves_source_filename_and_page_metadata():
    chunks = chunk_pages([_page("one two three", page=7, source="report.pdf")], chunk_size=2, overlap=0)

    assert [(chunk["source"], chunk["page"]) for chunk in chunks] == [("report.pdf", 7), ("report.pdf", 7)]


def test_empty_page_text_produces_no_chunks():
    assert chunk_pages([_page("")]) == []


def test_short_text_produces_exactly_one_chunk():
    chunks = chunk_pages([_page("one two")], chunk_size=10, overlap=0)

    assert len(chunks) == 1
    assert chunks[0]["text"] == "one two"


def test_multiple_pages_are_processed_independently():
    chunks = chunk_pages([
        _page("one two", page=1),
        _page("three four", page=2),
    ], chunk_size=2, overlap=0)

    assert [(chunk["text"], chunk["page"]) for chunk in chunks] == [
        ("one two", 1),
        ("three four", 2),
    ]


@pytest.mark.parametrize(
    ("chunk_size", "overlap"),
    [(0, 0), (-1, 0), (2, -1), (2, 2), (2, 3)],
)
def test_invalid_chunk_parameters_raise_value_error(chunk_size: int, overlap: int):
    with pytest.raises(ValueError):
        chunk_pages([_page("one two")], chunk_size=chunk_size, overlap=overlap)
