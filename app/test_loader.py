"""Fast, isolated tests for PDF page loading."""

from __future__ import annotations

from pathlib import Path

import pytest
from pypdf import PdfWriter

from app.rag import loader


class _TextPage:
    """Minimal page double that behaves like pypdf's page text interface."""

    def __init__(self, text: str | None):
        self._text = text

    def extract_text(self) -> str | None:
        return self._text


@pytest.fixture
def temporary_pdf(tmp_path: Path) -> Path:
    """Create a structurally valid, tiny PDF without relying on project data."""
    path = tmp_path / "sample_report.pdf"
    writer = PdfWriter()
    writer.add_blank_page(width=72, height=72)
    with path.open("wb") as output:
        writer.write(output)
    return path


@pytest.fixture
def fake_reader(monkeypatch: pytest.MonkeyPatch, temporary_pdf: Path) -> None:
    """Provide deterministic extracted text while retaining a real input file."""
    class FakeReader:
        def __init__(self, path: Path):
            assert Path(path) == temporary_pdf
            self.pages = [_TextPage("First page text"), _TextPage("Second page text")]

    monkeypatch.setattr(loader, "PdfReader", FakeReader)


def test_loads_existing_valid_pdf(temporary_pdf: Path):
    pages = loader.load_pdf(temporary_pdf)

    # The fixture is intentionally blank, so the loader correctly excludes it
    # while proving that a real PDF is opened without touching project data.
    assert pages == []


def test_extracts_page_text(temporary_pdf: Path, fake_reader: None):
    pages = loader.load_pdf(temporary_pdf)

    assert [page["text"] for page in pages] == ["First page text", "Second page text"]


def test_preserves_source_filename_and_page_number(temporary_pdf: Path, fake_reader: None):
    pages = loader.load_pdf(temporary_pdf)

    assert [page["source"] for page in pages] == ["sample_report.pdf", "sample_report.pdf"]
    assert [page["page"] for page in pages] == [1, 2]


def test_missing_pdf_raises_file_not_found(tmp_path: Path):
    missing_pdf = tmp_path / "missing.pdf"

    with pytest.raises(FileNotFoundError):
        loader.load_pdf(missing_pdf)
