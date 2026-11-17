"""Application settings for the InsightRAG UI (Phase 5).

Every value comes from an environment variable (optionally from ``.env`` via python-dotenv) and falls
back to the project's existing defaults. No secret is needed or stored; the Ollama client itself reads
``OLLAMA_HOST``.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from pathlib import Path

from app.rag.generator import MODEL_NAME
from app.structured.ingestion import DEFAULT_DB, MAX_FILE_BYTES

DETERMINISTIC_OPTIONS = {"temperature": 0, "seed": 42}


def _load_dotenv() -> None:
    try:
        from dotenv import load_dotenv
    except ImportError:  # optional
        return
    load_dotenv(override=False)


def redact_url(url: str) -> str:
    """Drop any ``user:password@`` part so a host URL can be shown or logged."""

    return re.sub(r"(?<=//)[^/@]*@", "", url or "")


def _env(name: str, default: str) -> str:
    value = os.environ.get(name)
    return value.strip() if value and value.strip() else default


def _env_int(name: str, default: int) -> int:
    try:
        return int(_env(name, str(default)))
    except ValueError:
        return default


@dataclass
class AppConfig:
    documents_dir: Path = Path("data/documents")
    qdrant_path: Path = Path("data/qdrant")
    db_path: Path = DEFAULT_DB
    dataset_upload_dir: Path = Path("data/structured/uploads")
    model: str = MODEL_NAME
    ollama_host: str = "http://localhost:11434"
    max_pdf_mb: int = 50
    max_dataset_mb: int = MAX_FILE_BYTES // (1024 * 1024)
    deterministic: bool = True
    generation_options: dict = field(default_factory=lambda: dict(DETERMINISTIC_OPTIONS))

    @classmethod
    def from_env(cls, load_dotenv: bool = True) -> "AppConfig":
        if load_dotenv:
            _load_dotenv()
        deterministic = _env("INSIGHTRAG_DETERMINISTIC", "1").lower() not in ("0", "false", "no", "off")
        return cls(
            documents_dir=Path(_env("INSIGHTRAG_DOCUMENTS_DIR", "data/documents")),
            qdrant_path=Path(_env("INSIGHTRAG_QDRANT_PATH", "data/qdrant")),
            db_path=Path(_env("INSIGHTRAG_DB_PATH", str(DEFAULT_DB))),
            dataset_upload_dir=Path(_env("INSIGHTRAG_DATASET_UPLOAD_DIR", "data/structured/uploads")),
            model=_env("INSIGHTRAG_MODEL", MODEL_NAME),
            ollama_host=_env("OLLAMA_HOST", "http://localhost:11434"),
            max_pdf_mb=_env_int("INSIGHTRAG_MAX_PDF_MB", 50),
            max_dataset_mb=min(_env_int("INSIGHTRAG_MAX_DATASET_MB", MAX_FILE_BYTES // (1024 * 1024)),
                               MAX_FILE_BYTES // (1024 * 1024)),  # never above the ingestion limit
            deterministic=deterministic,
            generation_options=dict(DETERMINISTIC_OPTIONS) if deterministic else {},
        )

    def public_dict(self) -> dict:
        """Settings shown in the UI (paths and model names only; nothing sensitive)."""

        return {"Documents folder": str(self.documents_dir), "Vector index (Qdrant, local)": str(self.qdrant_path),
                "Structured database (SQLite)": str(self.db_path), "Dataset uploads": str(self.dataset_upload_dir),
                "Ollama model": self.model, "Ollama host": redact_url(self.ollama_host),
                "Max PDF size": f"{self.max_pdf_mb} MB", "Max dataset size": f"{self.max_dataset_mb} MB",
                "Generation": "deterministic (temperature 0, seed 42)" if self.deterministic else "Ollama defaults"}
