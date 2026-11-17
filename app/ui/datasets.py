"""Structured dataset management for the UI (Phase 5).

Uploads go through the existing ``ingestion.ingest_file`` (type inference, identifier sanitising, one
transaction per dataset). This module adds the UI-level rules:

* extension and size are checked before the file is stored;
* uploaded files are stored under ``dataset_upload_dir`` and never overwritten;
* an existing dataset is replaced only when the user explicitly asks (``replace=True``);
  ``ingest_file`` itself would replace silently.

Previews are fixed ``SELECT * FROM "<table>" LIMIT n`` statements for tables that come from the discovered
schema. They run through ``sql_safety.execute`` (validation, read-only connection, authorizer, timeout,
row limit). The UI never accepts SQL text from the user.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

from app.structured import sql_safety
from app.structured.ingestion import IngestionError, ingest_file, list_tables, safe_identifier
from app.structured.schema import Schema, discover

from app.ui.documents import safe_filename

log = logging.getLogger("insightrag.ui.datasets")
DATASET_EXTENSIONS = (".csv", ".xlsx", ".xlsm")
PREVIEW_ROWS = 20


def _discard(path: Path) -> None:
    """Remove a rejected upload. On Windows a reader may still hold the file briefly; never mask the real error."""

    try:
        path.unlink(missing_ok=True)
    except OSError:
        import gc

        gc.collect()  # releases file handles of failed readers (e.g. openpyxl on a corrupt workbook)
        try:
            path.unlink(missing_ok=True)
        except OSError:
            log.warning("Could not remove a rejected upload")


class DatasetError(ValueError):
    """A user-facing reason why a dataset cannot be added."""


@dataclass
class DatasetUploadResult:
    dataset: str
    stored_as: Path
    tables: list[dict]
    replaced: bool


class DatasetManager:
    def __init__(self, db_path: Path | str, upload_dir: Path | str, max_mb: int):
        self.db_path = Path(db_path)
        self.upload_dir = Path(upload_dir)
        self.max_mb = max_mb

    # ------------------------------------------------------------------ listing
    def schema(self) -> Schema:
        if not self.db_path.is_file():
            return Schema([], [])
        return discover(self.db_path)

    def datasets(self) -> list[dict]:
        """``[{"dataset", "source_file", "tables": [{"name", "sheet", "rows", "columns": [{"name", "original_name", "type", ...}]}]}]``"""

        schema = self.schema()
        meta = {t["table_name"]: t for t in list_tables(self.db_path)}
        grouped: dict[str, dict] = {}
        for table in schema.tables:
            info = meta.get(table.name, {})
            entry = grouped.setdefault(table.dataset, {"dataset": table.dataset, "source_file": table.source_file, "tables": []})
            entry["tables"].append({
                "name": table.name, "original_name": table.original_name, "sheet": table.sheet, "rows": table.row_count,
                "warnings": info.get("warnings"),
                "columns": [{"name": c.name, "original_name": c.original_name, "type": c.type, "missing": c.missing,
                             "distinct": c.distinct, "examples": list(c.sample_values)[:5]} for c in table.columns]})
        return sorted(grouped.values(), key=lambda d: d["dataset"].lower())

    def preview(self, table_name: str, limit: int = PREVIEW_ROWS) -> sql_safety.QueryResult:
        known = {t.name for t in self.schema().tables}
        if table_name not in known:
            raise DatasetError(f"Unknown table '{table_name}'.")
        limit = max(1, min(int(limit), PREVIEW_ROWS * 5))
        return sql_safety.execute(f'SELECT * FROM "{table_name}" LIMIT {limit}', self.db_path, max_rows=limit)

    # ------------------------------------------------------------------ upload
    def exists(self, filename: str) -> bool:
        dataset_id = safe_identifier(Path(safe_filename(filename)).stem, prefix="ds")
        return any(t["dataset_id"] == dataset_id for t in list_tables(self.db_path))

    def upload(self, filename: str, data: bytes, replace: bool = False) -> DatasetUploadResult:
        name = safe_filename(filename)
        suffix = Path(name).suffix.lower()
        if suffix not in DATASET_EXTENSIONS:
            raise DatasetError(f"'{filename}' is not a supported dataset. Upload a .csv or .xlsx file.")
        if not data:
            raise DatasetError(f"'{filename}' is empty.")
        if len(data) > self.max_mb * 1024 * 1024:
            raise DatasetError(f"'{filename}' is {len(data) / 1024 / 1024:.1f} MB; the limit is {self.max_mb} MB.")
        if suffix in (".xlsx", ".xlsm") and not data.startswith(b"PK"):
            raise DatasetError(f"'{filename}' is not a valid Excel workbook (.xlsx files are zip archives).")
        existed = self.exists(name)
        if existed and not replace:
            raise DatasetError(f"A dataset named '{Path(name).stem}' is already loaded. Tick 'Replace existing dataset' "
                               "to replace it, or rename the file.")
        stored = self._store(name, data)
        try:
            tables = ingest_file(stored, self.db_path, dataset_name=Path(name).stem, replace=replace)
        except IngestionError as error:
            _discard(stored)
            raise DatasetError(str(error)) from error
        except Exception as error:
            _discard(stored)
            log.exception("Dataset ingestion failed")
            raise DatasetError(f"'{filename}' could not be loaded ({type(error).__name__}).") from error
        return DatasetUploadResult(Path(name).stem, stored, [
            {"name": t.name, "sheet": t.sheet, "rows": len(t.rows), "columns": [(c.name, c.type) for c in t.columns],
             "warnings": list(t.warnings)} for t in tables], replaced=existed)

    def _store(self, name: str, data: bytes) -> Path:
        """Write to a new file; an existing file of the same name is never overwritten."""

        self.upload_dir.mkdir(parents=True, exist_ok=True)
        stem, suffix = Path(name).stem, Path(name).suffix
        for n in range(1000):
            target = self.upload_dir / (name if n == 0 else f"{stem} ({n}){suffix}")
            try:
                with open(target, "xb") as handle:
                    handle.write(data)
                return target
            except FileExistsError:
                continue
        raise DatasetError("Too many stored copies of this file name.")
