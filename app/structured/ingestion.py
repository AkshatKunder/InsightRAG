"""CSV / Excel ingestion into a local SQLite database (Phase 4, Task 1).

* CSV (delimiter sniffed, UTF-8 with BOM tolerated) and ``.xlsx`` (every
  non-empty sheet becomes its own table; requires ``openpyxl``).
* The first non-empty row is the header. Original column names are kept in
  metadata; database identifiers are normalised to ``[a-z0-9_]``, unique, not
  starting with a digit and never an SQL keyword.
* Column types are inferred from the values: INTEGER, REAL, DATE (ISO
  ``YYYY-MM-DD``; Excel dates and common day/month formats are normalised),
  TEXT. Empty cells and common null markers become NULL; missing values are
  counted per column.
* Validation rejects empty files, files without a header or data rows,
  duplicate/blank headers that cannot be disambiguated, rows wider than the
  header, and files above the size/row limits. Problems are reported as
  ``IngestionError`` with a clear message; no partial table is written.
* Metadata lives in ``_datasets``, ``_tables`` and ``_columns`` tables in
  the same database (original names, types, missing counts, dimensions,
  source file, sheet, load time).
"""

from __future__ import annotations

import csv
import hashlib
import io
import re
import sqlite3
from dataclasses import dataclass, field
from datetime import date, datetime, timezone
from pathlib import Path

DEFAULT_DB = Path("data/structured/insightrag.db")
MAX_FILE_BYTES = 50 * 1024 * 1024
MAX_ROWS = 1_000_000
NULL_MARKERS = {"", "na", "n/a", "null", "none", "nan", "-", "--", "#n/a"}
SQL_KEYWORDS = {
    "select", "from", "where", "group", "order", "by", "table", "index", "insert", "update", "delete", "drop",
    "create", "alter", "join", "on", "as", "and", "or", "not", "null", "limit", "having", "union", "values",
    "into", "set", "case", "when", "then", "else", "end", "is", "in", "like", "between", "distinct", "all",
    "primary", "key", "default", "check", "references", "foreign",
}
_DATE_FORMATS = ("%Y-%m-%d", "%Y/%m/%d", "%d/%m/%Y", "%d-%m-%Y", "%d.%m.%Y", "%Y-%m", "%b %Y", "%B %Y")
META_TABLES = ("_datasets", "_tables", "_columns")


class IngestionError(ValueError):
    """The file cannot be ingested safely; nothing was written."""


@dataclass
class Column:
    original_name: str
    name: str
    type: str
    missing: int
    distinct: int


@dataclass
class TableData:
    original_name: str
    name: str
    columns: list[Column]
    rows: list[list]
    source_file: str
    sheet: str | None = None
    warnings: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Identifiers
# ---------------------------------------------------------------------------


def safe_identifier(name: str, taken: set[str] | None = None, prefix: str = "col") -> str:
    """Lower-case ``[a-z0-9_]`` identifier, unique within ``taken`` and not an SQL keyword."""

    ident = re.sub(r"[^a-z0-9]+", "_", str(name).strip().lower().replace("%", " pct ")).strip("_")
    if not ident:
        ident = prefix
    if ident[0].isdigit():
        ident = f"{prefix}_{ident}"
    if ident in SQL_KEYWORDS or ident.startswith("sqlite_") or ident.startswith("_"):
        ident = f"{ident}_{prefix}".lstrip("_")
    taken = taken if taken is not None else set()
    base, n = ident, 2
    while ident in taken:
        ident = f"{base}_{n}"
        n += 1
    taken.add(ident)
    return ident


# ---------------------------------------------------------------------------
# Value parsing and type inference
# ---------------------------------------------------------------------------


def _is_null(value) -> bool:
    return value is None or (isinstance(value, str) and value.strip().lower() in NULL_MARKERS)


def _parse_number(value):
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return value
    text = str(value).strip().replace(",", "").replace(" ", "")
    negative = text.startswith("(") and text.endswith(")")
    text = text.strip("()").lstrip("$€£")
    if re.fullmatch(r"-?\d+", text):
        number = int(text)
    elif re.fullmatch(r"-?\d*\.\d+(?:[eE]-?\d+)?|-?\d+(?:\.\d+)?[eE]-?\d+", text):
        number = float(text)
    else:
        return None
    return -number if negative else number


def _parse_date(value):
    if isinstance(value, datetime):
        return value.date().isoformat()
    if isinstance(value, date):
        return value.isoformat()
    text = str(value).strip()
    for fmt in _DATE_FORMATS:
        try:
            return datetime.strptime(text, fmt).date().isoformat()
        except ValueError:
            continue
    return None


def infer_type(values: list) -> str:
    present = [v for v in values if not _is_null(v)]
    if not present:
        return "TEXT"
    numbers = [_parse_number(v) for v in present]
    if all(n is not None for n in numbers):
        # INTEGER only when every value is a whole number written without decimals
        # (an Excel cell holding 3.0 still counts as whole).
        whole = all(isinstance(n, int) or (isinstance(v, float) and n.is_integer()) for n, v in zip(numbers, present))
        return "INTEGER" if whole else "REAL"
    if all(_parse_date(v) is not None for v in present):
        return "DATE"
    return "TEXT"


def convert(value, sql_type: str):
    if _is_null(value):
        return None
    if sql_type in ("INTEGER", "REAL"):
        number = _parse_number(value)
        return int(number) if sql_type == "INTEGER" else float(number)
    if sql_type == "DATE":
        return _parse_date(value)
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    return str(value).strip()


# ---------------------------------------------------------------------------
# Readers
# ---------------------------------------------------------------------------


def _check_file(path: Path) -> None:
    if not path.is_file():
        raise IngestionError(f"File not found: {path}")
    size = path.stat().st_size
    if size == 0:
        raise IngestionError(f"{path.name} is empty.")
    if size > MAX_FILE_BYTES:
        raise IngestionError(f"{path.name} is {size / 1e6:.1f} MB; the limit is {MAX_FILE_BYTES / 1e6:.0f} MB.")


def read_csv(path: Path) -> list[list]:
    raw = path.read_bytes()
    try:
        text = raw.decode("utf-8-sig")
    except UnicodeDecodeError:
        try:
            text = raw.decode("cp1252")
        except UnicodeDecodeError as error:
            raise IngestionError(f"{path.name} is not valid UTF-8 or Windows-1252 text.") from error
    if "\x00" in text:
        raise IngestionError(f"{path.name} looks like a binary file, not CSV.")
    sample = text[:20000]
    try:
        dialect = csv.Sniffer().sniff(sample, delimiters=",;\t|")
    except csv.Error:
        dialect = csv.excel
    try:
        return [row for row in csv.reader(io.StringIO(text), dialect)]
    except csv.Error as error:
        raise IngestionError(f"{path.name} is malformed CSV: {error}") from error


def read_xlsx(path: Path) -> dict[str, list[list]]:
    try:
        import openpyxl
    except ImportError as error:  # pragma: no cover - dependency is in requirements.txt
        raise IngestionError("Excel support requires the 'openpyxl' package.") from error
    try:
        workbook = openpyxl.load_workbook(path, read_only=True, data_only=True)
    except Exception as error:
        raise IngestionError(f"{path.name} is not a readable .xlsx workbook: {error}") from error
    sheets = {}
    try:
        for sheet in workbook.worksheets:
            sheets[sheet.title] = [list(row) for row in sheet.iter_rows(values_only=True)]
    finally:
        workbook.close()
    return sheets


def build_table(rows: list[list], table_name: str, source_file: str, sheet: str | None = None) -> TableData:
    """Validate a grid of cells (header first) and infer the table's columns."""

    rows = [r for r in rows if any(not _is_null(c) for c in r)]
    if not rows:
        raise IngestionError(f"{source_file}{f' [{sheet}]' if sheet else ''}: no data.")
    header, body = rows[0], rows[1:]
    while header and _is_null(header[-1]) and all(len(r) < len(header) or _is_null(r[len(header) - 1]) for r in body):
        header = header[:-1]
    if not header:
        raise IngestionError(f"{source_file}: missing header row.")
    if not body:
        raise IngestionError(f"{source_file}{f' [{sheet}]' if sheet else ''}: header but no data rows.")
    if len(body) > MAX_ROWS:
        raise IngestionError(f"{source_file}: {len(body)} rows exceed the limit of {MAX_ROWS}.")
    warnings = []
    width = len(header)
    for number, row in enumerate(body, start=2):
        extra = [c for c in row[width:] if not _is_null(c)]
        if extra:
            raise IngestionError(f"{source_file}: row {number} has {len(row)} values but the header has {width} columns.")
    body = [list(r[:width]) + [None] * (width - len(r)) for r in body]

    originals = []
    for index, name in enumerate(header):
        name = "" if _is_null(name) else str(name).strip()
        if not name:
            name = f"column_{index + 1}"
            warnings.append(f"Blank header in column {index + 1} named '{name}'.")
        originals.append(name)
    if len({o.lower() for o in originals}) != len(originals):
        warnings.append("Duplicate column names were made unique.")

    taken: set[str] = set()
    columns = []
    for index, original in enumerate(originals):
        values = [row[index] for row in body]
        sql_type = infer_type(values)
        present = [v for v in values if not _is_null(v)]
        columns.append(Column(original, safe_identifier(original, taken), sql_type,
                              missing=len(values) - len(present), distinct=len({str(v) for v in present})))
    converted = [[convert(row[i], columns[i].type) for i in range(width)] for row in body]
    return TableData(table_name, "", columns, converted, source_file, sheet, warnings)


# ---------------------------------------------------------------------------
# Store
# ---------------------------------------------------------------------------

_META_DDL = """
CREATE TABLE IF NOT EXISTS _datasets (
    dataset_id TEXT PRIMARY KEY, name TEXT NOT NULL, source_file TEXT NOT NULL,
    file_sha256 TEXT NOT NULL, file_type TEXT NOT NULL, loaded_at TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS _tables (
    table_name TEXT PRIMARY KEY, dataset_id TEXT NOT NULL REFERENCES _datasets(dataset_id),
    original_name TEXT NOT NULL, sheet TEXT, row_count INTEGER NOT NULL, column_count INTEGER NOT NULL,
    warnings TEXT);
CREATE TABLE IF NOT EXISTS _columns (
    table_name TEXT NOT NULL REFERENCES _tables(table_name), position INTEGER NOT NULL,
    column_name TEXT NOT NULL, original_name TEXT NOT NULL, data_type TEXT NOT NULL,
    missing INTEGER NOT NULL, distinct_values INTEGER NOT NULL, PRIMARY KEY (table_name, column_name));
"""


def connect(db_path: Path | str = DEFAULT_DB) -> sqlite3.Connection:
    path = Path(db_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path)
    conn.executescript(_META_DDL)
    return conn


def ingest_file(path: Path | str, db_path: Path | str = DEFAULT_DB, dataset_name: str | None = None,
                replace: bool = True) -> list[TableData]:
    """Load a CSV or .xlsx file into SQLite. Returns the stored tables (with their final names)."""

    path = Path(path)
    _check_file(path)
    suffix = path.suffix.lower()
    dataset_name = dataset_name or path.stem
    if suffix == ".csv":
        tables = [build_table(read_csv(path), dataset_name, path.name)]
    elif suffix in (".xlsx", ".xlsm"):
        tables = []
        for sheet, grid in read_xlsx(path).items():
            try:
                tables.append(build_table(grid, f"{dataset_name} {sheet}", path.name, sheet))
            except IngestionError as error:
                if "no data" in str(error):
                    continue  # empty sheets are skipped
                raise
        if not tables:
            raise IngestionError(f"{path.name}: no sheet contains data.")
    else:
        raise IngestionError(f"Unsupported file type '{suffix}'. Use .csv or .xlsx.")

    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    dataset_id = safe_identifier(dataset_name, prefix="ds")
    conn = connect(db_path)
    try:
        with conn:  # one transaction: all tables or none
            existing = {r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
            old = [r[0] for r in conn.execute("SELECT table_name FROM _tables WHERE dataset_id = ?", (dataset_id,))]
            if old and not replace:
                raise IngestionError(f"Dataset '{dataset_name}' is already loaded (use replace=True).")
            for name in old:
                conn.execute(f'DROP TABLE IF EXISTS "{name}"')
                conn.execute("DELETE FROM _columns WHERE table_name = ?", (name,))
                existing.discard(name)
            conn.execute("DELETE FROM _tables WHERE dataset_id = ?", (dataset_id,))
            conn.execute("DELETE FROM _datasets WHERE dataset_id = ?", (dataset_id,))
            conn.execute("INSERT INTO _datasets VALUES (?, ?, ?, ?, ?, ?)",
                         (dataset_id, dataset_name, path.name, digest, suffix.lstrip("."),
                          datetime.now(timezone.utc).isoformat(timespec="seconds")))
            taken = set(existing) | set(META_TABLES)
            for table in tables:
                table.name = safe_identifier(table.original_name, taken, prefix="t")
                cols = ", ".join(f'"{c.name}" {c.type if c.type != "DATE" else "TEXT"}' for c in table.columns)
                conn.execute(f'CREATE TABLE "{table.name}" ({cols})')
                marks = ", ".join("?" for _ in table.columns)
                conn.executemany(f'INSERT INTO "{table.name}" VALUES ({marks})', table.rows)
                conn.execute("INSERT INTO _tables VALUES (?, ?, ?, ?, ?, ?, ?)",
                             (table.name, dataset_id, table.original_name, table.sheet, len(table.rows),
                              len(table.columns), "; ".join(table.warnings) or None))
                conn.executemany("INSERT INTO _columns VALUES (?, ?, ?, ?, ?, ?, ?)", [
                    (table.name, i, c.name, c.original_name, c.type, c.missing, c.distinct)
                    for i, c in enumerate(table.columns)])
    finally:
        conn.close()
    return tables


def list_tables(db_path: Path | str = DEFAULT_DB) -> list[dict]:
    if not Path(db_path).is_file():
        return []
    conn = connect(db_path)
    try:
        conn.row_factory = sqlite3.Row
        return [dict(r) for r in conn.execute(
            "SELECT t.*, d.name AS dataset, d.source_file FROM _tables t JOIN _datasets d USING (dataset_id) ORDER BY table_name")]
    finally:
        conn.close()
