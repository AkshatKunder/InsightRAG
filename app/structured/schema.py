"""Schema discovery for the structured database (Phase 4, Task 2).

Everything reported here is verified against the stored data:

* **primary key**: a column whose values are all present and unique and
  whose name marks it as an identifier (``id``, ``*_id``, ``*_code``,
  ``*_key``). Uniqueness alone is not enough (a unique "sales" column is not
  a key).
* **relationship**: ``child.col -> parent.key`` only when the parent column
  is a verified key, both columns share a name or the child column is named
  ``<parent table or its singular>_<key>``, and every non-null child value
  exists in the parent (checked with a query). Nothing is assumed from names
  alone.
* **sample values**: up to 8 distinct values for low-cardinality TEXT
  columns, so the LLM can match filters ("Europe", "Mining") to real values.

``describe`` renders this as compact text for the NL-to-SQL prompt, using
only database identifiers plus the original column names as comments.
"""

from __future__ import annotations

import re
import sqlite3
from dataclasses import asdict, dataclass, field
from pathlib import Path

from app.structured.ingestion import DEFAULT_DB, META_TABLES

_KEY_NAME_RE = re.compile(r"(?:^id$|_id$|_code$|_key$|^code$|^key$)")
_MAX_SAMPLE_VALUES = 8
_MAX_CATEGORICAL = 25


@dataclass
class ColumnInfo:
    name: str
    original_name: str
    type: str
    missing: int
    distinct: int
    sample_values: list = field(default_factory=list)
    min: object = None
    max: object = None


@dataclass
class TableInfo:
    name: str
    original_name: str
    dataset: str
    source_file: str
    sheet: str | None
    row_count: int
    columns: list[ColumnInfo]
    primary_key: str | None = None


@dataclass
class Relationship:
    child_table: str
    child_column: str
    parent_table: str
    parent_column: str
    verified_rows: int


@dataclass
class Schema:
    tables: list[TableInfo]
    relationships: list[Relationship]

    def table(self, name: str) -> TableInfo | None:
        return next((t for t in self.tables if t.name == name), None)

    def to_dict(self) -> dict:
        return {"tables": [asdict(t) for t in self.tables], "relationships": [asdict(r) for r in self.relationships]}


def _q(identifier: str) -> str:
    return '"' + identifier.replace('"', '""') + '"'


def discover(db_path: Path | str = DEFAULT_DB) -> Schema:
    if not Path(db_path).is_file():
        return Schema([], [])
    conn = sqlite3.connect(f"file:{Path(db_path).as_posix()}?mode=ro", uri=True)
    try:
        conn.row_factory = sqlite3.Row
        try:
            meta = conn.execute(
                "SELECT t.table_name, t.original_name, t.sheet, t.row_count, d.name AS dataset, d.source_file "
                "FROM _tables t JOIN _datasets d USING (dataset_id) ORDER BY t.table_name").fetchall()
        except sqlite3.OperationalError:
            return Schema([], [])
        tables = []
        for row in meta:
            columns = []
            for c in conn.execute("SELECT * FROM _columns WHERE table_name = ? ORDER BY position", (row["table_name"],)):
                info = ColumnInfo(c["column_name"], c["original_name"], c["data_type"], c["missing"], c["distinct_values"])
                col, tab = _q(info.name), _q(row["table_name"])
                if info.type in ("INTEGER", "REAL", "DATE"):
                    info.min, info.max = conn.execute(f"SELECT MIN({col}), MAX({col}) FROM {tab}").fetchone()
                if info.type == "TEXT" and 0 < info.distinct <= _MAX_CATEGORICAL:
                    info.sample_values = [r[0] for r in conn.execute(
                        f"SELECT DISTINCT {col} FROM {tab} WHERE {col} IS NOT NULL ORDER BY {col} LIMIT {_MAX_SAMPLE_VALUES}")]
                columns.append(info)
            table = TableInfo(row["table_name"], row["original_name"], row["dataset"], row["source_file"], row["sheet"],
                              row["row_count"], columns)
            table.primary_key = _primary_key(table)
            tables.append(table)
        return Schema(tables, _relationships(conn, tables))
    finally:
        conn.close()


def _primary_key(table: TableInfo) -> str | None:
    for column in table.columns:
        if _KEY_NAME_RE.search(column.name) and column.missing == 0 and column.distinct == table.row_count and table.row_count:
            return column.name
    return None


def _singular(name: str) -> str:
    return name[:-1] if name.endswith("s") and not name.endswith("ss") else name


def _relationships(conn: sqlite3.Connection, tables: list[TableInfo]) -> list[Relationship]:
    found = []
    for parent in tables:
        if not parent.primary_key:
            continue
        key = parent.primary_key
        names = {key, f"{parent.name}_{key}", f"{_singular(parent.name)}_{key}"}
        if key == "id":
            names |= {f"{parent.name}_id", f"{_singular(parent.name)}_id"}
        for child in tables:
            if child is parent:
                continue
            for column in child.columns:
                if column.name not in names or column.type != next(c.type for c in parent.columns if c.name == key):
                    continue
                orphans, checked = conn.execute(
                    f"SELECT SUM(CASE WHEN p.{_q(key)} IS NULL THEN 1 ELSE 0 END), COUNT(c.{_q(column.name)}) "
                    f"FROM {_q(child.name)} c LEFT JOIN {_q(parent.name)} p ON c.{_q(column.name)} = p.{_q(key)} "
                    f"WHERE c.{_q(column.name)} IS NOT NULL").fetchone()
                if checked and not orphans:
                    found.append(Relationship(child.name, column.name, parent.name, key, checked))
    return found


def describe(schema: Schema) -> str:
    """Compact schema text for the LLM: tables, typed columns, keys, verified joins, sample values."""

    if not schema.tables:
        return "No structured tables are loaded."
    lines = []
    for table in schema.tables:
        source = f"{table.source_file}" + (f", sheet '{table.sheet}'" if table.sheet else "")
        lines.append(f"TABLE {table.name}  -- {table.row_count} rows, from {source}")
        for column in table.columns:
            notes = []
            if column.original_name != column.name:
                notes.append(f'original name "{column.original_name}"')
            if column.name == table.primary_key:
                notes.append("primary key")
            if column.type == "DATE":
                notes.append("ISO date text YYYY-MM-DD")
            if (column.type == "INTEGER" and column.min is not None and 1900 <= column.min <= column.max <= 2100
                    and re.search(r"year|fy|period", column.name)):
                notes.append(f"integer year: filter with {column.name} = {column.max}, never strftime")
            if column.min is not None and column.type in ("INTEGER", "REAL", "DATE"):
                notes.append(f"range {column.min} .. {column.max}")
            if column.sample_values:
                notes.append("values: " + ", ".join(repr(v) for v in column.sample_values))
            if column.missing:
                notes.append(f"{column.missing} missing")
            lines.append(f"  {column.name} {column.type}" + (f"  -- {'; '.join(notes)}" if notes else ""))
    if schema.relationships:
        lines.append("VERIFIED JOINS (every child value exists in the parent):")
        for rel in schema.relationships:
            lines.append(f"  {rel.child_table}.{rel.child_column} -> {rel.parent_table}.{rel.parent_column}")
    else:
        lines.append("No verified relationships between tables: do not join tables unless a join is listed.")
    return "\n".join(lines)


def user_tables(schema: Schema) -> set[str]:
    return {t.name for t in schema.tables if t.name not in META_TABLES}
