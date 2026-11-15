"""InsightRAG assistant with document RAG, structured SQL and hybrid answers (Phase 4).

    python -m app.assistant --ingest data/structured/samples/segment_kpis.csv
    python -m app.assistant --schema
    python -m app.assistant                       # interactive
    python -m app.assistant --ask "What was the total order revenue per region in the orders table?"

``app/main.py`` (document-only RAG) is unchanged and still works.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from app.structured.engine import Assistant, StructuredQA, document_vocabulary
from app.structured.ingestion import DEFAULT_DB, IngestionError, ingest_file
from app.structured.schema import describe, discover


def build_assistant(db_path: Path = DEFAULT_DB, generation_options: dict | None = None, with_rag: bool = True) -> tuple[Assistant, object]:
    from app.rag.generator import complete

    structured = StructuredQA(db_path, complete=lambda prompt: complete(prompt, generation_options)[0])
    if not with_rag:
        return Assistant(structured, None), None
    from app.rag.pipeline import build_pipeline

    rag = build_pipeline(generation_options=generation_options)
    return Assistant(structured, rag.answer, document_vocabulary(rag.retriever.chunks)), rag


def show(result: dict) -> None:
    print(f"\n[route: {result['route']}] {result['routing']['reason']}")
    print("\nAnswer:\n" + (result["answer"] or ""))
    sql = result.get("sql_result")
    if sql and sql.get("sql"):
        print("\nSQL:\n" + sql["sql"])
        if sql.get("columns"):
            print("\nRows:")
            print(" | ".join(sql["columns"]))
            for row in sql["rows"][:20]:
                print(" | ".join("" if v is None else str(v) for v in row))
        if sql.get("chart", {}).get("type"):
            print(f"\nChart: {sql['chart']['type']} ({sql['chart']['reason']})")
    if result.get("sources"):
        print("\nSources: " + json.dumps(result["sources"], default=str))


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--db", type=Path, default=DEFAULT_DB)
    parser.add_argument("--ingest", type=Path, nargs="*", help="CSV/XLSX files to load")
    parser.add_argument("--schema", action="store_true", help="Print the discovered schema")
    parser.add_argument("--ask", help="Answer one question and exit")
    parser.add_argument("--no-rag", action="store_true", help="Structured data only (no document pipeline)")
    args = parser.parse_args(argv)

    for path in args.ingest or []:
        try:
            for table in ingest_file(path, args.db):
                print(f"Loaded {path.name}{f' [{table.sheet}]' if table.sheet else ''} -> table {table.name} "
                      f"({len(table.rows)} rows, {len(table.columns)} columns)")
                for warning in table.warnings:
                    print(f"  warning: {warning}")
        except IngestionError as error:
            print(f"Not loaded: {error}")
    if args.schema:
        print(describe(discover(args.db)))
    if args.ingest is not None or args.schema:
        if not args.ask:
            return 0

    assistant, rag = build_assistant(args.db, with_rag=not args.no_rag)
    try:
        if args.ask:
            show(assistant.ask(args.ask))
            return 0
        while True:
            question = input("\nQuestion (or 'exit'): ").strip()
            if question.lower() == "exit":
                break
            if question:
                show(assistant.ask(question))
    finally:
        if rag is not None:
            rag.retriever.vector_retriever.vector_store.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
