"""Phase 4 tests: ingestion, schema discovery, SQL safety, NL-to-SQL, analytics, charts, routing and the
assistant. Temporary files/databases and fake LLMs only (no Ollama, no network)."""

import sqlite3
from pathlib import Path

import pytest

from app.structured import analytics, charts, nl2sql, sql_safety
from app.structured.engine import ABSTENTION, Assistant, StructuredQA, render_answer
from app.structured.ingestion import IngestionError, build_table, infer_type, ingest_file, list_tables, safe_identifier
from app.structured.router import classify
from app.structured.schema import describe, discover

SEGMENTS_CSV = (
    "Segment,Year,Sales (USD m),Employees (FTE)\n"
    "Europe,2024,29952,48544\nEurope,2025,28793,43878\n"
    "Mining,2024,2663,4758\nMining,2025,3232,5918\n"
)


@pytest.fixture
def db(tmp_path):
    db_path = tmp_path / "test.db"
    (tmp_path / "segment_kpis.csv").write_text(SEGMENTS_CSV, encoding="utf-8")
    ingest_file(tmp_path / "segment_kpis.csv", db_path)
    customers = tmp_path / "customers.csv"
    customers.write_text("Customer ID,Name,Region\nC1,Alpha Works,Europe\nC2,Beta Mills,Asia\n", encoding="utf-8")
    orders = tmp_path / "orders.csv"
    orders.write_text("Order ID,Customer ID,Order date,Amount\nO1,C1,2025-01-15,100\nO2,C1,2025-02-03,250.5\n"
                      "O3,C2,2024-12-30,80\n", encoding="utf-8")
    ingest_file(customers, db_path)
    ingest_file(orders, db_path)
    return db_path


# ---------------------------------------------------------------- Task 1: ingestion


def test_identifiers_are_safe_unique_and_keep_meaning():
    taken = set()
    assert safe_identifier("Sales (USD m)", taken) == "sales_usd_m"
    assert safe_identifier("Sales (USD m)", taken) == "sales_usd_m_2"
    assert safe_identifier("2025 revenue") == "col_2025_revenue"
    assert safe_identifier("select") == "select_col"
    assert safe_identifier("Margin %") == "margin_pct"
    assert safe_identifier("Year") == "year"


@pytest.mark.parametrize("values,expected", [
    (["1", "2", "3"], "INTEGER"), (["1.5", "2"], "REAL"), (["1,234", "(5)"], "INTEGER"),
    (["2025-01-31", "31/12/2024"], "DATE"), (["Europe", "1"], "TEXT"), (["", "NA", None], "TEXT"),
])
def test_type_inference(values, expected):
    assert infer_type(values) == expected


def test_csv_ingestion_metadata_and_missing_values(tmp_path):
    path = tmp_path / "data.csv"
    path.write_text("Name;Value;When\nA;1;2025-01-01\nB;;2025-02-01\n", encoding="utf-8")
    tables = ingest_file(path, tmp_path / "x.db")
    table = tables[0]
    assert [c.type for c in table.columns] == ["TEXT", "INTEGER", "DATE"]
    assert table.columns[1].missing == 1
    stored = list_tables(tmp_path / "x.db")
    assert stored[0]["row_count"] == 2 and stored[0]["column_count"] == 3


def test_excel_multiple_sheets_and_empty_sheet(tmp_path):
    openpyxl = pytest.importorskip("openpyxl")
    from datetime import datetime

    book = openpyxl.Workbook()
    book.active.title = "plants"
    book.active.append(["Plant ID", "Capacity"])
    book.active.append(["P1", 3.5])
    sheet = book.create_sheet("output")
    sheet.append(["Plant ID", "Date", "Tonnes"])
    sheet.append(["P1", datetime(2025, 3, 1), 12])
    book.create_sheet("empty")
    path = tmp_path / "ops.xlsx"
    book.save(path)

    tables = ingest_file(path, tmp_path / "x.db")
    assert [(t.sheet, t.name) for t in tables] == [("plants", "ops_plants"), ("output", "ops_output")]
    assert tables[1].rows == [["P1", "2025-03-01", 12]]


@pytest.mark.parametrize("content,message", [
    ("", "empty"),
    ("a,b\n", "no data rows"),
    ("a,b\n1,2,3\n", "row 2 has 3 values"),
])
def test_malformed_files_are_rejected(tmp_path, content, message):
    path = tmp_path / "bad.csv"
    path.write_text(content, encoding="utf-8")
    with pytest.raises(IngestionError, match=message):
        ingest_file(path, tmp_path / "x.db")
    assert not list_tables(tmp_path / "x.db")  # nothing written


def test_unsupported_and_binary_files(tmp_path):
    (tmp_path / "a.json").write_text("{}", encoding="utf-8")
    with pytest.raises(IngestionError, match="Unsupported"):
        ingest_file(tmp_path / "a.json", tmp_path / "x.db")
    (tmp_path / "b.csv").write_bytes(b"a,b\x00\n1,2\n")
    with pytest.raises(IngestionError, match="binary"):
        ingest_file(tmp_path / "b.csv", tmp_path / "x.db")


def test_reingest_replaces_dataset(tmp_path):
    path = tmp_path / "d.csv"
    path.write_text("a\n1\n", encoding="utf-8")
    ingest_file(path, tmp_path / "x.db")
    path.write_text("a\n1\n2\n", encoding="utf-8")
    ingest_file(path, tmp_path / "x.db")
    assert [t["row_count"] for t in list_tables(tmp_path / "x.db")] == [2]


def test_blank_and_duplicate_headers_get_warnings():
    table = build_table([["A", "", "a"], [1, 2, 3]], "t", "f.csv")
    assert [c.name for c in table.columns] == ["a", "column_2", "a_2"]
    assert len(table.warnings) == 2


# ---------------------------------------------------------------- Task 2: schema discovery


def test_schema_keys_relationships_and_description(db):
    schema = discover(db)
    customers, orders = schema.table("customers"), schema.table("orders")
    assert customers.primary_key == "customer_id" and orders.primary_key == "order_id"
    assert schema.table("segment_kpis").primary_key is None  # no identifier column: no key assumed
    assert [(r.child_table, r.child_column, r.parent_table) for r in schema.relationships] == [("orders", "customer_id", "customers")]
    text = describe(schema)
    assert 'original name "Sales (USD m)"' in text and "values: 'Europe', 'Mining'" in text
    assert "orders.customer_id -> customers.customer_id" in text and "integer year" in text


def test_relationship_not_assumed_when_values_missing(tmp_path):
    db_path = tmp_path / "r.db"
    (tmp_path / "parents.csv").write_text("parent_id,name\nP1,a\n", encoding="utf-8")
    (tmp_path / "children.csv").write_text("child_id,parent_id\nC1,P1\nC2,P9\n", encoding="utf-8")
    ingest_file(tmp_path / "parents.csv", db_path)
    ingest_file(tmp_path / "children.csv", db_path)
    assert discover(db_path).relationships == []


# ---------------------------------------------------------------- Task 3: SQL safety


@pytest.mark.parametrize("sql", [
    "DELETE FROM orders", "DROP TABLE orders", "UPDATE orders SET amount = 0", "INSERT INTO orders VALUES (1)",
    "ALTER TABLE orders ADD x INT", "SELECT 1; DROP TABLE orders", "PRAGMA table_info(orders)",
    "ATTACH DATABASE 'x.db' AS x", "SELECT * FROM orders -- comment", "WITH x AS (DELETE FROM orders) SELECT 1",
    "REPLACE INTO orders VALUES (1)", "", "VACUUM",
])
def test_destructive_or_unsafe_sql_rejected(db, sql):
    with pytest.raises(sql_safety.UnsafeSQL):
        sql_safety.validate(sql, db)
    assert sql_safety.execute(sql, db).error


def test_metadata_and_system_tables_are_not_readable(db):
    for sql in ("SELECT * FROM _columns", "SELECT * FROM sqlite_master", "SELECT * FROM _datasets"):
        assert "touches something other" in sql_safety.execute(sql, db).error


def test_valid_select_runs_read_only_with_limits(db):
    result = sql_safety.execute("SELECT customer_id, SUM(amount) AS total FROM orders GROUP BY customer_id ORDER BY total DESC;", db)
    assert result.ok and result.columns == ["customer_id", "total"] and result.rows == [("C1", 350.5), ("C2", 80.0)]
    limited = sql_safety.execute("SELECT * FROM orders", db, max_rows=2)
    assert len(limited.rows) == 2 and limited.truncated
    assert "Invalid SQL" in sql_safety.execute("SELECT nope FROM orders", db).error
    # Strings containing keywords are fine.
    assert sql_safety.execute("SELECT 'drop table' AS x, replace(name, 'a', 'b') FROM customers", db).ok


def test_readonly_connection_blocks_writes_even_if_validation_bypassed(db):
    conn = sql_safety._connect_readonly(db)
    with pytest.raises(sqlite3.OperationalError):
        conn.execute("DELETE FROM orders")
    conn.close()


def test_timeout_is_enforced(db):
    slow = "WITH RECURSIVE n(x) AS (SELECT 1 UNION ALL SELECT x + 1 FROM n) SELECT COUNT(*) FROM n"
    result = sql_safety.execute(slow, db, timeout_s=0.2)
    assert result.error == "Query exceeded the time limit."


# ---------------------------------------------------------------- Task 3: NL-to-SQL


def test_extract_sql_from_model_output():
    assert nl2sql.extract_sql("```sql\nSELECT 1;\n```\nThis returns one.") == "SELECT 1"
    assert nl2sql.extract_sql("Here you go: SELECT a FROM t WHERE b = 'x;y'; more") == "SELECT a FROM t WHERE b = 'x;y'"
    assert nl2sql.extract_sql("NO_SQL") == nl2sql.NO_SQL
    assert nl2sql.extract_sql("I don't know") is None


def test_generation_repairs_invalid_sql_and_uses_schema(db):
    prompts = []
    answers = iter(["SELECT nope FROM orders", "SELECT SUM(amount) AS total FROM orders"])

    def fake(prompt):
        prompts.append(prompt)
        return next(answers)

    gen = nl2sql.generate_sql("Total amount?", discover(db), db, fake)
    assert gen.sql == "SELECT SUM(amount) AS total FROM orders" and len(gen.attempts) == 2
    assert "TABLE orders" in prompts[0] and "no such column" in prompts[1]


def test_lint_rejects_date_functions_on_numeric_columns(db):
    answers = iter(["SELECT SUM(sales_usd_m) FROM segment_kpis WHERE strftime('%Y', year) = '2025'",
                    "SELECT SUM(sales_usd_m) FROM segment_kpis WHERE year = 2025"])
    gen = nl2sql.generate_sql("Sales 2025?", discover(db), db, lambda p: next(answers))
    assert gen.sql.endswith("year = 2025") and "numeric column 'year'" in gen.attempts[0]["error"]


def test_generation_gives_up_without_executing(db):
    gen = nl2sql.generate_sql("q", discover(db), db, lambda p: "DROP TABLE orders")
    assert gen.sql is None and len(gen.attempts) == 3  # never accepted, never executed
    # Only the first statement is ever extracted: a trailing DELETE is discarded, not executed.
    assert nl2sql.generate_sql("q", discover(db), db, lambda p: "SELECT 1; DELETE FROM orders").sql == "SELECT 1"
    no = nl2sql.generate_sql("q", discover(db), db, lambda p: "NO_SQL")
    assert no.no_sql and no.sql is None


# ---------------------------------------------------------------- Task 6: analytics


def test_period_changes_per_group_with_traceability():
    rows = [("Europe", 2024, 29952), ("Europe", 2025, 28793), ("Mining", 2024, 2663), ("Mining", 2025, 3232)]
    changes = analytics.period_changes(["segment", "year", "sales_usd_m"], rows)
    europe = next(c for c in changes if c["group"] == "Europe")
    assert europe["absolute_change"] == -1159 and europe["percent_change"] == -3.87
    assert europe["source_rows"] == [["Europe", 2024, 29952], ["Europe", 2025, 28793]]


def test_period_changes_refuse_duplicated_periods_and_zero_base():
    assert analytics.period_changes(["year", "v"], [(2024, 1), (2024, 2)]) == []
    assert analytics.percent_change(0, 5) is None


def test_shares_and_unit_checks():
    shares = analytics.shares(["segment", "employees"], [("A", 30), ("B", 10)])
    assert [s["share_percent"] for s in shares] == [75.0, 25.0]
    assert analytics.shares(["segment", "v"], [("A", -1), ("B", 2)]) == []
    assert analytics.same_unit("sales_usd_m", "revenue_usd") and not analytics.same_unit("sales_usd_m", "employees_fte")


def test_column_comparison_only_for_same_metric_different_periods():
    out = analytics.analyse("How did sales change?", ["sales_2024", "sales_2025"], [(100, 110)])
    assert out["column_comparison"]["percent_change"] == 10.0
    assert "column_comparison" not in analytics.analyse("How did sales change?", ["sales_2025", "sales_change"], [(100, 0)])


# ---------------------------------------------------------------- Task 7: charts


def test_chart_choices():
    q = "Revenue per region"
    assert charts.choose_chart(q, "SELECT region, SUM(x)", ["region", "revenue"], [("EU", 1), ("US", 2)])["type"] == "bar"
    monthly = [("2025-01", 5), ("2025-02", 6), ("2025-03", 4)]
    assert charts.choose_chart("monthly revenue", "SELECT", ["month", "revenue"], monthly)["type"] == "line"
    share = charts.choose_chart("share of employees by segment", "SELECT s, SUM(e)", ["segment", "employees"], [("A", 3), ("B", 1)])
    assert share["type"] == "pie"
    assert charts.choose_chart("share", "SELECT s, AVG(e)", ["segment", "employees"], [("A", 3), ("B", 1)])["type"] == "bar"
    assert charts.choose_chart(q, "", ["revenue"], [(5,)])["type"] is None
    assert charts.choose_chart(q, "", ["region", "revenue_usd", "employees"], [("a", 1, 2), ("b", 2, 3)])["type"] is None  # mixed units
    many = [(f"c{i}", i) for i in range(40)]
    assert charts.choose_chart(q, "", ["c", "revenue"], many)["type"] is None


# ---------------------------------------------------------------- Task 5: routing


VOCAB = {"sales", "europe", "segment", "employees", "report", "annual", "arcelormittal", "steel", "orders", "amount"}


@pytest.mark.parametrize("question,route", [
    ("What were ArcelorMittal's total sales in 2025?", "document"),
    ("Who is the CEO?", "document"),
    ("What is the total amount in the orders table?", "sql"),
    ("How many orders did Alpha Works place?", "sql"),  # distinctive value from the data
    ("What is the total of the orders data per region?", "sql"),
    ("Compare Europe sales in the segment kpi table with the annual report.", "hybrid"),
])
def test_routing(db, question, route):
    assert classify(question, discover(db), VOCAB)["route"] == route


def test_routing_without_tables_is_document(tmp_path):
    assert classify("total in the orders table", discover(tmp_path / "none.db"))["route"] == "document"


# ---------------------------------------------------------------- Tasks 4-5: engine and assistant


def test_structured_qa_end_to_end(db):
    qa = StructuredQA(db, complete=lambda p: "SELECT c.region, SUM(o.amount) AS total FROM orders o JOIN customers c "
                                              "ON o.customer_id = c.customer_id GROUP BY c.region")
    out = qa.ask("Total amount per region in the orders table?")
    assert out["error"] is None and out["rows"] == [["Asia", 80.0], ["Europe", 350.5]]
    assert "Asia" in out["answer"] and out["sql"].startswith("SELECT") and out["chart"]["type"] == "bar"
    assert {s["table"] for s in out["sources"]} == {"orders", "customers"}


def test_write_requests_refused_before_any_sql(db):
    calls = []
    qa = StructuredQA(db, complete=lambda p: calls.append(p) or "SELECT 1")
    out = qa.ask("Delete all rows from the orders table")
    assert out["sql"] is None and "read-only" in out["error"] and calls == []


def test_failed_query_never_produces_numbers(db):
    qa = StructuredQA(db, complete=lambda p: "NO_SQL")
    out = qa.ask("What is the profit margin in the orders table?")
    assert out["sql"] is None and out["rows"] == [] and not any(ch.isdigit() for ch in out["answer"])


def test_change_question_missing_a_period_is_flagged(db):
    qa = StructuredQA(db, complete=lambda p: "SELECT year, sales_usd_m FROM segment_kpis WHERE segment = 'Europe' AND year = 2025")
    out = qa.ask("How did Europe sales change between 2024 and 2025 in the segment kpi table?")
    assert out["incomplete"] == {"missing_periods": ["2024"]} and "cannot be calculated" in out["answer"]
    full = StructuredQA(db, complete=lambda p: "SELECT year, sales_usd_m FROM segment_kpis WHERE segment = 'Europe' ORDER BY year")
    result = full.ask("How did Europe sales change between 2024 and 2025 in the segment kpi table?")
    assert "-3.87%" in result["answer"] and "incomplete" not in result


def test_render_answer_for_errors_and_empty_results():
    assert "could not answer" in render_answer(sql_safety.QueryResult("x", error="Query failed: boom"), {})
    assert "no matching records" in render_answer(sql_safety.QueryResult("x", columns=["a"], rows=[]), {})


def test_assistant_hybrid_attribution_and_abstention(db):
    qa = StructuredQA(db, complete=lambda p: "SELECT sales_usd_m FROM segment_kpis WHERE segment = 'Europe' AND year = 2025")
    rag = lambda q: {"answer": "Europe's 2025 sales were $28,793 million.", "validation": {"overall_status": "SUPPORTED"},
                     "citations": [{"source": "annual_report.pdf", "page": 95}]}
    out = Assistant(qa, rag, VOCAB).ask("Compare Europe sales in the segment kpi table with the annual report.")
    assert out["route"] == "hybrid" and "From the structured data" in out["answer"] and "From the document (pages 95)" in out["answer"]
    assert {s["type"] for s in out["sources"]} == {"table", "document"}

    nothing = Assistant(StructuredQA(db, complete=lambda p: "NO_SQL"),
                        lambda q: {"answer": ABSTENTION, "validation": {"overall_status": "ABSTAINED"}, "citations": []}, VOCAB)
    out = nothing.ask("Compare Europe sales in the segment kpi table with the annual report.")
    assert out["abstained"] and out["answer"] == ABSTENTION


def test_assistant_document_route_uses_rag_unchanged(db):
    seen = []
    rag = lambda q: seen.append(q) or {"answer": "A", "validation": {"overall_status": "SUPPORTED"}, "citations": []}
    out = Assistant(StructuredQA(db, complete=lambda p: pytest.fail("no SQL for document questions")), rag, VOCAB).ask("Who is the CEO?")
    assert out["route"] == "document" and seen == ["Who is the CEO?"] and out["sql_result"] is None


def test_split_hybrid_gives_each_side_its_own_part():
    from app.structured.router import split_hybrid

    parts = split_hybrid("What was Mining's operating income in 2025 in the segment KPI table, and what was the total "
                         "operating income in 2025 according to the annual report?")
    assert parts["sql"] == "What was Mining's operating income in 2025 in the segment KPI table?"
    assert parts["document"] == "what was the total operating income in 2025 according to the annual report?"
    compare = split_hybrid("Compare Europe's 2025 sales in the segment KPI table with the Europe sales figure in the annual report.")
    assert compare["sql"] == "What is Europe's 2025 sales in the segment KPI table?"
    assert compare["document"] == "What is the Europe sales figure in the annual report in 2025?"
    bare = split_hybrid("Compare Europe sales in the segment kpi table with the annual report.")
    assert bare["document"] == "What is Europe sales according to the annual report?"
    assert split_hybrid("What were total sales in the orders table?")["split"] is False


def test_assistant_hybrid_sends_parts_to_each_side(db):
    asked = {}
    qa = StructuredQA(db, complete=lambda p: asked.setdefault("sql_prompt", p) and "SELECT sales_usd_m FROM segment_kpis "
                                                                                  "WHERE segment = 'Mining' AND year = 2025")
    rag = lambda q: asked.setdefault("doc", q) and {"answer": "Total operating income was $3,628 million.",
                                                    "validation": {"overall_status": "SUPPORTED"}, "citations": [{"page": 95}]}
    out = Assistant(qa, rag, VOCAB).ask("What was Mining's sales in 2025 in the segment kpi table, and what was the total "
                                        "operating income in 2025 according to the annual report?")
    assert out["sub_questions"]["split"] and "annual report" in asked["doc"] and "segment kpi table" not in asked["doc"]
    assert "QUESTION: What was Mining's sales in 2025 in the segment kpi table?" in asked["sql_prompt"]
