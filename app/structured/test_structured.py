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
    prompts = []
    qa = StructuredQA(db, complete=lambda p: prompts.append(p) or
                      "SELECT year, sales_usd_m FROM segment_kpis WHERE segment = 'Europe' AND year = 2025")
    out = qa.ask("How did Europe sales change between 2024 and 2025 in the segment kpi table?")
    assert out["incomplete"] == {"missing_periods": ["2024"]} and out["abstained"]
    assert "cannot calculate" in out["answer"] and "28,793" not in out["answer"]  # no single-period figure as an answer
    assert out["analytics"] == {} and len(prompts) == 2  # first attempt + one retry
    assert "returned no rows for 2024" in prompts[1]
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


# ---------------------------------------------------------------- Phase 4.1


def test_quarter_ranges_leap_years_and_q4():
    from app.structured.periods import quarter_range, quarters

    assert quarter_range(2025, 1) == ("2025-01-01", "2025-03-31")
    assert quarter_range(2024, 1) == ("2024-01-01", "2024-03-31")  # leap year: Q1 still ends 31 March
    assert quarter_range(2024, 4) == ("2024-10-01", "2024-12-31")  # never crosses the year boundary
    assert quarter_range(2023, 2) == ("2023-04-01", "2023-06-30")
    found = quarters("Revenue in the first quarter of 2025 vs Q4 2024 and 2023 Q3 and the 2nd quarter 2026")
    assert [q["label"] for q in found] == ["Q1 2025", "Q2 2026", "Q4 2024", "Q3 2023"]
    assert quarters("revenue in 2025") == []
    with pytest.raises(ValueError):
        quarter_range(2025, 5)


def test_quarter_filter_must_use_explicit_range(db):
    from app.structured.periods import quarter_filter_issue

    q = "Total amount in the first quarter of 2025 in the orders table?"
    assert quarter_filter_issue(q, "SELECT SUM(amount) FROM orders WHERE strftime('%Y-%m', order_date) = '2025-03'")
    assert quarter_filter_issue(q, "SELECT SUM(amount) FROM orders WHERE order_date BETWEEN '2025-01-01' AND '2025-03-31'") is None
    answers = iter(["SELECT SUM(amount) FROM orders WHERE strftime('%Y-%m', order_date) = '2025-03'",
                    "SELECT SUM(amount) FROM orders WHERE order_date BETWEEN '2025-01-01' AND '2025-03-31'"])
    prompts = []
    gen = nl2sql.generate_sql(q, discover(db), db, lambda p: prompts.append(p) or next(answers))
    assert "BETWEEN '2025-01-01' AND '2025-03-31'" in gen.sql and "Q1 2025 means dates from 2025-01-01" in prompts[0]


def test_missing_year_detection_ignores_non_year_numbers():
    from app.structured.periods import missing_years, requested_years

    q = "How did sales change between 2024 and 2025?"
    assert requested_years(q) == ["2024", "2025"] and requested_years("Sales in 2024 and 2025") == []
    assert missing_years(q, ["segment", "sales"], [["A", 20245], ["B", 2024.0]]) == ["2024", "2025"]  # 20245 is not a year
    assert missing_years(q, ["period", "v"], [["2024-03", 1], [2025, 2]]) == []
    assert missing_years(q, ["sales_2024", "sales_2025"], [[1, 2]]) == []


def test_missing_year_retry_succeeds_and_computes_from_both_periods(db):
    answers = iter(["SELECT year, sales_usd_m FROM segment_kpis WHERE segment = 'Europe' AND year = 2025",
                    "SELECT year, sales_usd_m FROM segment_kpis WHERE segment = 'Europe' ORDER BY year"])
    out = StructuredQA(db, complete=lambda p: next(answers)).ask(
        "How did Europe sales change between 2024 and 2025 in the segment kpi table?")
    assert "incomplete" not in out and "-3.87%" in out["answer"]
    assert out["generation"]["retry_for_missing_periods"]["missing"] == ["2024"]


def test_name_resolution_preserves_ranking_and_top_n(db):
    sql = "SELECT customer_id, SUM(amount) AS total FROM orders GROUP BY customer_id ORDER BY total DESC LIMIT 2"
    out = StructuredQA(db, complete=lambda p: sql).ask("Which customers have the highest total amount in the orders table?")
    assert out["columns"] == ["customer_id", "name", "total"]
    assert out["rows"] == [["C1", "Alpha Works", 350.5], ["C2", "Beta Mills", 80.0]]
    assert out["resolved_names"] == [{"column": "customer_id", "table": "customers", "key": "customer_id", "name_column": "name"}]
    assert {s["table"] for s in out["sources"]} == {"orders", "customers"}


def test_name_resolution_only_when_entity_asked_and_names_missing(db):
    sql = "SELECT customer_id, SUM(amount) AS total FROM orders GROUP BY customer_id"
    out = StructuredQA(db, complete=lambda p: sql).ask("Total amount per id in the orders table?")
    assert out["resolved_names"] == [] and out["columns"] == ["customer_id", "total"]
    named = "SELECT c.name, SUM(o.amount) AS total FROM orders o JOIN customers c ON o.customer_id = c.customer_id GROUP BY c.name"
    assert StructuredQA(db, complete=lambda p: named).ask("Total amount per customer in the orders table?")["resolved_names"] == []


def test_entity_name_hint_is_schema_driven(db):
    schema = discover(db)
    hint = nl2sql.entity_name_hints("Which customers ordered the most in the orders table?", schema)
    assert hint == ["The question asks for customers: return customers.name (join through a VERIFIED JOIN on customer_id "
                    "if needed), not only the ID."]
    # No hint for entities without a verified name column, or for non-"which" questions.
    assert nl2sql.entity_name_hints("Which sales representative handled most orders?", schema) == []
    assert nl2sql.entity_name_hints("How many customers are there?", schema) == []
    assert "name column" not in nl2sql.PROMPT  # the global prompt is unchanged from Phase 4


def test_citation_fallback_normalises_leading_article_only():
    from app.structured.engine import ground_citations

    table = {"source": "r.pdf", "page": 95, "text": "(in $ millions) Sales for the year ended December 31, Segment 2025 2024 "
                                                    "Europe 28,793 29,952 Total 61,352 62,441"}
    claim = "The Europe sales figure in the annual report for 2025 is $28,793 million."
    rag = {"answer": claim, "citations": [], "context": [table],
           "validation": {"overall_status": "UNSUPPORTED", "claims": [{"claim_text": claim, "status": "UNSUPPORTED"}]}}
    grounded = ground_citations("What is the Europe sales figure in the annual report in 2025?", rag)
    assert [c["page"] for c in grounded["citations"]] == [95] and grounded["validation"]["overall_status"] == "SUPPORTED"
    # A figure that is not in any context chunk is never given a citation.
    absent = "The Europe sales figure for 2025 is $31,000 million."
    wrong = {**rag, "answer": absent,
             "validation": {"overall_status": "UNSUPPORTED", "claims": [{"claim_text": absent, "status": "UNSUPPORTED"}]}}
    assert ground_citations("What is the Europe sales figure in 2025?", wrong) is wrong


def test_hybrid_citation_fallback_adds_supporting_context_and_revalidates():
    from app.structured.engine import ground_citations

    table = {"source": "r.pdf", "page": 95, "text": "(in $ millions) Sales for the year ended December 31, Segment 2025 2024 "
                                                    "Europe 28,793 29,952 Total 61,352 62,441"}
    other = {"source": "r.pdf", "page": 172, "text": "Consolidated statements of operations. Cost of sales 56,976."}
    rag = {"answer": "Europe's sales in 2025 were $28,793 million.", "citations": [other], "context": [table, other],
           "validation": {"overall_status": "UNSUPPORTED", "claims": [
               {"claim_text": "Europe's sales in 2025 were $28,793 million.", "status": "UNSUPPORTED"}]}}
    grounded = ground_citations("What were Europe's sales in 2025 in the annual report?", rag)
    assert [c["page"] for c in grounded["citations"]] == [172, 95]
    assert grounded["validation"]["overall_status"] == "SUPPORTED"
    assert grounded["citation_fallback"] == {"added_pages": [95], "status_before": "UNSUPPORTED", "status_after": "SUPPORTED",
                                             "leading_articles_normalised": True}


def test_hybrid_citation_fallback_never_adds_unsupporting_chunks():
    from app.structured.engine import ground_citations

    other = {"source": "r.pdf", "page": 172, "text": "Cost of sales 56,976."}
    rag = {"answer": "Europe's sales in 2025 were $99,999 million.", "citations": [], "context": [other],
           "validation": {"overall_status": "UNSUPPORTED", "claims": [{"claim_text": "Europe's sales in 2025 were $99,999 million.",
                                                                       "status": "UNSUPPORTED"}]}}
    assert ground_citations("q", rag) is rag  # unchanged: no chunk supports the figure
    supported = {**rag, "validation": {"overall_status": "SUPPORTED", "claims": []}}
    assert ground_citations("q", supported) is supported


def test_name_hint_failure_falls_back_to_plain_prompt_and_resolves_names(db):
    prompts = []

    def fake(prompt):
        prompts.append(prompt)
        if "The question asks for customers" in prompt:
            return "SELECT name, SUM(amount) FROM orders GROUP BY name"  # invalid: name is in customers
        return "SELECT customer_id, SUM(amount) AS total FROM orders GROUP BY customer_id ORDER BY total DESC LIMIT 1"

    out = StructuredQA(db, complete=fake).ask("Which customer has the highest total amount in the orders table?")
    assert out["rows"] == [["C1", "Alpha Works", 350.5]] and out["error"] is None
    assert "without_name_hint" in out["generation"] and len(prompts) == 4  # 3 hinted attempts + 1 plain


def test_years_are_not_formatted_with_thousands_separators():
    result = sql_safety.QueryResult("x", columns=["year", "sales_usd_m"], rows=[(2024, 29952), (2025, 28793)])
    text = render_answer(result, {})
    assert "year 2024; sales usd m 29,952" in text and "2,024" not in text


# ---------------------------------------------------------------- Phase 4.2: multi-period result scope

from app.structured import scope  # noqa: E402

CHANGE_Q = "How did Europe sales change between 2024 and 2025 in the segment kpi table?"
SCOPED_SQL = "SELECT year, sales_usd_m FROM segment_kpis WHERE segment = 'Europe' ORDER BY year"
UNFILTERED_SQL = "SELECT segment, year, sales_usd_m FROM segment_kpis GROUP BY segment, year"


def test_requested_filters_come_from_the_data_of_the_queried_tables(db):
    schema = discover(db)
    found = scope.requested_filters(CHANGE_Q, UNFILTERED_SQL, schema, db)
    assert found == [{"value": "Europe", "columns": [("segment_kpis", "segment")]}]
    # customers.region also contains 'Europe', but only tables the query reads are considered
    assert scope.requested_filters("How did sales change between 2024 and 2025?", UNFILTERED_SQL, schema, db) == []
    assert scope.requested_filters("Europe and Mining sales between 2024 and 2025", UNFILTERED_SQL, schema, db)[1]["value"] == "Mining"


def test_scope_issues_for_missing_filter_extra_categories_and_case_when():
    europe = [{"value": "Europe", "columns": [("segment_kpis", "segment")]}]
    assert scope.scope_issues(SCOPED_SQL, ["year", "sales"], [[2024, 1], [2025, 2]], europe, ["2024", "2025"]) == []
    # filter column returned: every row must carry the requested value
    issues = scope.scope_issues(UNFILTERED_SQL, ["segment", "year", "sales"],
                                [["Europe", 2024, 1], ["Mining", 2024, 2]], europe, ["2024", "2025"])
    assert issues and "Mining" in issues[0]
    # filter column not returned and no WHERE predicate
    assert "does not filter" in scope.scope_issues("SELECT year, SUM(sales_usd_m) FROM segment_kpis GROUP BY year",
                                                   ["year", "s"], [[2024, 1], [2025, 2]], europe)[0]
    # a CASE WHEN in the select list is not a filter
    case_sql = "SELECT year, SUM(CASE WHEN segment = 'Europe' THEN sales_usd_m END) AS s FROM segment_kpis GROUP BY year"
    assert scope.scope_issues(case_sql, ["year", "s"], [[2024, 1], [2025, 2]], europe)
    # IN (...) and HAVING are filters; periods outside the requested range are reported
    assert scope.scope_issues("SELECT year, sales_usd_m FROM segment_kpis WHERE segment IN ('Europe') AND year > 2000",
                              ["year", "sales_usd_m"], [[2023, 0], [2024, 1]], europe, ["2024", "2025"]) == [
        "the result contains periods outside 2024-2025 (2023)"]


def test_model_derived_columns_are_detected_but_real_columns_are_kept(db):
    schema = discover(db)
    cols = ["segment", "year", "sales_usd_m", "sales_change", "growth_pct", "yoy_delta", "employees_fte"]
    assert [cols[i] for i in scope.derived_columns(cols, schema)] == ["sales_change", "growth_pct", "yoy_delta"]
    assert scope.relevant_metrics("How did sales change?", cols, [2, 6], schema) == [2]
    assert scope.relevant_metrics("How did it change?", cols, [2, 6], schema) == [2, 6]  # nothing named: keep all


def test_unfiltered_multi_period_query_is_retried_with_the_filter(db):
    prompts = []
    answers = iter([UNFILTERED_SQL, SCOPED_SQL])
    out = StructuredQA(db, complete=lambda p: prompts.append(p) or next(answers)).ask(CHANGE_Q)
    assert len(prompts) == 2 and "segment = 'Europe'" in prompts[1] and "WHERE clause" in prompts[1]
    assert out["generation"]["retry_for_scope"]["scope_issues"] and out["scope"]["issues_final_query"] == []
    assert not out.get("abstained") and out["rows"] == [[2024, 29952], [2025, 28793]]
    assert "-1,159 (-3.87%)" in out["answer"] and "Mining" not in out["answer"]
    assert "2024: sales usd m 29,952" in out["answer"] and "2025: sales usd m 28,793" in out["answer"]


def test_refusal_when_the_filter_is_still_missing_after_one_retry(db):
    prompts = []
    out = StructuredQA(db, complete=lambda p: prompts.append(p) or UNFILTERED_SQL).ask(CHANGE_Q)
    assert len(prompts) == 2 and out["abstained"] and "cannot calculate" in out["answer"]
    assert out["rows"] == [] and out["analytics"] == {} and out["rejected_result"]["rows"]
    assert not any(v in out["answer"] for v in ("29,952", "28,793", "-1,159", "3.87"))
    # the hybrid assistant does not use an abstained structured result as evidence
    assert not Assistant._sql_ok(out)


def test_category_rows_in_a_grouped_result_are_still_out_of_scope(db):
    # Europe's figures are present (and its change would be right), but the other segments make the
    # result wrong for the question: it is rejected, not presented.
    sql = "SELECT segment, year, sales_usd_m FROM segment_kpis WHERE year IN (2024, 2025) ORDER BY segment, year"
    out = StructuredQA(db, complete=lambda p: sql).ask(CHANGE_Q)
    assert out["abstained"] and "Mining" in out["answer"] and "-3.87" not in out["answer"]


def test_misleading_model_change_columns_are_ignored_and_change_computed_in_python(db):
    sql = ("SELECT year, sales_usd_m, employees_fte, SUM(CASE WHEN year = 2024 THEN sales_usd_m ELSE 0 END) AS sales_change "
           "FROM segment_kpis WHERE segment = 'Europe' GROUP BY year ORDER BY year")
    out = StructuredQA(db, complete=lambda p: sql).ask(CHANGE_Q)
    assert out["columns"] == ["year", "sales_usd_m"] and out["rows"] == [[2024, 29952], [2025, 28793]]
    assert out["presentation"] == {"model_derived_columns_ignored": ["sales_change"], "metrics_not_asked": ["employees_fte"]}
    assert "sales_change" in out["raw_result"]["columns"]
    changes = out["analytics"]["period_changes"]
    assert [(c["metric"], c["absolute_change"], c["percent_change"]) for c in changes] == [("sales_usd_m", -1159, -3.87)]
    assert "sales change" not in out["answer"] and "-100" not in out["answer"]


def test_multiple_categories_and_periods_when_both_are_requested(db):
    q = "How did Europe and Mining sales change between 2024 and 2025 in the segment kpi table?"
    sql = "SELECT segment, year, sales_usd_m FROM segment_kpis WHERE segment IN ('Europe', 'Mining') ORDER BY segment, year"
    out = StructuredQA(db, complete=lambda p: sql).ask(q)
    assert not out.get("abstained") and len(out["rows"]) == 4
    groups = {c["group"]: c["absolute_change"] for c in out["analytics"]["period_changes"]}
    assert groups == {"Europe": -1159, "Mining": 569}
    assert "Europe, sales usd m: 29,952 (2024) to 28,793 (2025)" in out["answer"]


def test_scope_checks_do_not_apply_to_single_period_questions(db):
    sql = "SELECT segment, sales_usd_m FROM segment_kpis WHERE year = 2025"
    prompts = []
    out = StructuredQA(db, complete=lambda p: prompts.append(p) or sql).ask("Europe sales in 2025 in the segment kpi table?")
    assert len(prompts) == 1 and "scope" not in out and len(out["rows"]) == 2


def test_filter_must_hold_in_every_top_level_or_branch():
    europe = [{"value": "Europe", "columns": [("segment_kpis", "segment")]}]
    rows, cols = [[2024, 1], [2025, 2]], ["year", "s"]
    precedence_bug = "SELECT year, SUM(sales_usd_m) AS s FROM segment_kpis WHERE segment = 'Europe' AND year = 2024 OR year = 2025 GROUP BY year"
    assert scope.scope_issues(precedence_bug, cols, rows, europe)
    grouped = "SELECT year, SUM(sales_usd_m) AS s FROM segment_kpis WHERE segment = 'Europe' AND (year = 2024 OR year = 2025) GROUP BY year"
    assert scope.scope_issues(grouped, cols, rows, europe) == []
    both = ("SELECT year, SUM(sales_usd_m) AS s FROM segment_kpis WHERE (segment = 'Europe' AND year = 2024) "
            "OR (segment = 'Europe' AND year = 2025) GROUP BY year")
    assert scope.scope_issues(both, cols, rows, europe) == []


def test_year_strftime_rewrite_only_after_all_attempts_fail_and_is_revalidated(db):
    schema = discover(db)
    bad = ("SELECT strftime('%Y', year) AS period, SUM(sales_usd_m) AS sales FROM segment_kpis "
           "WHERE segment = 'Europe' AND year BETWEEN 2024 AND 2025 GROUP BY strftime('%Y', year)")
    assert nl2sql.rewrite_year_strftime(bad, schema)[1] == ["year", "year"]
    assert nl2sql.rewrite_year_strftime("SELECT strftime('%m', year) FROM segment_kpis", schema) is None  # only '%Y'
    assert nl2sql.rewrite_year_strftime("SELECT strftime('%Y', sales_usd_m) FROM segment_kpis", schema) is None  # not years
    gen = nl2sql.generate_sql(CHANGE_Q, schema, db, lambda p: bad)
    assert [a["valid"] for a in gen.attempts] == [False, False, False, True] and gen.attempts[-1]["from_attempt"] == 1
    assert "strftime" not in gen.sql and sql_safety.execute(gen.sql, db).rows == [(2024, 29952), (2025, 28793)]
    # a valid attempt is used as generated (no rewrite)
    ok = nl2sql.generate_sql(CHANGE_Q, schema, db, lambda p: SCOPED_SQL)
    assert len(ok.attempts) == 1 and ok.sql == SCOPED_SQL


def test_retry_with_or_precedence_bug_is_refused(db):
    first = "SELECT segment, year, sales_usd_m FROM segment_kpis GROUP BY segment, year"
    retry = ("SELECT year, SUM(sales_usd_m) AS sales FROM segment_kpis WHERE segment = 'Europe' AND year = 2024 "
             "OR year = 2025 GROUP BY year")
    answers = iter([first, retry])
    out = StructuredQA(db, complete=lambda p: next(answers)).ask(CHANGE_Q)
    assert out["abstained"] and out["rows"] == [] and "does not filter" in out["answer"]


# ---------------------------------------------------------------- Phase 4.3: deterministic shares and ratios

from app.structured import shares  # noqa: E402

SHARE_Q = "What share of 2025 employees in the segment kpi table does each segment have?"
RAW_2025 = "SELECT segment, employees_fte FROM segment_kpis WHERE year = 2025"
FRACTION_ONLY = ("SELECT segment, 1.0 * SUM(employees_fte) / SUM(employees_fte) OVER () AS share FROM segment_kpis "
                 "WHERE year = 2025 GROUP BY segment")


@pytest.fixture
def sales_db(db, tmp_path):
    path = tmp_path / "regional_sales.csv"
    path.write_text("Region,Product,Year,Revenue\nEurope,Coil,2025,60\nEurope,Plate,2025,40\nEurope,Coil,2024,10\n"
                    "Asia,Coil,2025,500\nAsia,Plate,2025,0\n", encoding="utf-8")
    ingest_file(path, db)
    return db


def test_share_and_ratio_intent_detection():
    assert shares.intent(SHARE_Q) == "share"
    assert shares.intent("What percentage of total revenue does each region contribute?") == "share"
    assert shares.intent("How much does each segment contribute to employees?") == "share"
    assert shares.intent("What is the ratio of operating income to sales for each segment?") == "ratio"
    assert shares.ratio_terms("What is operating income as a percentage of sales in 2025?") == ("operating income", "sales")
    assert shares.intent("Sales as a share of total sales by segment") == "share"  # 'of total' is a share, not a ratio
    for q in ("What is the average revenue per order for each customer region?", "What is the total revenue per product?"):
        assert shares.intent(q) is None


def test_model_computed_fraction_columns_are_detected(db):
    schema = discover(db)
    assert shares.computed_fraction_columns(FRACTION_ONLY, ["segment", "share"], schema) == [1]
    sql = ("SELECT segment, SUM(employees_fte) AS employees_fte, employees_fte * 100.0 / 50 AS x FROM segment_kpis "
           "GROUP BY segment")
    assert shares.computed_fraction_columns(sql, ["segment", "employees_fte", "x"], schema) == [2]
    assert shares.computed_fraction_columns(RAW_2025, ["segment", "employees_fte"], schema) == []


def test_percentage_of_total_from_raw_values_with_verified_denominator(db):
    out = StructuredQA(db, complete=lambda p: RAW_2025).ask(
        "What percentage of total 2025 employees does each segment have in the segment kpi table?")
    got = {s["label"]: (s["numerator"], s["denominator"], s["share_percent"]) for s in out["analytics"]["shares"]}
    assert got == {"Europe": (43878, 49796, 88.12), "Mining": (5918, 49796, 11.88)}
    assert out["share_validation"]["denominator_verified"] is True
    assert "43,878 / 49,796 = 88.12%" in out["answer"] and out["chart"]["type"] == "pie"
    assert out["rows"] == [["Europe", 43878], ["Mining", 5918]]


def test_fraction_only_result_is_retried_for_raw_values(db):
    prompts, answers = [], iter([FRACTION_ONLY, RAW_2025])
    out = StructuredQA(db, complete=lambda p: prompts.append(p) or next(answers)).ask(SHARE_Q)
    assert len(prompts) == 2 and "raw value of employees_fte" in prompts[1] and "window functions" in prompts[1]
    assert out["share_validation"]["ignored_columns_first_query"] == ["share"]
    assert [s["share_percent"] for s in out["analytics"]["shares"]] == [88.12, 11.88]


def test_refusal_when_raw_values_are_still_missing(db):
    out = StructuredQA(db, complete=lambda p: FRACTION_ONLY).ask(SHARE_Q)
    assert out["abstained"] and out["rows"] == [] and "cannot calculate these shares" in out["answer"]
    assert "%" not in out["answer"] and out["rejected_result"]["columns"] == ["segment", "share"]


def test_model_share_column_next_to_raw_values_is_ignored(db):
    sql = ("SELECT segment, SUM(employees_fte) AS employees_fte, SUM(employees_fte) * 100.0 / 1000 AS share_pct "
           "FROM segment_kpis WHERE year = 2025 GROUP BY segment")
    out = StructuredQA(db, complete=lambda p: sql).ask(SHARE_Q)
    assert out["columns"] == ["segment", "employees_fte"]
    assert out["presentation"]["model_computed_columns_ignored"] == ["share_pct"]
    assert [s["share_percent"] for s in out["analytics"]["shares"]] == [88.12, 11.88]
    assert "share_pct" in out["raw_result"]["columns"] and "Model-computed columns not used: share_pct" in out["answer"]


def test_employee_contribution_question(db):
    out = StructuredQA(db, complete=lambda p: RAW_2025).ask(
        "How much does each segment contribute to 2025 employees (FTE) in the segment kpi table?")
    assert {s["label"]: s["share_percent"] for s in out["analytics"]["shares"]} == {"Europe": 88.12, "Mining": 11.88}


def test_grouped_shares_within_each_period(db):
    sql = "SELECT year, segment, employees_fte FROM segment_kpis ORDER BY year, segment"
    out = StructuredQA(db, complete=lambda p: sql).ask(
        "What share of employees does each segment have per year in the segment kpi table?")
    by_group = {}
    for s in out["analytics"]["shares"]:
        by_group.setdefault(s["group"], {})[s["label"]] = s["share_percent"]
    assert by_group == {2024: {"Europe": 91.07, "Mining": 8.93}, 2025: {"Europe": 88.12, "Mining": 11.88}}
    assert {d["group"]: d["denominator"] for d in out["analytics"]["share_denominators"]} == {2024: 53302, 2025: 49796}
    assert out["chart"]["type"] != "pie"  # several wholes: not one pie


def test_zero_missing_and_negative_values_are_refused(db):
    schema = discover(db)

    def check(rows):
        return shares.check_shares("share of value by c", "SELECT c, value FROM x", ["c", "value"], rows, schema, db, [], [])

    assert "zero" in check([["a", 0], ["b", 0]])["issues"][0]
    assert "missing for b" in check([["a", 5], ["b", None]])["issues"][0]
    assert "negative for b" in check([["a", 5], ["b", -2]])["issues"][0]
    ok = check([["a", 1], ["b", 3]])
    assert ok["issues"] == [] and [s["share_percent"] for s in ok["shares"]] == [25.0, 75.0]
    assert not ok["denominator_verified"]


def test_missing_year_filter_gives_a_wrong_denominator_and_is_refused(db):
    both_years = "SELECT segment, SUM(employees_fte) AS employees_fte FROM segment_kpis GROUP BY segment"
    prompts = []
    out = StructuredQA(db, complete=lambda p: prompts.append(p) or both_years).ask(SHARE_Q)
    assert len(prompts) == 2 and out["abstained"] and out["analytics"] == {}
    assert "103,098" in out["answer"] and "49,796" in out["answer"] and "cannot calculate these shares" in out["answer"]


def test_category_filter_scope_and_selected_category(sales_db):
    q = "What share of Europe's 2025 revenue does each product have in the regional sales table?"
    unfiltered = "SELECT product, SUM(revenue) AS revenue FROM regional_sales WHERE year = 2025 GROUP BY product"
    filtered = ("SELECT product, SUM(revenue) AS revenue FROM regional_sales WHERE region = 'Europe' AND year = 2025 "
                "GROUP BY product")
    answers = iter([unfiltered, filtered])
    out = StructuredQA(sales_db, complete=lambda p: next(answers)).ask(q)
    assert out["generation"]["retry_for_raw_values"]["issues"] and out["share_validation"]["denominator_verified"]
    assert {s["label"]: s["share_percent"] for s in out["analytics"]["shares"]} == {"Coil": 60.0, "Plate": 40.0}
    # a named category of the share column selects the numerator; the denominator needs every category
    q2 = "What share of 2025 revenue did Europe have in the regional sales table?"
    only = ("SELECT region, SUM(revenue) AS revenue FROM regional_sales WHERE region = 'Europe' AND year = 2025 "
            "GROUP BY region")
    every = "SELECT region, SUM(revenue) AS revenue FROM regional_sales WHERE year = 2025 GROUP BY region"
    answers = iter([only, every])
    out = StructuredQA(sales_db, complete=lambda p: next(answers)).ask(q2)
    assert {s["label"]: s["share_percent"] for s in out["analytics"]["shares"]} == {"Asia": 83.33, "Europe": 16.67}
    assert "Europe (asked): 100 / 600 = 16.67%" in out["answer"]
    assert StructuredQA(sales_db, complete=lambda p: only).ask(q2)["abstained"]


def test_limited_and_non_additive_results_are_refused(db):
    limited = "SELECT segment, employees_fte FROM segment_kpis WHERE year = 2025 ORDER BY employees_fte DESC LIMIT 1"
    assert "LIMIT" in StructuredQA(db, complete=lambda p: limited).ask(SHARE_Q)["answer"]
    avg = "SELECT segment, AVG(employees_fte) AS employees_fte FROM segment_kpis WHERE year = 2025 GROUP BY segment"
    assert "average" in StructuredQA(db, complete=lambda p: avg).ask(SHARE_Q)["answer"]


def test_ratio_from_raw_columns_and_model_ratio_ignored(db):
    q = "What is the ratio of sales to employees for each segment in 2025 in the segment kpi table?"
    raw = ("SELECT segment, sales_usd_m, employees_fte, 1.0 * sales_usd_m / employees_fte AS ratio FROM segment_kpis "
           "WHERE year = 2025")
    out = StructuredQA(db, complete=lambda p: raw).ask(q)
    got = {r["label"]: r["ratio"] for r in out["analytics"]["ratios"]}
    assert got == {"Europe": round(28793 / 43878, 6), "Mining": round(3232 / 5918, 6)}
    assert out["presentation"]["model_computed_columns_ignored"] == ["ratio"] and "28,793 / 43,878" in out["answer"]
    only_ratio = "SELECT segment, 1.0 * sales_usd_m / employees_fte AS ratio FROM segment_kpis WHERE year = 2025"
    refused = StructuredQA(db, complete=lambda p: only_ratio).ask(q)
    assert refused["abstained"] and "cannot calculate this ratio" in refused["answer"]
    zero = shares.check_ratio(q, "SELECT a, sales, employees FROM x", ["segment", "sales", "employees"],
                              [["A", 5, 0]], discover(db), [], [])
    assert "undefined" in zero["issues"][0] and zero["ratios"] == []
