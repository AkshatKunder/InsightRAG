"""Phase 5 application tests. Backends are mocked: no Ollama, model download or running service is needed.

The Qdrant tests use a throw-away local Qdrant folder under ``tmp_path`` (qdrant-client's embedded mode)
and fake embeddings, so the project's own index is never touched.
"""

from __future__ import annotations

import io
import zipfile
from pathlib import Path

import numpy as np
import pytest

from app.structured.engine import ABSTENTION
from app.ui import presentation
from app.ui.config import AppConfig, redact_url
from app.ui.conversations import ConversationStore
from app.ui.datasets import DatasetError, DatasetManager
from app.ui.documents import DocumentError, DocumentLibrary, safe_filename, validate_pdf
from app.ui.service import InsightService, explain_error

ROOT = Path(__file__).resolve().parents[2]
SEGMENTS_CSV = "Segment,Year,Sales (USD m),Employees FTE\nEurope,2024,29952,48542\nEurope,2025,28793,43878\nMining,2024,3021,4760\nMining,2025,3232,5918\n"


# ---------------------------------------------------------------------- helpers
def make_pdf(pages: list[str]) -> bytes:
    """A minimal valid PDF with one Helvetica text line per page (correct xref offsets)."""

    objects = ["<< /Type /Catalog /Pages 2 0 R >>", None, "<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>"]
    kids = []
    for text in pages:
        stream = f"BT /F1 12 Tf 72 720 Td ({text}) Tj ET".encode("latin-1")
        objects.append(f"<< /Length {len(stream)} >>\nstream\n{stream.decode('latin-1')}\nendstream")
        content_id = len(objects)
        objects.append(f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Resources << /Font << /F1 3 0 R >> >> "
                       f"/Contents {content_id} 0 R >>")
        kids.append(f"{len(objects)} 0 R")
    objects[1] = f"<< /Type /Pages /Kids [{' '.join(kids)}] /Count {len(kids)} >>"
    out, offsets = io.BytesIO(), []
    out.write(b"%PDF-1.4\n")
    for i, body in enumerate(objects, start=1):
        offsets.append(out.tell())
        out.write(f"{i} 0 obj\n{body}\nendobj\n".encode("latin-1"))
    xref = out.tell()
    out.write(f"xref\n0 {len(objects) + 1}\n0000000000 65535 f \n".encode())
    for off in offsets:
        out.write(f"{off:010d} 00000 n \n".encode())
    out.write(f"trailer\n<< /Size {len(objects) + 1} /Root 1 0 R >>\nstartxref\n{xref}\n%%EOF\n".encode())
    return out.getvalue()


class FakeEmbeddings:
    def embed_documents(self, texts):
        return np.array([self._vec(t) for t in texts])

    def embed_query(self, text):
        return self._vec(text)

    @staticmethod
    def _vec(text):
        rng = np.random.default_rng(abs(hash(text)) % (2 ** 32))
        v = rng.random(384)
        return v / np.linalg.norm(v)


class FakeOllama:
    def __init__(self, models=("llama3.2:3b",), fail=False):
        self.models, self.fail = models, fail

    def list(self):
        if self.fail:
            raise ConnectionError("connection refused")
        return {"models": [{"model": m} for m in self.models]}


class FakeRAG:
    """Stands in for RAGPipeline: records questions and returns a validated, cited answer."""

    def __init__(self, chunks):
        self.questions = []
        self.retriever = type("R", (), {"chunks": chunks})()
        self.generator = type("G", (), {"options": None})()

    def answer(self, question):
        self.questions.append(question)
        chunk = {"source": "report.pdf", "page": 7, "text": "Total revenue was 120 million in 2025."}
        return {"answer": "Revenue was 120 million in 2025.", "citations": [chunk], "context": [chunk],
                "validation": {"overall_status": "SUPPORTED", "reason": "ok", "claims": [
                    {"status": "SUPPORTED", "claim_text": "Revenue was 120 million in 2025.", "source": "report.pdf",
                     "page": 7, "explanation": "value found"}]}}


def config_for(tmp_path) -> AppConfig:
    return AppConfig(documents_dir=tmp_path / "docs", qdrant_path=tmp_path / "qdrant", db_path=tmp_path / "data.db",
                     dataset_upload_dir=tmp_path / "uploads", max_pdf_mb=5, max_dataset_mb=5)


def make_service(tmp_path, sql="SELECT segment, employees_fte FROM segment_kpis WHERE year = 2025", ollama=None,
                 complete=None, with_docs=True, rag=None):
    from app.rag.vector_store import VectorStore

    cfg = config_for(tmp_path)
    built = {}

    def builder(paths, store, embedding_model, options, model):
        built["paths"] = list(paths)
        return rag or FakeRAG([{"text": "annual report revenue europe"}])

    service = InsightService(cfg, complete=complete or (lambda prompt, options, model: sql), rag_builder=builder,
                             store_factory=lambda path: VectorStore(path=path), embedding_factory=FakeEmbeddings,
                             ollama_client=ollama or FakeOllama())
    if with_docs:
        service.add_document("report.pdf", make_pdf(["Total revenue was 120 million in 2025.", "Second page text."]))
    service.built = built
    return service


@pytest.fixture
def service(tmp_path):
    svc = make_service(tmp_path)
    (tmp_path / "seg.csv").write_text(SEGMENTS_CSV, encoding="utf-8")
    svc.add_dataset("segment_kpis.csv", SEGMENTS_CSV.encode())
    yield svc
    svc.close()


# ---------------------------------------------------------------------- configuration
def test_config_from_environment_and_redaction(monkeypatch, tmp_path):
    monkeypatch.setenv("INSIGHTRAG_MODEL", "mymodel:1b")
    monkeypatch.setenv("INSIGHTRAG_DB_PATH", str(tmp_path / "x.db"))
    monkeypatch.setenv("INSIGHTRAG_DETERMINISTIC", "0")
    monkeypatch.setenv("INSIGHTRAG_MAX_DATASET_MB", "999")
    monkeypatch.setenv("OLLAMA_HOST", "http://user:secret@gpu-box:11434")
    cfg = AppConfig.from_env(load_dotenv=False)
    assert cfg.model == "mymodel:1b" and cfg.db_path == tmp_path / "x.db"
    assert cfg.generation_options == {} and cfg.max_dataset_mb == 50  # capped at the ingestion limit
    shown = str(cfg.public_dict())
    assert "secret" not in shown and "gpu-box:11434" in shown
    assert redact_url("http://a:b@h:1/x") == "http://h:1/x"
    monkeypatch.delenv("INSIGHTRAG_DETERMINISTIC")
    assert AppConfig.from_env(load_dotenv=False).generation_options == {"temperature": 0, "seed": 42}


# ---------------------------------------------------------------------- upload validation
def test_pdf_upload_validation():
    good = make_pdf(["Hello annual report"])
    assert validate_pdf("a.pdf", good, 5)[0]["text"] == "Hello annual report"
    with pytest.raises(DocumentError, match="Only .pdf"):
        validate_pdf("a.docx", good, 5)
    with pytest.raises(DocumentError, match="missing %PDF header"):
        validate_pdf("a.pdf", b"not a pdf at all", 5)
    with pytest.raises(DocumentError, match="empty"):
        validate_pdf("a.pdf", b"", 5)
    with pytest.raises(DocumentError, match="the limit is 1 MB"):
        validate_pdf("a.pdf", good + b" " * (1024 * 1024), 1)
    with pytest.raises(DocumentError, match="no extractable text"):
        validate_pdf("a.pdf", make_pdf([""]), 5)
    with pytest.raises(DocumentError, match="could not be read"):
        validate_pdf("a.pdf", b"%PDF-1.4\n garbage", 5)
    assert safe_filename("../../etc/evil name?.PDF") == "evil name.pdf"


def test_dataset_upload_validation(tmp_path):
    mgr = DatasetManager(tmp_path / "d.db", tmp_path / "up", max_mb=1)
    with pytest.raises(DatasetError, match="csv or .xlsx"):
        mgr.upload("data.json", b"{}")
    with pytest.raises(DatasetError, match="empty"):
        mgr.upload("data.csv", b"")
    with pytest.raises(DatasetError, match="the limit is 1 MB"):
        mgr.upload("data.csv", b"a,b\n" * 400_000)
    with pytest.raises(DatasetError, match="not a valid Excel"):
        mgr.upload("book.xlsx", b"plain text")
    with pytest.raises(DatasetError):
        mgr.upload("header_only.csv", b"a,b,c\n")  # ingestion error is passed on as a readable message
    assert not list((tmp_path / "up").glob("*")), "rejected files are not kept"


# ---------------------------------------------------------------------- document ingestion
def test_document_ingestion_index_status_and_no_overwrite(tmp_path):
    from app.rag.vector_store import VectorStore

    lib = DocumentLibrary(tmp_path / "docs", tmp_path / "q", max_mb=5)
    store = VectorStore(path=str(tmp_path / "q"))
    try:
        assert not lib.index_status(store).ok
        data = make_pdf(["Alpha page one", "Alpha page two"])
        name, pages = lib.check_new("alpha.pdf", data)
        lib.save(name, data)
        assert not lib.index_status(store).ok  # saved but not indexed: reported, not hidden
        assert lib.append_to_index(pages, FakeEmbeddings(), store) == 2
        status = lib.index_status(store)
        assert status.ok and status.documents[0].indexed and status.total_vectors == 2
        payloads = [p.payload for p in store.client.scroll("insightrag_documents", limit=10)[0]]
        assert {(p["source"], p["page"]) for p in payloads} == {("alpha.pdf", 1), ("alpha.pdf", 2)}  # page refs kept
        with pytest.raises(DocumentError, match="already exists"):
            lib.check_new("alpha.pdf", make_pdf(["Different content"]))
        with pytest.raises(DocumentError, match="same content"):
            lib.check_new("copy.pdf", data)
        with pytest.raises(DocumentError, match="already exists"):
            lib.save("alpha.pdf", b"%PDF-overwrite")
        assert (tmp_path / "docs" / "alpha.pdf").read_bytes() == data
        beta = make_pdf(["Beta only page"])
        name, pages = lib.check_new("beta.pdf", beta)
        lib.save(name, beta)
        lib.append_to_index(pages, FakeEmbeddings(), store)
        ids = sorted(p.id for p in store.client.scroll("insightrag_documents", limit=10)[0])
        assert ids == [0, 1, 2], "appending never overwrites existing points"
        assert lib.index_status(store).ok
        assert lib.rebuild_index(FakeEmbeddings(), store) == 3 and lib.index_status(store).ok
    finally:
        store.close()


def test_service_add_document_failure_removes_file(tmp_path):
    svc = make_service(tmp_path, with_docs=False)
    try:
        svc._embedding_model = type("Broken", (), {"embed_documents": lambda self, t: (_ for _ in ()).throw(RuntimeError("x"))})()
        with pytest.raises(DocumentError, match="could not be indexed"):
            svc.add_document("r.pdf", make_pdf(["Some text"]))
        assert not (tmp_path / "docs" / "r.pdf").exists()
    finally:
        svc.close()


def test_service_document_upload_builds_pipeline_over_library(service, tmp_path):
    assert service.health(refresh=True).documents.ok
    assert [p.name for p in service.built["paths"]] == ["report.pdf"]
    info = service.add_document("second.pdf", make_pdf(["Another document about steel"]))
    assert info["chunks"] == 1 and [p.name for p in service.built["paths"]] == ["report.pdf", "second.pdf"]
    with pytest.raises(DocumentError, match="already exists"):
        service.add_document("second.pdf", make_pdf(["changed"]))


# ---------------------------------------------------------------------- dataset ingestion
def test_dataset_ingestion_listing_preview_and_replace(tmp_path):
    mgr = DatasetManager(tmp_path / "d.db", tmp_path / "up", max_mb=5)
    result = mgr.upload("segment_kpis.csv", SEGMENTS_CSV.encode())
    assert result.tables[0]["name"] == "segment_kpis" and result.tables[0]["rows"] == 4 and not result.replaced
    ds = mgr.datasets()[0]
    cols = {c["name"]: c["type"] for c in ds["tables"][0]["columns"]}
    assert cols == {"segment": "TEXT", "year": "INTEGER", "sales_usd_m": "INTEGER", "employees_fte": "INTEGER"}
    assert ds["tables"][0]["columns"][2]["original_name"] == "Sales (USD m)"
    preview = mgr.preview("segment_kpis", 2)
    assert preview.ok and len(preview.rows) == 2 and preview.columns[0] == "segment"
    with pytest.raises(DatasetError, match="Unknown table"):
        mgr.preview('segment_kpis"; DROP TABLE segment_kpis; --')
    with pytest.raises(DatasetError, match="already loaded"):
        mgr.upload("segment_kpis.csv", SEGMENTS_CSV.encode())
    replaced = mgr.upload("segment_kpis.csv", (SEGMENTS_CSV + "Brazil,2025,1,2\n").encode(), replace=True)
    assert replaced.replaced and replaced.tables[0]["rows"] == 5
    stored = sorted(p.name for p in (tmp_path / "up").iterdir())
    assert stored == ["segment_kpis (1).csv", "segment_kpis.csv"], "stored uploads are never overwritten"


def test_excel_dataset_ingestion(tmp_path):
    openpyxl = pytest.importorskip("openpyxl")
    book = openpyxl.Workbook()
    sheet = book.active
    sheet.title = "orders"
    sheet.append(["Order ID", "Region", "Revenue"])
    sheet.append(["O1", "EU", 10.5])
    sheet.append(["O2", "US", 20])
    book.create_sheet("empty")
    buffer = io.BytesIO()
    book.save(buffer)
    mgr = DatasetManager(tmp_path / "d.db", tmp_path / "up", max_mb=5)
    result = mgr.upload("sales.xlsx", buffer.getvalue())
    assert [(t["name"], t["sheet"], t["rows"]) for t in result.tables] == [("sales_orders", "orders", 2)]
    corrupt = io.BytesIO()
    with zipfile.ZipFile(corrupt, "w") as z:
        z.writestr("x.txt", "not a workbook")
    with pytest.raises(DatasetError):
        mgr.upload("broken.xlsx", corrupt.getvalue())


# ---------------------------------------------------------------------- routing through the existing Assistant
def test_document_question_is_routed_to_rag(service):
    out = service.ask("What was the revenue in 2025 according to the annual report?")
    assert out["ok"] and out["result"]["route"] == "document"
    assert service._rag.questions == ["What was the revenue in 2025 according to the annual report?"]
    view = presentation.present(out["result"])
    assert view["source_label"] == "Document (RAG)" and view["document"]["pages"] == [("report.pdf", 7)]


def test_sql_question_is_routed_to_structured_engine(service):
    out = service.ask("What share of 2025 employees in the segment kpi table does each segment have?")
    result = out["result"]
    assert result["route"] == "sql" and service._rag.questions == []
    assert result["sql_result"]["share_validation"]["denominator_verified"] is True
    view = presentation.present(result)
    assert view["structured"]["derived"][0]["title"] == "Share of total"
    assert view["structured"]["raw"]["columns"] == ["segment", "employees fte (FTE)"]


def test_hybrid_question_uses_both_engines(service):
    q = "What were Europe's 2025 sales in the segment kpi table, and what was the revenue according to the annual report?"
    service._complete_fn = lambda prompt, options, model: "SELECT sales_usd_m FROM segment_kpis WHERE segment = 'Europe' AND year = 2025"
    out = service.ask(q)
    result = out["result"]
    assert result["route"] == "hybrid" and service._rag.questions
    view = presentation.present(result)
    assert view["structured"]["raw"]["rows"] == [[28793]] and view["document"]["status"] == "SUPPORTED"


def test_write_questions_stay_read_only(service):
    out = service.ask("Delete all rows from the segment kpi table")
    assert out["ok"] and "read-only" in out["result"]["answer"]
    service._complete_fn = lambda prompt, options, model: "DROP TABLE segment_kpis"
    out = service.ask("Show everything in the segment kpi table")
    assert out["ok"] and out["result"]["sql_result"]["rows"] == []
    assert service.datasets.preview("segment_kpis").ok  # the table is still there


# ---------------------------------------------------------------------- errors and degraded operation
def test_error_messages_are_actionable_and_do_not_leak(tmp_path):
    import ollama

    cfg = config_for(tmp_path)
    assert "Cannot reach Ollama" in explain_error(ConnectionError("refused"), cfg)
    assert "ollama pull llama3.2:3b" in explain_error(ollama.ResponseError("model 'x' not found", 404), cfg)
    assert "in use by another process" in explain_error(
        RuntimeError("Storage folder data/qdrant is already accessed by another instance of Qdrant client"), cfg)
    hidden = explain_error(ValueError("password=hunter2 at /secret/path"), cfg)
    assert "hunter2" not in hidden and "ValueError" in hidden


def test_ollama_unavailable_is_reported(tmp_path):
    svc = make_service(tmp_path, ollama=FakeOllama(fail=True), with_docs=False,
                       complete=lambda p, o, m: (_ for _ in ()).throw(ConnectionError("refused")))
    try:
        health = svc.health(refresh=True)
        assert not health.ollama.ok and not health.can_answer_data
        svc.add_dataset("segment_kpis.csv", SEGMENTS_CSV.encode())
        out = svc.ask("How many employees in 2025 in the segment kpi table?")
        assert not out["ok"] and "Cannot reach Ollama" in out["error"]
    finally:
        svc.close()


def test_missing_model_is_reported(tmp_path):
    svc = make_service(tmp_path, ollama=FakeOllama(models=("other:1b",)), with_docs=False)
    try:
        model = svc.health(refresh=True).model
        assert not model.ok and "ollama pull llama3.2:3b" in model.message
    finally:
        svc.close()


def test_qdrant_unavailable_keeps_sql_working(tmp_path):
    cfg = config_for(tmp_path)

    def locked(path):
        raise RuntimeError("Storage folder is already accessed by another instance of Qdrant client")

    svc = InsightService(cfg, complete=lambda p, o, m: "SELECT segment, sales_usd_m FROM segment_kpis WHERE year = 2025",
                         store_factory=locked, embedding_factory=FakeEmbeddings, ollama_client=FakeOllama())
    svc.add_dataset("segment_kpis.csv", SEGMENTS_CSV.encode())
    health = svc.health(refresh=True)
    assert not health.vector_index.ok and "another process" in health.vector_index.message
    with pytest.raises(DocumentError):
        svc.add_document("x.pdf", make_pdf(["text"]))
    assert not (tmp_path / "docs" / "x.pdf").exists()
    data = svc.ask("What were the 2025 sales per segment in the segment kpi table?")
    assert data["ok"] and data["result"]["sql_result"]["rows"]
    doc = svc.ask("Who is the chief executive officer?")
    view = presentation.present(doc["result"])
    assert view["document"]["unavailable"] and "another process" in view["document"]["unavailable"]
    assert view["abstained"]


def test_no_documents_document_question_explains(tmp_path):
    svc = make_service(tmp_path, with_docs=False)
    try:
        out = svc.ask("Who is the chief executive officer?")
        assert out["ok"] and "No PDF documents" in out["result"]["answer"]
        assert svc.ask("   ")["error"] == "Please enter a question."
    finally:
        svc.close()


def test_unexpected_backend_error_is_contained(service):
    class Boom(FakeRAG):
        def answer(self, question):
            raise KeyError("internal detail")

    service._rag = Boom([])
    out = service.ask("Who is the chief executive officer of the company?")
    assert not out["ok"] and "KeyError" in out["error"] and "internal detail" not in out["error"]


# ---------------------------------------------------------------------- conversations
def test_conversation_isolation_clear_delete_and_interruption():
    a, b = ConversationStore(), ConversationStore()  # two browser sessions
    first = a.active
    q = first.add_question("What were total sales in 2025?")
    a.active.add_answer(q, {"ok": True, "result": {"answer": "61,352"}})
    assert first.title.startswith("What were total sales") and len(b.active.messages) == 0
    second = a.new()
    assert second.id != first.id and second.messages == [] and a.new() is second  # no piling of empty chats
    second.add_question("Unanswered question")
    assert second.mark_interrupted() == 1 and second.messages[0].status == "interrupted"
    a.select(first.id)
    assert [m.role for m in a.active.messages] == ["user", "assistant"]
    failed = a.active.add_question("Broken?")
    a.active.add_answer(failed, {"ok": False, "error": "Cannot reach Ollama"})
    assert failed.status == "error" and a.active.messages[-1].content == "Cannot reach Ollama"
    a.active.clear()
    assert a.active.messages == [] and a.active.title == "New conversation"
    a.delete(first.id)
    assert len(a) == 1 and a.active_id == second.id
    a.delete(second.id)
    assert len(a) == 1 and a.active.messages == []


# ---------------------------------------------------------------------- presentation
def test_presentation_labels_raw_derived_and_rejected_results():
    sql = {"answer": "x", "sql": "SELECT year, sales_usd_m FROM t", "columns": ["year", "sales_usd_m"],
           "rows": [[2024, 10], [2025, 12]], "truncated": True, "error": None,
           "analytics": {"period_changes": [{"group": None, "metric": "sales_usd_m", "from_period": 2024, "to_period": 2025,
                                             "from_value": 10, "to_value": 12, "absolute_change": 2, "percent_change": 20.0}]},
           "chart": {"type": "line", "x": "year", "y": ["sales_usd_m"], "data": [], "reason": "r"},
           "presentation": {"model_derived_columns_ignored": ["sales_change"], "metrics_not_asked": []},
           "raw_result": {"columns": ["year", "sales_usd_m", "sales_change"], "rows": [[2024, 10, None], [2025, 12, 2]]},
           "sources": [{"table": "t", "source_file": "t.csv", "sheet": None}], "generation": {"retry_for_scope": {}}}
    view = presentation.present({"route": "sql", "routing": {"reason": "r"}, "answer": "x", "sql_result": sql})
    s = view["structured"]
    assert s["raw"]["columns"] == ["year", "sales usd m (USD m)"] and s["derived"][0]["title"] == "Period change"
    assert s["derived"][0]["rows"][0][-2:] == [2, 20.0] and s["chart"]["type"] == "line"
    assert any("sales_change" in n for n in s["notes"]) and any("regenerated once" in n for n in s["notes"])
    assert s["warnings"] and s["original_query_result"]["columns"][-1] == "sales change"
    refused = {"answer": "I cannot calculate these shares reliably", "sql": "SELECT", "columns": [], "rows": [],
               "abstained": True, "rejected_result": {"columns": ["segment", "share"], "rows": [["A", 0.5]]},
               "analytics": {"shares": [{"label": "A", "share_percent": 50}]}, "chart": {"type": "pie"}}
    r = presentation.present({"route": "sql", "answer": refused["answer"], "abstained": True, "sql_result": refused})["structured"]
    assert r["derived"] == [] and r["chart"] is None and r["raw"] is None and r["rejected"]["rows"] == [["A", 0.5]]
    assert presentation.unit_of("employees_fte") == "FTE" and presentation.unit_of("segment") is None


def test_presentation_of_citations_and_hybrid_sections():
    rag = FakeRAG([]).answer("q")
    rag["calculation"] = {"kind": "change", "sentence": "Sales fell by 1.", "difference": -1, "percent": -1.0, "unit": "USD m",
                          "operands": [{"raw": "10", "unit": "USD m", "label": "2024", "page": 7}]}
    view = presentation.present({"route": "hybrid", "answer": "both", "rag_result": rag,
                                 "sql_result": {"answer": "s", "columns": ["a"], "rows": [[1]], "sources": []},
                                 "sub_questions": {"sql": "data part", "document": "doc part", "split": True}})
    assert view["source_label"].startswith("Hybrid") and view["sub_questions"] == {"data": "data part", "document": "doc part"}
    doc = view["document"]
    assert doc["citations"][0]["page"] == 7 and doc["claims"][0]["status"] == "SUPPORTED"
    assert doc["status_label"] == "Supported by the cited evidence" and doc["calculation"]["operands"][0]["page"] == 7
    abst = presentation.present_rag({"answer": ABSTENTION, "citations": [], "validation": {"overall_status": "ABSTAINED"}})
    assert abst["abstained"] and abst["citations"] == []


# ---------------------------------------------------------------------- application startup (Streamlit AppTest)
class FakeAppService:
    """Minimal service for the Streamlit script: no backend, deterministic answers."""

    def __init__(self, config):
        from app.ui.documents import IndexStatus
        from app.ui.service import ComponentStatus, Health

        self.config = config
        ok = ComponentStatus(True, "ok")
        self._health = Health(ok, ok, ok, IndexStatus(True, [], 1, "1 document"), ComponentStatus(True, "1 table"))
        self.datasets = type("D", (), {"datasets": lambda self: [], "preview": None})()
        self.questions = []

    def health(self, refresh=False):
        return self._health

    def warm_up(self):
        pass

    def document_search_status(self):
        return self._health.ollama

    def ask(self, question):
        self.questions.append(question)
        result = {"route": "sql", "routing": {"reason": "data", "signals": {}}, "answer": "Result: total 42.",
                  "abstained": False, "sql_result": {"answer": "Result: total 42.", "sql": "SELECT SUM(x) AS total FROM t",
                                                     "columns": ["total"], "rows": [[42]], "sources": [], "chart": {"type": None}}}
        return {"ok": True, "result": result, "error": None, "elapsed_s": 0.1}


@pytest.fixture
def app(monkeypatch):
    from streamlit.testing.v1 import AppTest

    import app.ui.service as service_module

    monkeypatch.setattr(service_module, "InsightService", FakeAppService)
    import streamlit as st

    st.cache_resource.clear()
    test = AppTest.from_file(str(ROOT / "app" / "streamlit_app.py"), default_timeout=60)
    yield test
    st.cache_resource.clear()


def test_application_starts_and_answers(app):
    app.run()
    assert not app.exception
    markdown = " ".join(m.value for m in app.markdown)
    assert "irag-brand" in " ".join(m.value for m in app.sidebar.markdown) and "Ask questions about your documents" in markdown
    assert any("total sales in 2025" in b.label for b in app.button), "example questions are shown"
    app.chat_input[0].set_value("What is the total in the t table?").run()
    assert not app.exception
    assert any("Structured data (SQL)" in m.value for m in app.markdown)
    assert app.dataframe, "the result table is rendered"
    assert any("SELECT SUM(x)" in c.value for c in app.code), "the SQL is available in an expander"
    new =[b for b in app.sidebar.button if "New conversation" in b.label][0]
    new.click().run()
    assert not app.exception and "Ask questions about your documents" in " ".join(m.value for m in app.markdown)
