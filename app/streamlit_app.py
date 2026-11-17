"""InsightRAG web application (Phase 5).

    streamlit run app/streamlit_app.py

The UI only displays. Questions go to ``app.ui.service.InsightService``, which calls the existing
``engine.Assistant`` (router, RAG pipeline, NL-to-SQL with read-only execution).
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:  # `streamlit run app/streamlit_app.py` puts app/ (not the project root) on sys.path
    sys.path.insert(0, str(ROOT))

import streamlit as st  # noqa: E402

from app.ui import components as ui  # noqa: E402
from app.ui.config import AppConfig  # noqa: E402
from app.ui.conversations import ConversationStore  # noqa: E402
from app.ui.datasets import DatasetError  # noqa: E402
from app.ui.documents import DocumentError  # noqa: E402
from app.ui.service import InsightService, explain_error  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")


@st.cache_resource(show_spinner=False)
def get_service() -> InsightService:
    return InsightService(AppConfig.from_env())


def _flash(kind: str, text: str) -> None:
    st.session_state.setdefault("flash", []).append((kind, text))


def _show_flash() -> None:
    for kind, text in st.session_state.pop("flash", []):
        getattr(st, kind)(text)


def sidebar(service: InsightService, store: ConversationStore, health) -> None:
    with st.sidebar:
        st.markdown('<p class="irag-brand">Insight<span>RAG</span></p>'
                    '<p class="irag-tag">Local AI business intelligence · documents + data</p>', unsafe_allow_html=True)
        if st.button("➕ New conversation", width="stretch"):
            store.new()
            st.rerun()

        st.markdown("##### Conversations")
        for conv in store.list():
            active = conv.id == store.active_id
            cols = st.columns([5, 1])
            label = ("▸ " if active else "") + conv.title
            if cols[0].button(label, key=f"conv-{conv.id}", width="stretch", disabled=active):
                store.select(conv.id)
                st.rerun()
            if cols[1].button("✕", key=f"del-{conv.id}", help="Delete this conversation"):
                store.delete(conv.id)
                st.rerun()

        st.divider()
        docs = health.documents.documents
        with st.expander(f"📄 Documents ({len(docs)})", expanded=not docs):
            for doc in docs:
                unknown = doc.indexed_vectors is None  # the index could not be opened
                state = "indexed" if doc.indexed else ("index unavailable" if unknown else "not indexed")
                st.markdown(f"{ui.status_dot(doc.indexed, warn=unknown)} **{doc.name}**  \n"
                            f"<span class='irag-muted'>{doc.pages} pages · {doc.chunks} chunks · {state}</span>",
                            unsafe_allow_html=True)
            if not health.documents.ok:
                st.caption(health.documents.message)
            with st.form("doc-upload", clear_on_submit=True):
                files = st.file_uploader("Upload PDF", type=["pdf"], accept_multiple_files=True,
                                         help=f"Text-based PDFs up to {service.config.max_pdf_mb} MB. Existing files are never overwritten.")
                if st.form_submit_button("Add documents", width="stretch") and files:
                    for f in files:
                        with st.spinner(f"Indexing {f.name} (embedding chunks)…"):
                            try:
                                info = service.add_document(f.name, f.getvalue())
                                _flash("success", f"Added {info['name']}: {info['pages']} pages, {info['chunks']} chunks indexed.")
                            except DocumentError as error:
                                _flash("error", str(error))
                            except Exception as error:
                                _flash("error", explain_error(error, service.config))
                    st.rerun()

        datasets = service.datasets.datasets() if health.tables.ok else []
        with st.expander(f"🗄️ Datasets ({len(datasets)})", expanded=not datasets):
            for ds in datasets:
                st.markdown(f"**{ds['dataset']}** <span class='irag-muted'>· {ds['source_file']}</span>", unsafe_allow_html=True)
                for t in ds["tables"]:
                    st.markdown(f"<span class='irag-muted'>└ table <code>{t['name']}</code> · {t['rows']} rows · "
                                f"{len(t['columns'])} columns</span>", unsafe_allow_html=True)
            if datasets and st.button("Browse tables and columns", width="stretch"):
                st.session_state["view"] = "datasets"
                st.rerun()
            with st.form("data-upload", clear_on_submit=True):
                files = st.file_uploader("Upload CSV / Excel", type=["csv", "xlsx"], accept_multiple_files=True,
                                         help=f"Up to {service.config.max_dataset_mb} MB. Each Excel sheet becomes a table.")
                replace = st.checkbox("Replace existing dataset with the same name")
                if st.form_submit_button("Add datasets", width="stretch") and files:
                    for f in files:
                        try:
                            info = service.add_dataset(f.name, f.getvalue(), replace=replace)
                            tables = ", ".join(f"{t['name']} ({t['rows']} rows)" for t in info["tables"])
                            _flash("success", f"{'Replaced' if info['replaced'] else 'Loaded'} {info['dataset']}: {tables}.")
                            for t in info["tables"]:
                                for w in t["warnings"]:
                                    _flash("warning", f"{t['name']}: {w}")
                        except DatasetError as error:
                            _flash("error", str(error))
                        except Exception as error:
                            _flash("error", explain_error(error, service.config))
                    st.rerun()

        st.divider()
        with st.expander("⚙️ Settings & status"):
            rows = [("Ollama", health.ollama), ("Model", health.model), ("Vector index", health.vector_index),
                    ("Document search", service.document_search_status()), ("Datasets", health.tables)]
            for name, status in rows:
                st.markdown(f"{ui.status_dot(status.ok)} **{name}** <span class='irag-muted'>{status.message}</span>",
                            unsafe_allow_html=True)
            deterministic = st.toggle("Deterministic generation (temperature 0, seed 42)", value=service.config.deterministic)
            if deterministic != service.config.deterministic:
                service.set_deterministic(deterministic)
            st.session_state["debug"] = st.toggle("Show routing details", value=st.session_state.get("debug", False))
            c1, c2 = st.columns(2)
            if c1.button("Re-check", width="stretch"):
                service.retry_components()
                st.rerun()
            if c2.button("Rebuild document index", width="stretch", disabled=not health.vector_index.ok,
                         help="Re-embeds every library PDF from scratch."):
                with st.spinner("Rebuilding the document index…"):
                    try:
                        info = service.rebuild_index()
                        _flash("success", f"Index rebuilt: {info['vectors']} vectors.")
                    except Exception as error:
                        _flash("error", explain_error(error, service.config))
                st.rerun()
            st.caption("Configuration (environment variables, see README):")
            st.json(service.config.public_dict(), expanded=False)


def datasets_view(service: InsightService) -> None:
    st.markdown("## Datasets")
    if st.button("← Back to chat"):
        st.session_state["view"] = "chat"
        st.rerun()
    for ds in service.datasets.datasets():
        st.markdown(f"### {ds['dataset']}  \n<span class='irag-muted'>source file: {ds['source_file']}</span>",
                    unsafe_allow_html=True)
        for t in ds["tables"]:
            sheet = f" · sheet {t['sheet']}" if t["sheet"] else ""
            with st.expander(f"Table {t['name']} · {t['rows']} rows · {len(t['columns'])} columns{sheet}", expanded=True):
                if t.get("warnings"):
                    st.warning(t["warnings"])
                ui._dataframe({"columns": ["Column", "Original header", "Inferred type", "Missing", "Distinct", "Examples"],
                               "rows": [[c["name"], c["original_name"], c["type"], c["missing"], c["distinct"],
                                         ", ".join(map(str, c["examples"]))] for c in t["columns"]]})
                preview = service.datasets.preview(t["name"])
                if preview.ok:
                    st.caption(f"Preview (first {len(preview.rows)} rows, read-only)")
                    ui._dataframe({"columns": preview.columns, "rows": [list(r) for r in preview.rows]})
                else:
                    st.error(f"Preview failed: {preview.error}")


def chat_view(service: InsightService, store: ConversationStore, health) -> None:
    conv = store.active
    interrupted = conv.mark_interrupted()
    if interrupted:
        st.warning("The previous question was interrupted before it was answered. Ask it again to retry.")

    queued = st.session_state.pop("queued_question", None)
    if not conv.messages and not queued:
        queued = ui.welcome(health)

    for i, message in enumerate(conv.messages):
        with st.chat_message(message.role, avatar="🧑" if message.role == "user" else "✨"):
            if message.role == "user":
                st.markdown(message.content)
                if message.status == "interrupted":
                    st.caption("⚠️ Interrupted: no answer was produced.")
            else:
                ui.render_response(message.response or {}, key=f"{conv.id}-{i}", debug=st.session_state.get("debug", False))

    if not (health.can_answer_documents or health.can_answer_data):
        st.caption("⚠️ " + (health.ollama.message if not health.ollama.ok else health.model.message
                            if not health.model.ok else "Upload a document or dataset to start."))
    prompt = st.chat_input("Ask about your documents or data…")
    question = prompt or queued
    if question:
        pending = conv.add_question(question)
        with st.chat_message("user", avatar="🧑"):
            st.markdown(question)
        with st.chat_message("assistant", avatar="✨"):
            with st.spinner("Thinking… routing the question, retrieving evidence and validating"):
                response = service.ask(question)
            conv.add_answer(pending, response)
            ui.render_response(response, key=f"{conv.id}-{len(conv.messages) - 1}", debug=st.session_state.get("debug", False))
        st.rerun()

    if conv.messages and st.button("🧹 Clear this conversation"):
        conv.clear()
        st.rerun()


def main() -> None:
    st.set_page_config(page_title="InsightRAG", page_icon="✨", layout="wide")
    st.markdown(ui.CSS, unsafe_allow_html=True)
    service = get_service()
    if "conversations" not in st.session_state:
        st.session_state["conversations"] = ConversationStore()
    store: ConversationStore = st.session_state["conversations"]
    try:
        if not st.session_state.get("_warm"):
            with st.spinner("Starting InsightRAG: reading the document library and loading the retrieval models "
                            "(slow only on the first start)…"):
                service.health()
                service.warm_up()
            st.session_state["_warm"] = True
        health = service.health()
    except Exception as error:  # never leave a blank page
        st.error(explain_error(error, service.config))
        return
    sidebar(service, store, health)
    _show_flash()
    if st.session_state.get("view") == "datasets":
        datasets_view(service)
    else:
        chat_view(service, store, health)


main()
