"""Streamlit rendering for the InsightRAG app (Phase 5). Display only: no routing, SQL or validation here."""

from __future__ import annotations

import html

import streamlit as st

from app.ui.presentation import format_value, present

EXAMPLES = {
    "Document (annual report)": [
        "What were ArcelorMittal's total sales in 2025?",
        "How many employees did ArcelorMittal have at the end of 2025?",
        "What are ArcelorMittal's reportable segments?",
    ],
    "Structured data (sample datasets)": [
        "What is the total order revenue per customer region in the orders data?",
        "How did Europe sales change between 2024 and 2025 in the segment KPI table?",
        "What share of 2025 employees in the segment KPI data does each segment have?",
    ],
    "Hybrid (data + document)": [
        "Compare Europe's 2025 sales in the segment KPI table with the Europe sales figure in the annual report.",
    ],
}

CSS = """
<style>
.block-container {padding-top: 2rem; max-width: 1100px;}
.irag-brand {font-size: 1.45rem; font-weight: 700; letter-spacing: .2px; margin-bottom: 0;}
.irag-brand span {color: #7c9cff;}
.irag-tag {color: #9aa4b2; font-size: .82rem; margin-top: -2px;}
.irag-badge {display: inline-block; padding: 2px 10px; border-radius: 999px; font-size: .75rem; font-weight: 600;
             margin: 0 6px 6px 0; border: 1px solid transparent;}
.irag-document {background: rgba(124,156,255,.14); color: #a9bcff; border-color: rgba(124,156,255,.35);}
.irag-sql {background: rgba(52,211,153,.12); color: #6ee7b7; border-color: rgba(52,211,153,.35);}
.irag-hybrid {background: rgba(251,191,36,.12); color: #fcd34d; border-color: rgba(251,191,36,.35);}
.irag-ok {background: rgba(52,211,153,.12); color: #6ee7b7;}
.irag-warn {background: rgba(251,191,36,.12); color: #fcd34d;}
.irag-bad {background: rgba(248,113,113,.12); color: #fca5a5;}
.irag-card {border: 1px solid rgba(255,255,255,.08); border-radius: 12px; padding: 14px 16px; background: rgba(255,255,255,.02);
            height: 100%;}
.irag-card h4 {margin: 0 0 6px 0; font-size: .95rem;}
.irag-card p {color: #b4bcc8; font-size: .86rem; margin: 0;}
.irag-muted {color: #9aa4b2; font-size: .82rem;}
.irag-cite {border-left: 3px solid #7c9cff; padding: 4px 10px; margin: 4px 0; color: #c9d1dc; font-size: .86rem;
            white-space: pre-wrap;}
</style>
"""


def badge(text: str, kind: str) -> str:
    return f'<span class="irag-badge irag-{kind}">{html.escape(text)}</span>'


def status_dot(ok: bool, warn: bool = False) -> str:
    return "🟢" if ok else ("🟡" if warn else "🔴")


# ---------------------------------------------------------------------- welcome
def welcome(health) -> str | None:
    """Welcome screen; returns an example question when one is clicked."""

    st.markdown("## Ask questions about your documents and data")
    st.markdown('<p class="irag-muted">InsightRAG answers from your uploaded PDF documents and CSV/Excel datasets, '
                "locally. Every figure is either cited to a document page or computed from a read-only SQL result.</p>",
                unsafe_allow_html=True)
    cols = st.columns(3)
    cards = [
        ("📄 Documents · RAG", "Hybrid BM25 + vector search over your PDFs, cross-encoder reranking, an answer from the "
         "local model, and claim-level citation validation. Unsupported answers abstain."),
        ("🗄️ Data · SQL", "Questions about CSV/Excel tables become validated, read-only SQLite queries (time limit, "
         "row limit). Changes, shares and ratios are calculated in Python from the raw rows."),
        ("🔀 Hybrid", "A question that names both the data and the document is split. Each part is answered by its "
         "own engine, and the findings are shown side by side."),
    ]
    for col, (title, body) in zip(cols, cards):
        col.markdown(f'<div class="irag-card"><h4>{title}</h4><p>{body}</p></div>', unsafe_allow_html=True)
    st.write("")
    st.markdown("**Supported formats:** PDF documents with a text layer (`.pdf`) · datasets as `.csv` or Excel `.xlsx`.")
    if not health.documents.ok or not health.tables.ok:
        st.info("Upload documents and datasets from the sidebar. "
                + ("" if health.documents.ok else f"Documents: {health.documents.message} ")
                + ("" if health.tables.ok else f"Data: {health.tables.message}"))
    st.markdown("#### Try an example")
    chosen = None
    for group, questions in EXAMPLES.items():
        st.markdown(f'<p class="irag-muted">{html.escape(group)}</p>', unsafe_allow_html=True)
        for q in questions:
            if st.button(q, key=f"ex-{hash(q)}", width="stretch"):
                chosen = q
    return chosen


# ---------------------------------------------------------------------- results
def _dataframe(table: dict, height="auto"):
    import pandas as pd

    frame = pd.DataFrame(table["rows"], columns=table["columns"])
    st.dataframe(frame, hide_index=True, width="stretch", height=height)


def _chart(spec: dict) -> None:
    import altair as alt
    import pandas as pd

    data = pd.DataFrame(spec["data"])
    x, ys = spec["x"], spec["y"]
    kind = spec["type"]
    if kind == "pie":
        chart = alt.Chart(data).mark_arc(innerRadius=55).encode(
            theta=alt.Theta(f"{ys[0]}:Q"), color=alt.Color(f"{x}:N", legend=alt.Legend(title=x.replace('_', ' '))),
            tooltip=[x, ys[0]])
    else:
        long = data.melt(id_vars=[x], value_vars=ys, var_name="metric", value_name="value")
        if kind == "line":
            chart = alt.Chart(long).mark_line(point=True).encode(
                x=alt.X(f"{x}:O", title=x.replace("_", " ")), y=alt.Y("value:Q", title=", ".join(ys)),
                color="metric:N", tooltip=[x, "metric", "value"])
        else:
            chart = alt.Chart(long).mark_bar().encode(
                x=alt.X(f"{x}:N", sort=None, title=x.replace("_", " ")), y=alt.Y("value:Q", title=", ".join(ys)),
                color="metric:N", xOffset="metric:N", tooltip=[x, "metric", "value"])
    st.altair_chart(chart.properties(height=320), width="stretch")


def render_structured(view: dict, key: str, show_answer: bool = True) -> None:
    if show_answer:
        st.markdown(_text(view["answer"]))
    for warning in view["warnings"]:
        st.warning(warning)
    if view["error"] and not view["abstained"] and view["error"] not in view["answer"]:
        st.caption(f"Data engine: {view['error']}")
    if view["derived"]:
        for table in view["derived"]:
            st.markdown(f"**Derived · {table['title']}** " + badge("calculated", "warn"), unsafe_allow_html=True)
            _dataframe({"columns": table["columns"], "rows": [[format_value(v) for v in r] for r in table["rows"]]})
            st.caption(table["note"])
    if view["raw"]:
        st.markdown(f"**{view['raw_title']}** " + badge("raw", "ok"), unsafe_allow_html=True)
        _dataframe(view["raw"])
    if view["chart"]:
        _chart(view["chart"])
        st.caption(f"Chart: {view['chart']['type']}. {view['chart_reason'] or ''}")
    if view["notes"]:
        for note in view["notes"]:
            st.caption("ℹ️ " + note)
    if view["tables"]:
        st.caption("Tables used: " + ", ".join(
            f"{t['table']} ({t['source_file']}{' / ' + t['sheet'] if t.get('sheet') else ''})" for t in view["tables"]))
    if view["sql"]:
        with st.expander("SQL query (validated, read-only)"):
            st.code(view["sql"], language="sql")
    if view["original_query_result"]:
        with st.expander("Original query result (before model-computed columns were removed)"):
            _dataframe(view["original_query_result"])
    if view["rejected"]:
        with st.expander("Rejected result (failed validation; not used for the answer)"):
            _dataframe(view["rejected"])


def render_document(view: dict, key: str, show_answer: bool = True) -> None:
    if view["unavailable"]:
        st.warning(f"Document search is unavailable: {view['unavailable']}")
        return
    if show_answer:
        st.markdown(_text(view["answer"]))
    kind = "ok" if view["status"] == "SUPPORTED" else ("bad" if view["status"] in ("ABSTAINED", "UNSUPPORTED") else "warn")
    st.markdown(badge(view["status_label"], kind), unsafe_allow_html=True)
    if view["reason"] and view["status"] != "SUPPORTED":
        st.caption(view["reason"])
    if view["calculation"]:
        calc = view["calculation"]
        st.markdown("**Derived · calculation from cited values** " + badge("calculated", "warn"), unsafe_allow_html=True)
        _dataframe({"columns": ["Value", "Unit", "Label", "Page"],
                    "rows": [[o["value"], o["unit"], o["label"], o["page"]] for o in calc["operands"]]})
        st.caption(calc["sentence"] or "")
    if view["citations"]:
        pages = ", ".join(f"{s} p.{p}" for s, p in view["pages"])
        st.markdown(f"**Sources:** {html.escape(pages)}")
        for c in view["citations"]:
            title = f"[{c['n']}] {c['source']} — page {c['page']}" + (f" · {c['section']}" if c.get("section") else "")
            with st.expander(title):
                st.markdown(f'<div class="irag-cite">{html.escape(c["text"])}</div>', unsafe_allow_html=True)
    if view["claims"]:
        with st.expander("Claim validation"):
            for claim in view["claims"]:
                where = f" ({claim['source']}, p.{claim['page']})" if claim.get("source") else ""
                st.markdown(f"- **{claim['status']}** · {html.escape(claim['claim'] or '')}{html.escape(where)}  \n"
                            f"  <span class='irag-muted'>{html.escape(claim['explanation'] or '')}</span>",
                            unsafe_allow_html=True)
    if view["completeness"] == "incomplete":
        st.caption("ℹ️ The completeness check found items the document lists that the answer omits.")


def _text(text: str) -> str:
    # Answers are plain text with "- " bullets; keep line breaks, and stop "$" from being read as LaTeX.
    return (text or "").replace("$", "\\$").replace("\n", "  \n")


def render_response(response: dict, key: str, debug: bool = False) -> None:
    if not response.get("ok"):
        st.error(response.get("error") or "The question could not be answered.")
        return
    view = present(response["result"])
    route_kind = {"document": "document", "sql": "sql", "hybrid": "hybrid"}.get(view["route"], "document")
    st.markdown(badge(view["source_label"], route_kind)
                + (badge("abstained", "bad") if view["abstained"] else "")
                + f'<span class="irag-muted">{response.get("elapsed_s", 0):.1f}s</span>', unsafe_allow_html=True)
    if view["route"] == "hybrid":
        if view["sub_questions"]:
            st.caption(f"Split into · data: “{view['sub_questions']['data']}” · document: “{view['sub_questions']['document']}”")
        left, right = st.columns(2)
        with left:
            st.markdown("##### 🗄️ Structured-data findings")
            if view["structured"]:
                render_structured(view["structured"], key + "-sql")
        with right:
            st.markdown("##### 📄 Document evidence")
            if view["document"]:
                render_document(view["document"], key + "-doc")
    elif view["route"] == "sql" and view["structured"]:
        render_structured(view["structured"], key + "-sql")
    elif view["document"]:
        render_document(view["document"], key + "-doc")
    else:
        st.markdown(_text(view["answer"]))
    if debug:
        with st.expander("Routing details"):
            st.json({"route": view["route"], "reason": view["routing_reason"],
                     "signals": (response["result"].get("routing") or {}).get("signals")})
