"""Phase 3 unit and regression tests (no models, no Qdrant, no PDF)."""

import pytest

from app.rag import generator as generator_module
from app.rag.citation_validator import validate_citation
from app.rag.document_context import (
    ScopeIndex,
    classify_scope,
    find_segment_names,
    page_sections,
    render_chunk,
)
from app.rag.generator import Generator, build_prompt
from app.rag.multihop import decompose, merge_rankings
from app.rag.pipeline import ABSTENTION, PipelineConfig, RAGPipeline
from app.rag.scope_validation import arithmetic_issue, is_contradiction, row_label
from app.rag import citation_validator as cv

SRC = "report.pdf"
SEGMENT_LIST_PAGE = {"source": SRC, "page": 1, "text": "The Company reports its operations in three reportable segments: North America, Europe and Mining."}
SEGMENT_PAGE_TEXT = (
    "Consolidated narrative about the group.\n"
    "North America\n"
    "Performance for the year ended December 31,\n"
    "Sales 12,335 11,896\n"
    "Crude steel production (thousand tonnes) 7,755 7,538\n"
    "AMNS Calvert\n"
    "Performance for the year ended December 31,\n"
    "Sales 900 800"
)


# ---------------------------------------------------------------- Task 1: scope-aware context


def test_segment_names_come_from_the_document():
    assert find_segment_names([SEGMENT_LIST_PAGE]) == ["North America", "Europe", "Mining"]


def test_classify_scope_and_jv_alias():
    names = ["North America", "India and joint ventures"]
    assert classify_scope(None, names) == "consolidated"
    assert classify_scope("North America", names) == "segment:North America"
    assert classify_scope("India and JVs", names) == "segment:India and JVs"
    assert classify_scope("AMNS Calvert", names) == "unit:AMNS Calvert"


def test_page_sections_require_heading_above_performance_table():
    sections = page_sections(SEGMENT_PAGE_TEXT, ["North America"])
    assert [(s["heading"], s["scope"]) for s in sections] == [
        ("North America", "segment:North America"), ("AMNS Calvert", "unit:AMNS Calvert")]
    # A footnote or sentence line above the table is not a heading.
    assert page_sections("Others include holding companies.\nPerformance for the year ended", []) == []


def test_scope_index_enriches_chunks_without_changing_text():
    page = {"source": SRC, "page": 2, "text": SEGMENT_PAGE_TEXT}
    words = SEGMENT_PAGE_TEXT.split()
    chunks = [{"source": SRC, "page": 2, "text": " ".join(words[:8])}, {"source": SRC, "page": 2, "text": " ".join(words[8:])}]
    index = ScopeIndex(chunks, [SEGMENT_LIST_PAGE, page])

    first, second = index.enrich(chunks)
    assert first["text"] == chunks[0]["text"]  # text unchanged (index stays valid)
    assert first["scope"] == "consolidated" and first["sections"][0]["heading"] == "North America"
    assert second["scope"] == "segment:North America"
    rendered = render_chunk(first)
    assert "[Scope: North America segment only (not group total)]" in rendered
    assert rendered.index("[Scope:") < rendered.index("North America Performance")


def test_render_chunk_without_metadata_is_plain_text():
    assert render_chunk({"text": "Total sales 61,352"}) == "Total sales 61,352"


def test_chart_values_are_attached_when_the_chunk_holds_them():
    class V:
        def __init__(self, category, series, value):
            self.category, self.series, self.value = category, series, value

    page = {"source": SRC, "page": 3, "text": "Overview 55.6 54.0 57.9 54.3 2025 2024"}
    chunk = {"source": SRC, "page": 3, "text": page["text"]}
    charts = {3: [V("Crude steel production (million tonnes)", "2025", "55.6"), V("Steel shipments (million tonnes)", "2025", "54.0")]}
    meta = ScopeIndex([chunk], [page], charts).metadata(chunk)
    assert meta["chart"] == ["Crude steel production (million tonnes): 2025 = 55.6", "Steel shipments (million tonnes): 2025 = 54.0"]


# ---------------------------------------------------------------- Task 2/6: prompt and generation options


def test_scope_prompt_rules_and_legacy_prompt():
    chunk = {"text": "Sales 12,335", "scope": "segment:North America", "sections": []}
    prompt = build_prompt("What were total sales?", [chunk])
    assert "consolidated" in prompt and "[Scope at start: North America segment only" in prompt
    legacy = build_prompt("What were total sales?", [chunk], scope_aware=False)
    assert "[Scope" not in legacy and "8. Use group-level" not in legacy


def test_generator_passes_options_and_records_settings(monkeypatch):
    sent = {}

    def fake_chat(model, messages, options=None):
        sent["options"] = options
        return {"message": {"content": "Sales were $1 million."}}

    monkeypatch.setattr(generator_module.ollama, "chat", fake_chat)
    gen = Generator(options={"temperature": 0, "seed": 7})
    gen.generate("q", [{"text": "t"}])
    assert sent["options"] == {"temperature": 0, "seed": 7}
    assert gen.settings()["options_applied"] is True


def test_generator_defaults_send_no_options(monkeypatch):
    def fake_chat(model, messages):  # production signature: no options keyword
        return {"message": {"content": "ok"}}

    monkeypatch.setattr(generator_module.ollama, "chat", fake_chat)
    gen = Generator()
    assert gen.generate("q", [{"text": "t"}]) == "ok"
    assert gen.settings()["options"] is None


def test_generator_falls_back_when_options_rejected(monkeypatch):
    calls = []

    def fake_chat(model, messages, **kwargs):
        calls.append(kwargs)
        if "options" in kwargs:
            raise TypeError("unexpected keyword argument 'options'")
        return {"message": {"content": "ok"}}

    monkeypatch.setattr(generator_module.ollama, "chat", fake_chat)
    gen = Generator(options={"seed": 1})
    assert gen.generate("q", [{"text": "t"}]) == "ok"
    assert gen.last_options_applied is False and len(calls) == 2


# ---------------------------------------------------------------- Task 3: scope/metric validation (Phase 2 regressions)


def _status(answer, citation, question=None):
    return validate_citation(answer, [citation], question=question)


SEGMENT_CITATION = {
    "source": SRC, "page": 97, "scope": "segment:Sustainable Solutions",
    "sections": [{"heading": "Mining", "scope": "segment:Mining", "word_offset": 6}],
    "text": "ArcelorMittal Sustainable Solutions results were stable. Mining Performance for the year ended December 31, "
            "(in millions of USD unless otherwise shown) 2025 2024 Iron ore production (million tonnes) 35.3 27.9 7.4 26.5 %",
}


def test_regression_segment_figure_rejected_for_group_claim():
    # Phase 2 Q018: Mining-segment iron ore reported as the company figure.
    result = _status("ArcelorMittal's iron ore production was 35.3 million tonnes in 2025.", SEGMENT_CITATION,
                     "What was ArcelorMittal's iron ore production in 2025?")
    assert result["overall_status"] == "UNSUPPORTED"
    assert "Mining" in result["claims"][0]["explanation"] and is_contradiction(result["claims"][0]["explanation"])


def test_segment_figure_accepted_when_question_names_segment():
    result = _status("Mining iron ore production was 35.3 million tonnes in 2025.", SEGMENT_CITATION,
                     "What was the Mining segment's iron ore production in 2025?")
    assert result["overall_status"] == "SUPPORTED"


def test_regression_narrative_segment_phrase():
    citation = {"source": SRC, "page": 59, "text": "In 2025, iron ore production in the Mining segment totaled approximately 35.3 million tonnes (27.9 million tonnes in 2024)."}
    result = _status("ArcelorMittal's iron ore production was 35.3 million tonnes in 2025.", citation)
    assert result["overall_status"] == "UNSUPPORTED"


def test_regression_alternative_measure_rom():
    # Phase 2 Q009: aggregate run-of-mine production given as iron ore production.
    citation = {"source": SRC, "page": 70, "text": "2025 aggregate ROM iron ore production, millions of tonnes3 112.3 Coal"}
    result = _status("ArcelorMittal's iron ore production in 2025 was 112.3 million tonnes.", citation)
    assert result["overall_status"] == "UNSUPPORTED"
    assert "aggregate" in result["claims"][0]["explanation"] or "rom" in result["claims"][0]["explanation"].lower()


def test_regression_shipments_number_for_production_claim():
    # Phase 2 Q008 run 3: the shipments figure stated as crude steel production.
    citation = {"source": SRC, "page": 26, "text": "Steel shipments for the year ended December 31, 2025 totaled 54.0 million tonnes."}
    assert _status("Crude steel production in 2025 was 54.0 million tonnes.", citation)["overall_status"] == "UNSUPPORTED"
    assert _status("Steel shipments in 2025 were 54.0 million tonnes.", citation)["overall_status"] == "SUPPORTED"


def test_chart_values_use_their_own_label_unit_and_year():
    citation = {
        "source": SRC, "page": 3,
        "chart": ["Crude steel production (million tonnes): 2025 = 55.6, 2024 = 57.9",
                  "Steel shipments (million tonnes): 2025 = 54.0, 2024 = 54.3",
                  "Sales (billion USD): 2025 = 61.4, 2024 = 62.4"],
        "text": "presence in Asia. 55.6 54.0 61.4 57.9 54.3 62.4 2025 2024 Crude steel production (million tonnes) Steel shipments (million tonnes) Sales (billion USD)",
    }
    assert _status("Crude steel production was 55.6 million tonnes in 2025.", citation)["overall_status"] == "SUPPORTED"
    assert _status("Crude steel production was 54.0 million tonnes in 2025.", citation)["overall_status"] == "UNSUPPORTED"
    assert _status("Crude steel production was 57.9 million tonnes in 2025.", citation)["overall_status"] == "UNSUPPORTED"


def test_every_reported_number_needs_support():
    # Phase 2 Q022 run 3: one supported figure must not carry an unsupported one.
    citation = {"source": SRC, "page": 26, "text": "Steel shipments for the year ended December 31, 2025 totaled 54.0 million tonnes."}
    result = _status("In 2025, steel shipments totaled 54.0 million tonnes, while crude steel production totaled 7,755 thousand tonnes.", citation)
    assert result["overall_status"] == "UNSUPPORTED"


def test_consolidated_table_still_supported():
    citation = {"source": SRC, "page": 95, "text": "(in $ millions) Sales for the year ended December 31, Segment 2025 2024 Europe 28,793 29,952 Total 61,352 62,441"}
    assert _status("Total sales in 2025 were $61,352 million.", citation)["overall_status"] == "SUPPORTED"


def test_row_label_skips_dates_and_cell_runs():
    text = "Steel shipments for the year ended December 31, 2025 totaled 54.0 million tonnes"
    assert "shipments" in row_label(text, text.index("54.0"))
    table = "Crude steel production (thousand tonnes) 7,755 7,538 217"
    assert row_label(table, table.index("7,538")).startswith("Crude steel production")


# ---------------------------------------------------------------- Task 5: arithmetic


@pytest.mark.parametrize("sentence,ok", [
    ("Iron ore production increased by 6.4 million tonnes (15.1%), from 42.4 million tonnes in 2024 to 48.8 million tonnes in 2025.", True),
    ("Iron ore production increased by 26.5%, from 42.4 million tonnes in 2024 to 48.8 million tonnes in 2025.", False),
    ("Sales decreased by $1,089 million, from $62,441 million to $61,352 million.", True),
    ("Sales decreased by $2,000 million, from $62,441 million to $61,352 million.", False),
    ("Sales increased from $62,441 million to $61,352 million.", False),
    ("In 2025, crude steel production (55.6 million tonnes) was higher than shipments (54.0 million tonnes).", True),
    ("In 2025, crude steel production (55.6 million tonnes) was lower than shipments (54.0 million tonnes).", False),
    ("Sales were $61,352 million in 2025.", True),
])
def test_arithmetic_issue(sentence, ok):
    assert (arithmetic_issue(sentence, cv._quantities(sentence)) is None) == ok


def test_arithmetic_error_makes_supported_claim_unsupported():
    citation = {"source": SRC, "page": 95, "text": "Iron ore production (million tonnes) 48.8 42.4 6.4 15.1 %"}
    result = _status("Iron ore production increased by 26.5%, from 42.4 million tonnes in 2024 to 48.8 million tonnes in 2025.", citation)
    assert result["overall_status"] == "UNSUPPORTED" and is_contradiction(result["claims"][0]["explanation"])


# ---------------------------------------------------------------- Task 4: multi-hop


@pytest.mark.parametrize("question,expected", [
    ("Which reportable segment had the largest workforce at the end of 2025, and what were that segment's sales in 2025?",
     ["Which reportable segment had the largest workforce at the end of 2025", "what were that segment's sales in 2025"]),
    ("In 2025, was ArcelorMittal's crude steel production higher or lower than its steel shipments, and by how much?",
     ["ArcelorMittal's crude steel production 2025", "ArcelorMittal's steel shipments 2025"]),
])
def test_decompose_multi_fact_questions(question, expected):
    assert decompose(question) == expected


@pytest.mark.parametrize("question", [
    "What were ArcelorMittal's total sales in 2025?",
    "How did ArcelorMittal's operating income change between 2024 and 2025, and what was the main reason?",
    "Under the laws of which country was ArcelorMittal incorporated, and on what date?",
])
def test_single_fact_questions_are_not_split(question):
    assert decompose(question) == [question]


def test_merge_rankings_interleaves_and_deduplicates_pages():
    a = [{"source": SRC, "page": 1, "x": 1}, {"source": SRC, "page": 2}]
    b = [{"source": SRC, "page": 1}, {"source": SRC, "page": 3}, {"source": SRC, "page": 4}]
    merged = merge_rankings([a, b], limit=3, sub_queries=["qa", "qb"])
    assert [m["page"] for m in merged] == [1, 2, 3]
    assert merged[0]["x"] == 1 and merged[0]["sub_query"] == "qa" and merged[2]["sub_query"] == "qb"


# ---------------------------------------------------------------- pipeline


class FakeRetriever:
    def __init__(self, pages_by_query):
        self.pages_by_query, self.calls = pages_by_query, []

    def search(self, query, top_k=5, candidate_k=50):
        self.calls.append((query, top_k, candidate_k))
        return [{"source": SRC, "page": p, "text": t, "score": 1.0} for p, t in self.pages_by_query(query)]


class FakeReranker:
    def rerank(self, query, docs, top_k=5):
        return [{**d, "reranker_score": 1.0, "final_score": 1.0} for d in docs[:top_k]]


class FakeGenerator:
    def __init__(self, answer):
        self.answer, self.scope_aware, self.seen = answer, True, None

    def generate(self, question, chunks):
        self.seen = chunks
        return self.answer


TABLE = "(in $ millions) Sales for the year ended December 31, Segment 2025 2024 Europe 28,793 29,952 Total 61,352 62,441"
EMPLOYEES = "FTE employees by segment As of December 31, Segment 2025 2024 Europe 43,878 48,544 Total 125,554 125,416"


def test_pipeline_single_query_matches_legacy_flow():
    retriever = FakeRetriever(lambda q: [(95, TABLE), (7, "other text")])
    pipe = RAGPipeline(retriever, FakeReranker(), FakeGenerator("Total sales in 2025 were $61,352 million."), PipelineConfig.legacy())
    result = pipe.answer("What were total sales in 2025?")
    assert retriever.calls == [("What were total sales in 2025?", 20, 50)]
    assert result["validation"]["overall_status"] == "SUPPORTED" and result["sub_queries"] == ["What were total sales in 2025?"]


def test_pipeline_multihop_retrieves_each_fact_and_cites_both():
    def pages(query):
        return [(130, EMPLOYEES), (132, "workforce narrative")] if "workforce" in query else [(95, TABLE), (204, "notes")]

    retriever = FakeRetriever(pages)
    answer = "Europe had the largest workforce at the end of 2025 with 43,878 employees. Europe's sales in 2025 were $28,793 million."
    pipe = RAGPipeline(retriever, FakeReranker(), FakeGenerator(answer), PipelineConfig())
    result = pipe.answer("Which reportable segment had the largest workforce at the end of 2025, and what were that segment's sales in 2025?")
    assert len(retriever.calls) == 2
    assert [c["page"] for c in result["context"]][:2] == [130, 95]
    assert {130, 95} <= {c["page"] for c in result["citations"]}


def test_pipeline_abstains_on_verified_contradiction():
    retriever = FakeRetriever(lambda q: [(97, SEGMENT_CITATION["text"])])
    pipe = RAGPipeline(retriever, FakeReranker(), FakeGenerator("ArcelorMittal's iron ore production was 35.3 million tonnes in 2025."), PipelineConfig())
    result = pipe.answer("What was ArcelorMittal's iron ore production in 2025?")
    assert result["answer"] == ABSTENTION and result["citations"] == []
    assert result["validation"]["overall_status"] == "ABSTAINED"
    assert result["rejected"]["answer"].startswith("ArcelorMittal's iron ore")


class SequenceGenerator(FakeGenerator):
    def __init__(self, answers):
        super().__init__(answers[0])
        self.answers, self.questions = list(answers), []

    def generate(self, question, chunks):
        self.questions.append(question)
        return self.answers[min(len(self.questions), len(self.answers)) - 1]


def test_pipeline_revises_once_with_validator_feedback():
    table = "ArcelorMittal Iron ore production (million tonnes) 48.8 42.4 6.4 15.1 %"
    retriever = FakeRetriever(lambda q: [(95, table)])
    generator = SequenceGenerator([
        "Iron ore production increased by 26.5%, from 42.4 million tonnes in 2024 to 48.8 million tonnes in 2025.",
        "Iron ore production increased by 15.1%, from 42.4 million tonnes in 2024 to 48.8 million tonnes in 2025.",
    ])
    pipe = RAGPipeline(retriever, FakeReranker(), generator, PipelineConfig())
    result = pipe.answer("How did iron ore production change between 2024 and 2025?")
    assert len(generator.questions) == 2 and "Arithmetic error" in generator.questions[1]
    assert result["answer"].startswith("Iron ore production increased by 15.1%")
    assert result["validation"]["overall_status"] == "SUPPORTED"
    assert result["revision"]["original_answer"].startswith("Iron ore production increased by 26.5%")


def test_pipeline_no_revision_without_contradiction_and_legacy_never_revises():
    retriever = FakeRetriever(lambda q: [(95, TABLE)])
    generator = SequenceGenerator(["Total sales in 2025 were $61,352 million."])
    RAGPipeline(retriever, FakeReranker(), generator, PipelineConfig()).answer("What were total sales in 2025?")
    assert len(generator.questions) == 1
    legacy_gen = SequenceGenerator(["ArcelorMittal's iron ore production was 35.3 million tonnes in 2025."])
    result = RAGPipeline(FakeRetriever(lambda q: [(97, SEGMENT_CITATION["text"])]), FakeReranker(), legacy_gen,
                         PipelineConfig.legacy()).answer("What was ArcelorMittal's iron ore production in 2025?")
    assert len(legacy_gen.questions) == 1 and result["revision"] is None and result["rejected"] is None


def test_pipeline_keeps_answer_when_value_merely_not_found():
    retriever = FakeRetriever(lambda q: [(1, "Unrelated narrative about steel.")])
    pipe = RAGPipeline(retriever, FakeReranker(), FakeGenerator("Cost of sales in 2025 was $56,976 million."), PipelineConfig())
    result = pipe.answer("What was the cost of sales in 2025?")
    assert result["answer"].startswith("Cost of sales") and result["rejected"] is None
