"""Phase 3.1 regression tests: Q018 retrieval rewrite, Q022 calculation, Q001/Q002 completeness (no models)."""

import pytest

from app.rag.calculation import apply, calculate, find_operand, plan
from app.rag.completeness import check, feedback
from app.rag.pipeline import PipelineConfig, RAGPipeline
from app.rag.query_rewriter import consolidated_first, rewrite_queries

SRC = "report.pdf"
SEGMENTS = ["North America", "Brazil", "Europe", "India and joint ventures", "Sustainable Solutions", "Mining"]

CONSOLIDATED_TABLE = {
    "source": SRC, "page": 95, "scope": "consolidated", "sections": [],
    "text": "Performance for the year ended December 31, 2025 2024 change % change Steel shipments (million tonnes) "
            "54.0 54.3 (0.3) (0.6) % Iron ore production (million tonnes) 48.8 42.4 6.4 15.1 % Average steel selling price",
}
MINING_TABLE = {
    "source": SRC, "page": 97, "scope": "segment:Mining", "sections": [],
    "text": "Mining Performance for the year ended December 31, 2025 2024 change % change Iron ore production "
            "(million tonnes) 35.3 27.9 7.4 26.5 %",
}
CHART_PAGE = {
    "source": SRC, "page": 3, "scope": "consolidated", "sections": [],
    "chart": ["Crude steel production (million tonnes): 2025 = 55.6, 2024 = 57.9",
              "Steel shipments (million tonnes): 2025 = 54.0, 2024 = 54.3"],
    "text": "ArcelorMittal overview. 55.6 54.0 57.9 54.3 2025 2024 Crude steel production (million tonnes) Steel shipments (million tonnes)",
}
Q018 = "How did ArcelorMittal's iron ore production change between 2024 and 2025?"
Q022 = "In 2025, was ArcelorMittal's crude steel production higher or lower than its steel shipments, and by how much?"


# ---------------------------------------------------------------- Task 1: consolidated rewrite (Q018)


def test_change_question_gets_consolidated_table_rewrite():
    assert rewrite_queries(Q018, SEGMENTS) == [{
        "query": "iron ore production 2025 2024 change % change", "kind": "change", "consolidated_first": True}]


@pytest.mark.parametrize("question", [
    "How did the Mining segment's iron ore production change between 2024 and 2025?",
    "How did Europe's sales change between 2024 and 2025?",
])
def test_segment_questions_keep_their_own_retrieval(question):
    assert rewrite_queries(question, SEGMENTS) == []


def test_unrelated_questions_are_not_rewritten():
    assert rewrite_queries("What were ArcelorMittal's total sales in 2025?", SEGMENTS) == []


def test_consolidated_first_is_a_stable_scope_reorder():
    ranked = [MINING_TABLE, CONSOLIDATED_TABLE, {"page": 1, "text": "x"}]
    assert [r["page"] for r in consolidated_first(ranked)] == [95, 1, 97]


# ---------------------------------------------------------------- Task 2: calculation (Q022, Q018)


def test_q022_difference_computed_from_chart_values():
    result = calculate(Q022, [CHART_PAGE])
    assert result["difference"] == 1.6 and result["direction"] == "higher"
    assert [o["raw"] for o in result["operands"]] == ["55.6", "54.0"]
    assert "a difference of 1.6 million tonnes" in result["sentence"]


def test_q022_wrong_llm_arithmetic_is_replaced():
    wrong = "Crude steel production (2025) was higher than steel shipments (2025) by 0.6 million tonnes."
    answer, citations = apply(wrong, {"claims": []}, [], calculate(Q022, [CHART_PAGE]))
    assert "0.6" not in answer and "1.6 million tonnes" in answer
    assert [c["page"] for c in citations] == [3]  # operand passage added to the citations


def test_q018_change_uses_consolidated_row_not_segment_table():
    result = calculate(Q018, [MINING_TABLE, CONSOLIDATED_TABLE])
    assert [o["raw"] for o in result["operands"]] == ["42.4", "48.8"] and result["percent"] == 15.1
    assert result["sentence"].startswith("Iron ore production increased by 6.4 million tonnes (15.1%)")


def test_segment_figures_used_when_the_question_names_the_segment():
    question = "How did the Mining segment's iron ore production change between 2024 and 2025?"
    result = calculate(question, [MINING_TABLE])
    assert result is not None and [o["raw"] for o in result["operands"]] == ["27.9", "35.3"]


def test_no_calculation_when_operands_are_missing_or_mismatched():
    assert calculate(Q022, [CONSOLIDATED_TABLE]) is None  # no crude steel figure in the evidence
    other_unit = {"source": SRC, "page": 9, "text": "2025 2024 Steel shipments (thousand tonnes) 54,000 54,300"}
    chart_only_production = {**CHART_PAGE, "chart": [CHART_PAGE["chart"][0]], "text": "55.6"}
    assert calculate(Q022, [chart_only_production, other_unit]) is None  # units differ
    assert plan("What were total sales in 2025?") is None


def test_operand_label_must_not_name_another_measure():
    capacity = {"source": SRC, "page": 60, "text": "2025 2024 Crude steel production capacity (million tonnes) 74.6 74.0"}
    assert find_operand("crude steel production", "2025", [capacity], Q022) is None


def test_contradicted_sentences_are_dropped_and_explanations_kept():
    validation = {"claims": [{"claim_text": "Iron ore production increased by 26.5% to 35.3 million tonnes.", "status": "UNSUPPORTED",
                              "explanation": "Scope/metric mismatch: the passage figure reports the Mining segment/unit."}]}
    answer, _ = apply("Iron ore production increased by 26.5% to 35.3 million tonnes. Mining is a segment.",
                      validation, [], calculate(Q018, [CONSOLIDATED_TABLE]))
    assert "35.3" not in answer and "Mining is a segment." in answer and "48.8 million tonnes in 2025" in answer


def test_other_uses_of_by_are_kept():
    keep = "The main reason was a $1.9 billion gain, partly offset by a $0.4 billion expense."
    answer, _ = apply(keep, {"claims": []}, [], calculate(Q018, [CONSOLIDATED_TABLE]))
    assert answer.startswith(keep)


# ---------------------------------------------------------------- Task 3: completeness (Q001, Q002)

SEGMENT_LIST = {"source": SRC, "page": 94, "text": "ArcelorMittal reports its operations in six reportable segments: North America, "
                "Brazil, Europe, India and JVs, Sustainable Solutions and Mining, with the remainder of its operations in Others."}
DEFINITION = {"source": SRC, "page": 3, "text": "Company overview ArcelorMittal is one of the world's leading integrated steel and mining companies."}
Q002 = "What are ArcelorMittal's reportable segments?"
Q001 = "What is ArcelorMittal's primary business?"


@pytest.mark.parametrize("answer,status,missing,extra", [
    ("ArcelorMittal has six reportable segments: North America, Brazil, Europe, India and JVs, Sustainable Solutions and Mining.", "complete", [], []),
    ("Segments: 1. North America 2. Brazil 3. Europe 4. India and joint ventures 5. Sustainable Solutions 6. Mining", "complete", [], []),
    ("Segments: 1. North America 2. Brazil 3. Europe 4. India and JVs 5. Sustainable Solutions and Mining 6. Others.", "incomplete", [], ["Others"]),
    ("North America, Brazil and Europe.", "incomplete", ["India and JVs", "Sustainable Solutions", "Mining"], []),
])
def test_q002_list_completeness(answer, status, missing, extra):
    result = check(Q002, answer, [SEGMENT_LIST])
    assert (result["status"], result["missing"], result["extra"]) == (status, missing, extra)
    assert result["count"] == 6


@pytest.mark.parametrize("answer,status", [
    ("ArcelorMittal's primary business is steel production.", "incomplete"),
    ("ArcelorMittal is a steel and mining company.", "complete"),  # concise, without "integrated"
    ("It is one of the world's leading integrated steel and mining companies.", "complete"),
])
def test_q001_definition_completeness(answer, status):
    result = check(Q001, answer, [DEFINITION])
    assert result["status"] == status and result["terms"] == ["steel", "mining"]


def test_completeness_not_applicable_without_evidence_or_pattern():
    assert check(Q002, "anything", [DEFINITION]) is None
    assert check("What were total sales in 2025?", "Sales were $61,352 million.", [SEGMENT_LIST]) is None


def test_list_with_inconsistent_count_is_not_trusted():
    odd = {"source": SRC, "page": 1, "text": "The Company has four reportable segments: A Corp, B Corp."}
    assert check("What are the Company's reportable segments?", "A Corp.", [odd]) is None


def test_feedback_names_missing_and_extra_items():
    text = feedback(check(Q002, "1. North America 2. Others", [SEGMENT_LIST]))
    assert "6 reportable segments" in text and "Others" in text and "Mining" in text


# ---------------------------------------------------------------- pipeline integration and backward compatibility


class Retriever:
    def __init__(self, by_query):
        self.by_query, self.calls = by_query, []

    def search(self, query, top_k=5, candidate_k=50):
        self.calls.append(query)
        return [dict(c) for c in self.by_query(query)]


class Reranker:
    def rerank(self, query, docs, top_k=5):
        return docs[:top_k]


class Generator:
    def __init__(self, answers):
        self.answers, self.questions, self.scope_aware = list(answers), [], True

    def generate(self, question, chunks):
        self.questions.append(question)
        return self.answers[min(len(self.questions), len(self.answers)) - 1]


def test_pipeline_q018_retrieves_consolidated_table_and_computes_change():
    retriever = Retriever(lambda q: [CONSOLIDATED_TABLE, MINING_TABLE] if "% change" in q else [MINING_TABLE])
    generator = Generator(["Iron ore production increased by 26.5%, from 27.9 million tonnes in 2024 to 35.3 million tonnes in 2025.",
                           "I could not find this information in the provided document."])
    result = RAGPipeline(retriever, Reranker(), generator, PipelineConfig()).answer(Q018)
    assert retriever.calls[1] == "iron ore production 2025 2024 change % change"
    assert 95 in [c["page"] for c in result["context"]] and 95 in [c["page"] for c in result["citations"]]
    assert result["answer"] == "Iron ore production increased by 6.4 million tonnes (15.1%), from 42.4 million tonnes in 2024 to 48.8 million tonnes in 2025."
    assert result["validation"]["overall_status"] == "SUPPORTED"


def test_pipeline_q002_incomplete_answer_is_revised_and_reported():
    retriever = Retriever(lambda q: [SEGMENT_LIST])
    generator = Generator(["Segments: 1. North America 2. Brazil 3. Europe 4. India and JVs 5. Sustainable Solutions and Mining 6. Others.",
                           "ArcelorMittal has six reportable segments: North America, Brazil, Europe, India and JVs, Sustainable Solutions and Mining."])
    result = RAGPipeline(retriever, Reranker(), generator, PipelineConfig()).answer(Q002)
    assert len(generator.questions) == 2 and "Others" in generator.questions[1]
    assert result["validation"]["completeness"]["status"] == "complete"


def test_phase3_and_legacy_configs_skip_phase3_1_steps():
    for config in (PipelineConfig.phase3(), PipelineConfig.legacy()):
        retriever = Retriever(lambda q: [SEGMENT_LIST])
        generator = Generator(["1. North America 2. Others"])
        result = RAGPipeline(retriever, Reranker(), generator, config).answer(Q002)
        assert len(retriever.calls) == 1 and len(generator.questions) == 1
        assert "completeness" not in result["validation"] and result["calculation"] is None
