"""Deterministic unit tests for general-purpose evidence selection."""

from app.rag.citation_selector import EvidenceScoreConfig, select_citations


def chunk(page, text, score=0.8, source="report.pdf", **metadata):
    return {
        "source": source,
        "page": page,
        "text": text,
        "final_score": score,
        **metadata,
    }


def test_direct_evidence_outranks_unrelated_evidence():
    citations = select_citations(
        "What were total sales in 2025?",
        "Total sales were 61,352 million in 2025.",
        [
            chunk(10, "Total sales in 2025 were 61,352 million dollars.", 0.8),
            chunk(11, "The office cafeteria was renovated in 2025.", 0.8),
        ],
        min_relevance_threshold=0.0,
    )

    assert citations[0]["page"] == 10
    assert citations[0]["final_score"] > citations[1]["final_score"]


def test_facility_sale_is_not_evidence_for_total_sales():
    citations = select_citations(
        "What were the company's total sales in 2025?",
        "Total sales were 61,352 million in 2025.",
        [
            chunk(10, "Total sales in 2025 were 61,352 million dollars.", 0.9),
            chunk(11, "The company completed the sale of a manufacturing facility in 2025.", 0.9),
        ],
    )

    assert [item["page"] for item in citations] == [10]


def test_remuneration_is_not_evidence_for_employee_count():
    citations = select_citations(
        "How many employees did the company have in 2025?",
        "The company employed 125,554 full-time equivalent employees in 2025.",
        [
            chunk(20, "In 2025 the company employed 125,554 full-time equivalent employees.", 0.9),
            chunk(21, "Board remuneration for directors was 12 million dollars in 2025.", 0.9),
        ],
    )

    assert [item["page"] for item in citations] == [20]


def test_country_operations_evidence_beats_product_description():
    citations = select_citations(
        "In how many countries does the company have steel-making operations?",
        "The company has steel-making operations in 14 countries.",
        [
            chunk(30, "The company has steel-making operations in 14 countries.", 0.8),
            chunk(31, "The company sells steel products to automotive customers in global markets.", 0.8),
        ],
    )

    assert [item["page"] for item in citations] == [30]


def test_selection_has_no_page_preferences():
    citations = select_citations(
        "What was total revenue in 2025?",
        "Total revenue was 50,000 million in 2025.",
        [chunk(999, "Total revenue was 50,000 million in 2025.", 0.9)],
    )

    assert citations[0]["page"] == 999


def test_selector_does_not_force_max_sources():
    citations = select_citations(
        "What were sales?",
        "Sales were 100 million.",
        [
            chunk(1, "Sales were 100 million.", 0.9),
            chunk(2, "The office has a kitchen.", 0.2),
        ],
        max_sources=3,
    )

    assert len(citations) == 1


def test_selector_returns_empty_when_no_candidate_meets_threshold():
    citations = select_citations(
        "What were total sales?",
        "Sales were 100 million.",
        [chunk(1, "Coffee is served every Tuesday.", 0.1)],
    )

    assert citations == []


def test_duplicate_chunks_are_not_repeated():
    citations = select_citations(
        "What were sales?",
        "Sales were 100 million.",
        [
            chunk(1, "Sales were 100 million.", 0.9),
            chunk(1, "Sales were 100 million.", 0.8),
        ],
    )

    assert len(citations) == 1
    assert citations[0]["page"] == 1


def test_different_sources_with_same_page_are_distinct():
    citations = select_citations(
        "What were sales in 2025?",
        "Sales were 100 million in 2025.",
        [
            chunk(1, "Sales were 100 million in 2025.", source="a.pdf"),
            chunk(1, "Sales were 100 million in 2025.", source="b.pdf"),
        ],
    )

    assert {item["source"] for item in citations} == {"a.pdf", "b.pdf"}


def test_identical_text_on_different_pages_of_same_source_is_distinct():
    citations = select_citations(
        "What were sales in 2025?",
        "Sales were 100 million in 2025.",
        [
            chunk(1, "Sales were 100 million in 2025."),
            chunk(2, "Sales were 100 million in 2025."),
        ],
    )

    assert {item["page"] for item in citations} == {1, 2}


def test_missing_optional_scores_do_not_raise_or_block_direct_evidence():
    citations = select_citations(
        "What were sales?",
        "Sales were 100 million.",
        [{"source": "report.pdf", "page": 1, "text": "Sales were 100 million."}],
    )

    assert citations[0]["page"] == 1


def test_million_and_billion_values_are_numerically_equivalent():
    citations = select_citations(
        "What were total sales?",
        "Total sales were 61.352 billion.",
        [chunk(1, "Total sales were 61,352 million dollars.", 0.8)],
    )

    assert citations[0]["evidence_signals"]["numerical_consistency"] == 1.0


def test_metadata_is_preserved_without_mutating_input():
    original = chunk(1, "Sales were 100 million.", custom_field="kept")
    citations = select_citations("What were sales?", "Sales were 100 million.", [original])

    assert citations[0]["custom_field"] == "kept"
    assert original["final_score"] == 0.8
    assert citations[0] is not original


def test_threshold_is_configurable():
    config = EvidenceScoreConfig(min_relevance_threshold=0.95)
    citations = select_citations(
        "What were sales?", "Sales were 100 million.",
        [chunk(1, "Sales were 100 million.", 0.5)], config=config,
    )

    assert citations == []
