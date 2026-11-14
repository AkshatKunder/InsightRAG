"""Phase 1 selector regressions: localized context penalty and abstention."""

from app.rag.citation_selector import select_citations


def chunk(page, text, score=0.8, source="report.pdf"):
    return {"source": source, "page": page, "text": text, "final_score": score}


def test_mismatch_words_far_from_the_answer_number_do_not_penalize():
    # The divestment sentence is in the same chunk but far from the figure.
    text = (
        "Total sales for the year ended December 31, 2025 were 61,352 million dollars. "
        + "Shipments and prices are discussed in the following section. " * 6
        + "Cost of sales included impairment related to the divestment of operations."
    )
    citations = select_citations(
        "What were the company's total sales in 2025?",
        "Total sales in 2025 were $61,352 million.",
        [chunk(95, text, 0.9)],
    )

    assert [item["page"] for item in citations] == [95]
    assert citations[0]["evidence_signals"]["context_penalty"] == 1.0


def test_mismatch_words_next_to_the_answer_number_still_penalize():
    citations = select_citations(
        "What were the company's total sales in 2025?",
        "Total sales in 2025 were $500 million.",
        [chunk(11, "In 2025 the company completed the sale of a facility for 500 million dollars.", 0.9)],
    )

    assert citations == []


def test_years_are_not_used_as_anchors_for_the_penalty_window():
    # "2025" is in the answer and next to the divestment; it must not pull the
    # divestment wording into the window around the real figure.
    text = (
        "Total sales were 61,352 million dollars. "
        + "Shipments and prices are discussed in the following section. " * 6
        + "In 2025 the company recorded charges related to the divestment of operations."
    )
    citations = select_citations(
        "What were the company's total sales in 2025?",
        "Total sales in 2025 were $61,352 million.",
        [chunk(95, text, 0.9)],
    )

    assert citations and citations[0]["evidence_signals"]["context_penalty"] == 1.0


def test_country_count_beats_customer_text_elsewhere_in_chunk():
    text = (
        "The company has steel-making operations in 14 countries. "
        + "Production figures are presented below by segment and year. " * 6
        + "The company sells products to customers in 130 countries."
    )
    citations = select_citations(
        "In how many countries does the company have steel-making operations?",
        "The company has steel-making operations in 14 countries.",
        [chunk(3, text, 0.9)],
    )

    assert [item["page"] for item in citations] == [3]


def test_abstention_answer_selects_no_evidence():
    citations = select_citations(
        "What was revenue from cryptocurrency mining in 2025?",
        "I could not find this information in the provided document.",
        [chunk(1, "Total sales in 2025 were 61,352 million dollars.", 0.9)],
        min_relevance_threshold=0.0,
    )

    assert citations == []
