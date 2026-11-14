"""Phase 1 validator regressions.

Covers prose counts near financial tables, counted nouns, rounding-aware
matching, incorrect numerical answers and abstention.
"""

import pytest

from app.rag.citation_validator import is_abstention, validate_citation

# Shape of the report's company-overview page: a KPI caption with
# "(billion USD)" directly precedes a prose count.
OVERVIEW_TEXT = (
    "55.6 48.8 54.0 61.4 57.9 42.4 54.3 62.4 2025 2024 Crude steel production "
    "(million tonnes) Iron ore production (million tonnes) Steel shipments "
    "(million tonnes) Sales (billion USD) ArcelorMittal has steel-making "
    "operations in 14 countries, including 34 integrated and mini-mill "
    "steel-making facilities. As of December 31, 2025, ArcelorMittal had "
    "approximately 125,554 employees."
)

# Shape of the report's segment sales table.
SALES_TABLE_TEXT = (
    "ArcelorMittal segment results for the years ended "
    "December 31, 2025, and 2024: (in $ millions) Sales for the year ended "
    "December 31, Segment 2025 2024 change % change North America 12,335 11,896 "
    "439 3.7 % Europe 28,793 29,952 (1,159) (3.9) % Total 61,352 62,441 (1,089) (1.7) %"
)


def test_prose_count_after_financial_caption_does_not_inherit_currency():
    result = validate_citation(
        "ArcelorMittal has steel-making operations in 14 countries.",
        [{"text": OVERVIEW_TEXT, "page": 3}],
    )
    assert result["overall_status"] == "SUPPORTED"


def test_count_of_a_different_noun_does_not_support_country_count():
    result = validate_citation(
        "ArcelorMittal has steel-making operations in 14 countries.",
        [{"text": "The Company operates 14 research sites in 9 countries around the world, supporting operations."}],
    )
    assert result["is_valid"] is False


def test_table_total_still_inherits_usd_millions():
    result = validate_citation(
        "ArcelorMittal's total sales in 2025 were $61,352 million.",
        [{"text": SALES_TABLE_TEXT}],
    )
    assert result["overall_status"] == "SUPPORTED"


def test_rounded_billions_match_table_millions():
    result = validate_citation(
        "Total sales in 2025 were $61.4 billion.",
        [{"text": SALES_TABLE_TEXT}],
    )
    assert result["overall_status"] == "SUPPORTED"


@pytest.mark.parametrize("answer, evidence", [
    # Wrong value with the correct unit.
    ("ArcelorMittal's total sales in 2025 were $60,500 million.", SALES_TABLE_TEXT),
    ("ArcelorMittal's total sales in 2025 were $62,441 million.", "Total sales in 2025 were $61,352 million."),
    # Right digits, wrong scale.
    ("ArcelorMittal's total sales in 2025 were $61,352 billion.", SALES_TABLE_TEXT),
    ("ArcelorMittal's total sales in 2025 were $61.352 million.", SALES_TABLE_TEXT),
    # Wrong currency.
    ("ArcelorMittal's total sales in 2025 were €61,352 million.", SALES_TABLE_TEXT),
    # Segment value presented as the total.
    ("ArcelorMittal's total sales in 2025 were $28,793 million.", "Total sales in 2025 were $61,352 million."),
    # Off-by-one and rounded-but-wrong counts.
    ("ArcelorMittal has steel-making operations in 15 countries.", OVERVIEW_TEXT),
    ("ArcelorMittal had 125,000 employees in 2025.", OVERVIEW_TEXT),
    ("ArcelorMittal had 126,554 employees in 2025.", OVERVIEW_TEXT),
    # A number that appears only as a count of something else (34 facilities).
    ("ArcelorMittal has steel-making operations in 34 countries.", OVERVIEW_TEXT),
])
def test_incorrect_numerical_answers_are_not_supported(answer, evidence):
    result = validate_citation(answer, [{"text": evidence}])
    assert result["is_valid"] is False
    assert result["overall_status"] != "SUPPORTED"


def test_rounding_is_limited_to_the_written_precision():
    # 125,600 is 125,554 rounded to the nearest hundred, so it is acceptable.
    assert validate_citation("ArcelorMittal had 125,600 employees.", [{"text": OVERVIEW_TEXT}])["is_valid"] is True
    # A small, exact-looking number may not absorb a large error.
    assert validate_citation("The company had 100 employees.", [{"text": "The company had 149 employees."}])["is_valid"] is False


@pytest.mark.parametrize("answer", [
    "I could not find this information in the provided document.",
    "The answer cannot be determined from the provided context.",
    "This information is not available in the provided document.",
])
def test_abstention_is_reported_separately(answer):
    result = validate_citation(answer, [{"text": OVERVIEW_TEXT}])
    assert is_abstention(answer) is True
    assert result["overall_status"] == "ABSTAINED"
    assert result["is_valid"] is False
    assert result["claims"] == []


def test_hedged_answer_that_states_a_number_is_still_validated():
    answer = "I could not find the exact figure, but sales were $70,000 million."
    assert is_abstention(answer) is False
    assert validate_citation(answer, [{"text": SALES_TABLE_TEXT}])["overall_status"] == "UNSUPPORTED"
