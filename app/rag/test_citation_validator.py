from app.rag.citation_validator import extract_claims, validate_citation


def test_supported_number():
    answer = "ArcelorMittal employed 125,554 employees."

    citations = [
        {
            "text": "As of December 31, 2025, ArcelorMittal employed approximately 125,554 full-time equivalent employees."
        }
    ]

    result = validate_citation(answer, citations)

    assert result["is_valid"] is True


def test_unsupported_number():
    answer = "ArcelorMittal employed 150,000 employees."

    citations = [
        {
            "text": "ArcelorMittal employed approximately 125,554 employees."
        }
    ]

    result = validate_citation(answer, citations)

    assert result["is_valid"] is False


def test_missing_citations():
    answer = "ArcelorMittal generated sales of $61,352 million."

    result = validate_citation(answer, [])

    assert result["is_valid"] is False


def test_millions_to_billions_conversion():
    answer = "ArcelorMittal's sales were $61.352 billion."

    citations = [
        {
            "text": "Total sales were 61,352 million dollars."
        }
    ]

    result = validate_citation(answer, citations)

    assert result["is_valid"] is True


def test_answer_without_numbers():
    answer = "I could not find this information in the provided document."

    citations = [
        {
            "text": "ArcelorMittal employed approximately 125,554 employees."
        }
    ]

    result = validate_citation(answer, citations)

    assert result["is_valid"] is False


def test_rounded_numerical_equivalence():
    result = validate_citation("Total sales were $61.4 billion.", [
        {"text": "Total sales were $61,352 million in 2025."}
    ])
    assert result["is_valid"] is True


def test_correct_metric_and_reporting_period():
    result = validate_citation("ArcelorMittal employed 125,554 FTE employees in 2025.", [
        {"text": "In 2025, ArcelorMittal employed approximately 125,554 full-time equivalent employees.", "source": "report.pdf", "page": 10}
    ])
    assert result["overall_status"] == "SUPPORTED"
    assert result["claims"][0]["page"] == 10


def test_wrong_reporting_period_is_unsupported():
    result = validate_citation("ArcelorMittal employed 125,554 employees in 2025.", [
        {"text": "In 2024, ArcelorMittal employed 125,554 employees."}
    ])
    assert result["claims"][0]["status"] == "UNSUPPORTED"


def test_same_number_with_wrong_metric_is_not_support():
    result = validate_citation("ArcelorMittal employed 125,554 employees.", [
        {"text": "Board remuneration was 125,554 dollars."}
    ])
    assert result["is_valid"] is False
    assert result["claims"][0]["status"] == "INSUFFICIENT_EVIDENCE"


def test_facility_sale_does_not_support_total_sales():
    result = validate_citation("Total company sales were $61.352 billion in 2025.", [
        {"text": "The sale of a facility generated $61,352 million in 2025."}
    ])
    assert result["is_valid"] is False


def test_board_remuneration_does_not_support_employee_count():
    result = validate_citation("The company employed 200 employees.", [
        {"text": "Board remuneration was paid to 200 directors."}
    ])
    assert result["is_valid"] is False


def test_country_count_requires_explicit_count_evidence():
    result = validate_citation("Steel-making operations were present in 14 countries.", [
        {"text": "Steel-making operations are located in France, Spain, and Brazil."}
    ])
    assert result["claims"][0]["status"] == "UNSUPPORTED"


def test_non_numerical_factual_claim():
    result = validate_citation("The company has steel-making operations in Europe.", [
        {"text": "The company has steel-making operations in Europe."}
    ])
    assert result["is_valid"] is True


def test_insufficient_evidence_is_distinct_from_conflict():
    result = validate_citation("The company opened a new research center.", [
        {"text": "The company published its annual report."}
    ])
    assert result["overall_status"] == "INSUFFICIENT_EVIDENCE"


def test_multiple_claims_with_only_some_supported():
    result = validate_citation("The company employed 100 employees. Total sales were $2 billion.", [
        {"text": "The company employed 100 employees."},
        {"text": "Total sales were $1 billion."},
    ])
    assert result["is_valid"] is False
    assert len(result["supported_claims"]) == 1
    assert len(result["unsupported_claims"]) == 1


def test_different_passages_can_support_different_claims():
    result = validate_citation("The company employed 100 employees. Total sales were $2 billion.", [
        {"text": "The company employed 100 employees.", "source": "workforce.pdf", "page": 2},
        {"text": "Total sales were $2,000 million.", "source": "financials.pdf", "page": 8},
    ])
    assert result["is_valid"] is True
    assert {claim["source"] for claim in result["claims"]} == {"workforce.pdf", "financials.pdf"}


def test_unrelated_passages_are_not_combined_to_manufacture_support():
    result = validate_citation("Total sales were $2 billion in 2025.", [
        {"text": "Total sales were reported in 2025."},
        {"text": "A facility sale generated $2 billion."},
    ])
    assert result["is_valid"] is False


def test_claim_extraction_includes_non_numeric_claims():
    claims = extract_claims("Steel-making operations were present in Europe.")
    assert claims[0]["value"] is None
    assert "operation" in claims[0]["metric"]
