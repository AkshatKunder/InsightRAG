from app.rag.citation_validator import validate_citation


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
