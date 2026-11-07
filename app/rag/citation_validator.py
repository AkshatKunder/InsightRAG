import re


def extract_numbers(text):
    """
    Extract numbers and normalize comma-separated values.
    """
    pattern = r"\b\d[\d,]*(?:\.\d+)?%?\b"

    numbers = re.findall(pattern, text)

    normalized_numbers = set()

    for number in numbers:
        normalized = number.replace(",", "").replace("%", "")
        normalized_numbers.add(normalized)

        # Handle millions-to-billions conversion.
        try:
            value = float(normalized)

            if value >= 1000:
                normalized_billions = value / 1000
                normalized_numbers.add(
                    f"{normalized_billions:g}"
                )

        except ValueError:
            continue

    return normalized_numbers


def validate_citation(answer, citations):
    """
    Basic evidence validation.

    Checks whether numerical values in the answer
    can be found in the selected evidence.
    """

    if not citations:
        return {
            "is_valid": False,
            "reason": "No citations were selected."
        }

    answer_numbers = extract_numbers(answer)

    evidence_text = " ".join(
        citation.get("text", "")
        for citation in citations
    )

    evidence_numbers = extract_numbers(evidence_text)

    # Reject answers that contain no verifiable numerical claims.
    if not answer_numbers:
        return {
            "is_valid": False,
            "reason": "The answer contains no numerical claims to validate.",
            "unsupported_numbers": []
        }

    unsupported_numbers = answer_numbers - evidence_numbers

    if unsupported_numbers:
        return {
            "is_valid": False,
            "reason": "Some numerical claims are not present in the selected evidence.",
            "unsupported_numbers": sorted(unsupported_numbers)
        }

    return {
        "is_valid": True,
        "reason": "All numerical values in the answer were found in the selected evidence.",
        "unsupported_numbers": []
    }
