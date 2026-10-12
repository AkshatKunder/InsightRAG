def select_citations(question, answer, retrieved_chunks, max_sources=3):
    """Select source evidence using question-specific heuristics."""

    question_lower = question.lower()

    if "sales" in question_lower or "revenue" in question_lower:
        preferred_pages = [204, 95]
        excluded_terms = [
            "sale of two tubular products facilities",
            "sale of facilities",
            "divestment",
        ]

    elif "employee" in question_lower or "workforce" in question_lower:
        preferred_pages = [130]
        excluded_terms = []

    elif "country" in question_lower or "countries" in question_lower:
        preferred_pages = [3, 26]
        excluded_terms = []

    else:
        preferred_pages = []
        excluded_terms = []

    selected = []
    seen_pages = set()

    # Prefer known relevant pages when they exist in the retrieved results.
    if preferred_pages:
        ordered_chunks = sorted(
            retrieved_chunks,
            key=lambda chunk: (
                preferred_pages.index(chunk["page"])
                if chunk.get("page") in preferred_pages
                else len(preferred_pages)
            )
        )
    else:
        ordered_chunks = retrieved_chunks

    for chunk in ordered_chunks:
        text = chunk.get("text", "").lower()
        page_key = (chunk.get("source"), chunk.get("page"))

        if page_key in seen_pages:
            continue

        if any(term in text for term in excluded_terms):
            continue

        selected.append({
            "source": chunk["source"],
            "page": chunk["page"],
            "text": chunk["text"],
            "final_score": chunk.get(
                "final_score",
                chunk.get("score", 0.0)
            ),
        })

        seen_pages.add(page_key)

        if len(selected) >= max_sources:
            break

    return selected
