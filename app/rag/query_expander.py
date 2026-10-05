def expand_query(query):

    query_lower = query.lower()

    expansions = []

    if "sales" in query_lower or "revenue" in query_lower:
        expansions.extend([
            "sales",
            "revenue",
            "annual sales",
            "sales for the year",
            "2025",
            "2024",
            "billion USD",
            "financial results"
        ])

    if "employee" in query_lower or "employees" in query_lower:
        expansions.extend([
            "employees",
            "FTE",
            "full time equivalent",
            "2025",
            "workforce"
        ])

    if "country" in query_lower or "countries" in query_lower:
        expansions.extend([
            "countries",
            "steel-making operations",
            "operations",
            "2025"
        ])

    if not expansions:
        return query

    return query + " " + " ".join(expansions)
