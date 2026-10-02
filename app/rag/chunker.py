def chunk_pages(pages, chunk_size=700, overlap=100):
    """
    Split page-level text into overlapping chunks.

    Args:
        pages: List of dictionaries returned by load_pdf()
        chunk_size: Approximate number of words per chunk
        overlap: Number of overlapping words between chunks

    Returns:
        List of dictionaries containing chunk text and metadata.
    """

    chunks = []

    for page in pages:
        words = page["text"].split()

        start = 0

        while start < len(words):
            end = start + chunk_size

            chunk_words = words[start:end]

            if not chunk_words:
                break

            chunks.append({
                "text": " ".join(chunk_words),
                "source": page["source"],
                "page": page["page"],
            })

            start += chunk_size - overlap

    return chunks