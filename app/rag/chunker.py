def chunk_pages(pages, chunk_size=400, overlap=100):

    if chunk_size <= 0:
        raise ValueError("chunk_size must be greater than zero")
    if overlap < 0 or overlap >= chunk_size:
        raise ValueError("overlap must be greater than or equal to zero and less than chunk_size")

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
