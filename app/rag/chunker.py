def chunk_pages(pages, chunk_size=400, overlap=100):

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
