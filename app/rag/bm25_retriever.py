from rank_bm25 import BM25Okapi


class BM25Retriever:

    def __init__(self, chunks):
        """
        Initialize BM25 using document chunks.
        """

        self.chunks = chunks

        # Tokenize each chunk
        tokenized_chunks = [
            chunk["text"].lower().split()
            for chunk in chunks
        ]

        self.bm25 = BM25Okapi(tokenized_chunks)

    def search(self, query, top_k=5):
        """
        Search chunks using BM25 keyword matching.
        """

        query_tokens = query.lower().split()

        scores = self.bm25.get_scores(query_tokens)

        # Get indexes of highest-scoring chunks
        ranked_indexes = scores.argsort()[::-1][:top_k]

        results = []

        for index in ranked_indexes:

            results.append({
                "text": self.chunks[index]["text"],
                "source": self.chunks[index]["source"],
                "page": self.chunks[index]["page"],
                "score": float(scores[index])
            })

        return results
