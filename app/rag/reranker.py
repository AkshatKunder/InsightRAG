from sentence_transformers import CrossEncoder

MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"


class Reranker:
    def __init__(self):
        print(f"Loading reranker model: {MODEL_NAME}")
        self.model = CrossEncoder(MODEL_NAME)

    def rerank(self, query, documents, top_k=5):
        pairs = [[query, document["text"]] for document in documents]

        scores = self.model.predict(pairs)

        reranked = []

        min_score = min(scores)
        max_score = max(scores)

        for document, score in zip(documents, scores):
            result = document.copy()

            # Normalize reranker score to 0-1
            if max_score == min_score:
                normalized_reranker = 0.5
            else:
                normalized_reranker = (
                    float(score) - min_score
                ) / (max_score - min_score)

            # Normalize hybrid RRF score to 0-1
            rrf_scores = [doc["score"] for doc in documents]
            min_rrf = min(rrf_scores)
            max_rrf = max(rrf_scores)

            if max_rrf == min_rrf:
                normalized_rrf = 0.5
            else:
                normalized_rrf = (
                    result["score"] - min_rrf
                ) / (max_rrf - min_rrf)

            result["reranker_score"] = float(score)

            # Hybrid retrieval remains the stronger signal.
            result["final_score"] = (
                0.7 * normalized_rrf
                + 0.3 * normalized_reranker
            )

            reranked.append(result)

        reranked.sort(
            key=lambda x: x["final_score"],
            reverse=True
        )

        return reranked[:top_k]
