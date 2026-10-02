from sentence_transformers import CrossEncoder


MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"


class Reranker:

    def __init__(self):
        print(f"Loading reranker model: {MODEL_NAME}")
        self.model = CrossEncoder(MODEL_NAME)

    def rerank(self, query, documents, top_k=5):

        pairs = []

        for document in documents:
            pairs.append(
                [
                    query,
                    document["text"]
                ]
            )

        scores = self.model.predict(pairs)

        reranked = []

        for document, score in zip(documents, scores):

            result = document.copy()

            result["reranker_score"] = float(score)

            reranked.append(result)

        reranked.sort(
            key=lambda x: x["reranker_score"],
            reverse=True
        )

        return reranked[:top_k]
