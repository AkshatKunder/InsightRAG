from app.rag.embeddings import EmbeddingModel
from app.rag.vector_store import VectorStore


class Retriever:

    def __init__(self):
        self.embedding_model = EmbeddingModel()
        self.vector_store = VectorStore()

    def search(self, query, top_k=5):
        """
        Search Qdrant for chunks semantically related to the query.
        """

        # Convert question into an embedding
        query_embedding = self.embedding_model.embed_query(query)

        # Search Qdrant
        results = self.vector_store.client.query_points(
            collection_name="insightrag_documents",
            query=query_embedding.tolist(),
            limit=top_k,
            with_payload=True
        ).points

        retrieved_chunks = []

        for result in results:
            retrieved_chunks.append({
                "text": result.payload["text"],
                "source": result.payload["source"],
                "page": result.payload["page"],
                "score": result.score
            })

        return retrieved_chunks
