from sentence_transformers import SentenceTransformer


MODEL_NAME = "BAAI/bge-small-en-v1.5"


class EmbeddingModel:
    def __init__(self):
        print(f"Loading embedding model: {MODEL_NAME}")
        self.model = SentenceTransformer(MODEL_NAME)

    def embed_documents(self, texts):
        """
        Convert multiple documents into embeddings.
        """
        return self.model.encode(
            texts,
            normalize_embeddings=True,
            show_progress_bar=True
        )

    def embed_query(self, query):
        """
        Convert a user query into an embedding.
        """
        return self.model.encode(
            query,
            normalize_embeddings=True
        )
