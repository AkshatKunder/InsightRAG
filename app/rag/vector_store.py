from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct


COLLECTION_NAME = "insightrag_documents"
VECTOR_SIZE = 384


class VectorStore:
    def __init__(self, path="data/qdrant"):
        self.client = QdrantClient(path=path)

    def create_collection(self):
        """
        Create the Qdrant collection if it does not already exist.
        """

        collections = self.client.get_collections().collections

        collection_names = [collection.name for collection in collections]

        if COLLECTION_NAME not in collection_names:
            self.client.create_collection(
                collection_name=COLLECTION_NAME,
                vectors_config=VectorParams(
                    size=VECTOR_SIZE,
                    distance=Distance.COSINE
                )
            )

            print(f"Created collection: {COLLECTION_NAME}")

        else:
            print(f"Collection already exists: {COLLECTION_NAME}")

    def add_documents(self, chunks, embeddings):
        """
        Store document chunks, embeddings, and metadata in Qdrant.
        """

        points = []

        for index, (chunk, embedding) in enumerate(
            zip(chunks, embeddings)
        ):

            point = PointStruct(
                id=index,
                vector=embedding.tolist(),
                payload={
                    "text": chunk["text"],
                    "source": chunk["source"],
                    "page": chunk["page"]
                }
            )

            points.append(point)

        self.client.upsert(
            collection_name=COLLECTION_NAME,
            points=points
        )

        print(f"Inserted {len(points)} vectors into Qdrant.")

    def count_documents(self):
        """
        Return the number of vectors stored in the collection.
        """

        result = self.client.count(
            collection_name=COLLECTION_NAME,
            exact=True
        )

        return result.count

    def close(self):
        self.client.close()
