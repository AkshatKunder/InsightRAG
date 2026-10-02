from app.rag.loader import load_pdf
from app.rag.chunker import chunk_pages
from app.rag.embeddings import EmbeddingModel
from app.rag.vector_store import VectorStore


PDF_PATH = "data/documents/annual_report.pdf"


# 1. Load PDF
print("Loading PDF...")

pages = load_pdf(PDF_PATH)

print(f"Pages loaded: {len(pages)}")


# 2. Create chunks
print("\nCreating chunks...")

chunks = chunk_pages(pages)

print(f"Chunks created: {len(chunks)}")


# 3. Load embedding model
print("\nLoading embedding model...")

embedding_model = EmbeddingModel()


# 4. Generate embeddings
print("\nGenerating embeddings...")

texts = [chunk["text"] for chunk in chunks]

embeddings = embedding_model.embed_documents(texts)

print(f"Embeddings generated: {len(embeddings)}")
print(f"Embedding dimension: {embeddings.shape[1]}")


# 5. Create Qdrant vector store
print("\nInitializing Qdrant...")

vector_store = VectorStore()

vector_store.create_collection()


# 6. Insert vectors
print("\nInserting vectors into Qdrant...")

vector_store.add_documents(
    chunks,
    embeddings
)


# 7. Verify
count = vector_store.count_documents()

print("\n" + "=" * 60)
print("QDRANT TEST SUCCESSFUL")
print("=" * 60)
print(f"Vectors stored: {count}")
