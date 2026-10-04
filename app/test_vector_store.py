from app.rag.loader import load_pdf
from app.rag.chunker import chunk_pages
from app.rag.embeddings import EmbeddingModel
from app.rag.vector_store import VectorStore


PDF_PATH = "data/documents/annual_report.pdf"


pages = load_pdf(PDF_PATH)

print(f"Loaded {len(pages)} pages.")


chunks = chunk_pages(
    pages,
    chunk_size=400,
    overlap=100
)

print(f"Created {len(chunks)} chunks.")


embedding_model = EmbeddingModel()

texts = [chunk["text"] for chunk in chunks]

embeddings = embedding_model.embed_documents(texts)

vector_store = VectorStore()

if vector_store.client.collection_exists("insightrag_documents"):
    vector_store.client.delete_collection("insightrag_documents")
    print("Deleted old Qdrant collection.")


vector_store.create_collection()


vector_store.add_documents(
    chunks,
    embeddings
)


print(
    f"Qdrant now contains "
    f"{vector_store.count_documents()} vectors."
)


vector_store.close()
