from app.rag.loader import load_pdf
from app.rag.chunker import chunk_pages
from app.rag.embeddings import EmbeddingModel


pdf_path = "data/documents/annual_report.pdf"

# Load PDF
pages = load_pdf(pdf_path)

# Create chunks
chunks = chunk_pages(pages)

# Load embedding model
embedding_model = EmbeddingModel()

# Take first 5 chunks for testing
texts = [chunk["text"] for chunk in chunks[:5]]

# Generate embeddings
embeddings = embedding_model.embed_documents(texts)

print("\nEmbedding test successful!")
print(f"Number of embeddings: {len(embeddings)}")
print(f"Embedding dimension: {embeddings.shape[1]}")
print(f"First embedding preview: {embeddings[0][:10]}")
