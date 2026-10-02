from app.rag.loader import load_pdf
from app.rag.chunker import chunk_pages
from app.rag.bm25_retriever import BM25Retriever


PDF_PATH = "data/documents/annual_report.pdf"


print("Loading PDF...")

pages = load_pdf(PDF_PATH)

print(f"Pages loaded: {len(pages)}")


print("\nCreating chunks...")

chunks = chunk_pages(pages)

print(f"Chunks created: {len(chunks)}")


print("\nBuilding BM25 index...")

retriever = BM25Retriever(chunks)

print("BM25 index created.")


query = "How many employees did ArcelorMittal have in 2025?"

print(f"\nQuery: {query}")

results = retriever.search(query, top_k=5)


print("\n" + "=" * 70)
print("BM25 RESULTS")
print("=" * 70)


for i, result in enumerate(results, start=1):

    print(f"\nResult {i}")
    print(f"Score: {result['score']:.4f}")
    print(f"Source: {result['source']}")
    print(f"Page: {result['page']}")

    print("-" * 70)
    print(result["text"][:700])
