from app.rag.loader import load_pdf
from app.rag.chunker import chunk_pages


pdf_path = "data/documents/annual_report.pdf"

pages = load_pdf(pdf_path)

chunks = chunk_pages(pages)

print(f"Pages extracted: {len(pages)}")
print(f"Chunks created: {len(chunks)}")

for i, chunk in enumerate(chunks[:5], start=1):
    print("\n" + "=" * 70)
    print(f"Chunk: {i}")
    print(f"Source: {chunk['source']}")
    print(f"Page: {chunk['page']}")
    print("=" * 70)
    print(chunk["text"][:500])