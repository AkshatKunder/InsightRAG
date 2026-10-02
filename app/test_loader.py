from app.rag.loader import load_pdf

pdf_path = "data/documents/annual_report.pdf"

pages = load_pdf(pdf_path)

print(f"Total pages extracted: {len(pages)}")

for page in pages[:3]:
    print("\n" + "=" * 60)
    print(f"Source: {page['source']}")
    print(f"Page: {page['page']}")
    print("=" * 60)
    print(page["text"][:1000])