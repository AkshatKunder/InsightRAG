from pathlib import Path
from pypdf import PdfReader


def load_pdf(pdf_path: str):
    """
    Extract text from a PDF page by page.
    """

    pdf_path = Path(pdf_path)

    reader = PdfReader(pdf_path)

    pages = []

    for page_number, page in enumerate(reader.pages, start=1):
        text = page.extract_text()

        if text and text.strip():
            pages.append({
                "source": pdf_path.name,
                "page": page_number,
                "text": text.strip()
            })

    return pages