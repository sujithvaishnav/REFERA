import fitz
import os

def extract_text_from_pdf(pdf_path: str):
    """
    Extracts text from each page of a PDF document using PyMuPDF (fitz).
    Ensures safe file handle closing and cleans up extra whitespace.
    """
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"PDF file not found at: {pdf_path}")

    pages = []
    with fitz.open(pdf_path) as document:
        for page_num in range(len(document)):
            page = document[page_num]
            text = page.get_text("text") or ""
            
            # Normalize whitespace
            cleaned_text = " ".join(text.split())
            
            if cleaned_text:
                pages.append({
                    "page": page_num + 1,
                    "text": cleaned_text
                })

    return pages