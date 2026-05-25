import fitz

def extract_text_from_pdf(pdf_path):
    document = fitz.open(pdf_path)

    pages = []

    for page_num in range(len(document)):
        page = document[page_num]
        text = page.get_text()

        text = text.replace("\n", " ")
        text = " ".join(text.split())

        pages.append({
            "page": page_num + 1,
            "text": text
        })

    return pages