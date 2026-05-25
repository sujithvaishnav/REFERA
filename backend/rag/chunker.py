def chunk_text(pages, chunk_size=1200, overlap=200):

    chunks = []

    for item in pages:

        text = item["text"]
        page = item["page"]

        start = 0

        while start < len(text):

            end = start + chunk_size

            chunk = text[start:end]

            chunks.append({
                "text": chunk,
                "page": page
            })

            start += chunk_size - overlap

    return chunks