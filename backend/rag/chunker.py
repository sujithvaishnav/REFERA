from langchain_text_splitters import RecursiveCharacterTextSplitter

def chunk_text(
    pages,
    chunk_size=1200,
    overlap=200
):
    """
    Chunks document pages into semantically cohesive passages while strictly
    preserving the source page number for each chunk without injecting noisy marker tags.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=overlap,
        separators=["\n\n", "\n", ". ", " ", ""]
    )

    chunks = []
    for page_data in pages:
        page_num = page_data.get("page", 1)
        text = page_data.get("text", "")

        if not text.strip():
            continue

        page_splits = splitter.split_text(text)
        for split in page_splits:
            clean_split = split.strip()
            if clean_split:
                chunks.append({
                    "text": clean_split,
                    "page": page_num
                })

    return chunks