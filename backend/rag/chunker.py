from langchain_text_splitters import RecursiveCharacterTextSplitter

def chunk_text(
    pages,
    chunk_size=1200,
    overlap=200
):

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=overlap
    )

    merged_text = ""

    page_boundaries = []

    current_pos = 0

    for page in pages:

        marker = (
            f"\n\n---PAGE {page['page']}---\n\n"
        )

        merged_text += marker
        current_pos += len(marker)

        start_pos = current_pos

        merged_text += page["text"]

        current_pos += len(page["text"])

        page_boundaries.append(
            (
                start_pos,
                current_pos,
                page["page"]
            )
        )

    splits = splitter.split_text(
        merged_text
    )

    chunks = []

    cursor = 0

    for split in splits:

        page_num = 1

        for start,end,page in page_boundaries:

            if start <= cursor <= end:
                page_num = page
                break

        chunks.append({
            "text": split,
            "page": page_num
        })

        cursor += len(split)

    return chunks