import chromadb
import uuid
import os

# Resolve path relative to the backend directory
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
client = chromadb.PersistentClient(
    path=os.path.join(BASE_DIR, "chroma_db")
)

collection = client.get_or_create_collection(
    name="pdf_docs"
)

def delete_existing_document(
    filename
):

    existing = collection.get(
        where={
            "source": filename
        }
    )

    ids = existing["ids"]

    if ids:

        collection.delete(
            ids=ids
        )

def store_chunks(chunks, filename, generate_embedding):

    delete_existing_document(
        filename
    )

    documents = []
    embeddings = []
    metadatas = []
    ids = []

    for chunk in chunks:

        documents.append(
            chunk["text"]
        )

        embeddings.append(
            generate_embedding(
                chunk["text"]
            )
        )

        metadatas.append({
            "page": chunk["page"],
            "source": filename,
            "chunk_length": len(chunk["text"])
        })

        ids.append(
            str(uuid.uuid4())
        )

    collection.add(
        documents=documents,
        embeddings=embeddings,
        metadatas=metadatas,
        ids=ids
    )
