import chromadb
import uuid

client = chromadb.PersistentClient(
    path="../backend/chroma_db"
)

collection = client.get_or_create_collection(
    name="pdf_docs"
)

def store_chunks(chunks, filename, generate_embedding):

    for i, chunk in enumerate(chunks):

        embedding = generate_embedding(
            chunk["text"]
        )

        collection.add(
            documents=[chunk["text"]],
            embeddings=[embedding],
            metadatas=[{
                "page": chunk["page"],
                "source": filename,
                "chunk_length": len(chunk["text"])
            }],
            ids=[str(uuid.uuid4())]
        )