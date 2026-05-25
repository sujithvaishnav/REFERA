from fastapi import FastAPI, UploadFile, File
import shutil
import os
from typing import List

from rag.parser import extract_text_from_pdf
from rag.chunker import chunk_text
from rag.embeddings import generate_embedding
from rag.vectordb import store_chunks
from rag.retriever import hybrid_retrieve
from rag.generator import generate_answer
from rag.vectordb import collection
from rag.reranker import rerank_documents


from fastapi.responses import StreamingResponse
import json

app = FastAPI()
chat_history = []

BASE_DIR = os.path.dirname(
    os.path.abspath(__file__)
)

UPLOAD_DIR = os.path.join(
    BASE_DIR,
    "uploads"
)

os.makedirs(
    UPLOAD_DIR,
    exist_ok=True
)

@app.post("/upload")

async def upload_pdf(file: UploadFile = File(...)):

    file_path = os.path.join(
        UPLOAD_DIR,
        file.filename
    )

    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    pages = extract_text_from_pdf(
        file_path
    )

    chunks = chunk_text(pages)

    store_chunks(
        chunks,
        file.filename,
        generate_embedding
    )

    return {
        "message": "PDF uploaded successfully"
    }

@app.get("/ask")

def ask_question(query: str,selected_docs: str = None):

    global chat_history

    conversation_context = ""

    for item in chat_history[-5:]:

        conversation_context += f"""

        User: {item['question']}

        Assistant: {item['answer']}

        """

    enhanced_query = f"""

    Previous Conversation:
    {conversation_context}

    Current Question:
    {query}

    """
    docs_filter = None

    if selected_docs:
        docs_filter = selected_docs.split(",")
    
    retrieved_docs = hybrid_retrieve(
            enhanced_query
        )

    retrieved_docs = rerank_documents(
        enhanced_query,
        retrieved_docs,
        top_k=5
    )

    completion, sources = generate_answer(
        enhanced_query,
        retrieved_docs
    )

    def stream_generator():

        full_answer = ""

        for chunk in completion:

            content = chunk.choices[0].delta.content

            if content:

                full_answer += content

                yield f"data: {json.dumps({'token': content})}\n\n"

        chat_history.append({
            "question": query,
            "answer": full_answer
        })

        yield f"data: {json.dumps({'done': True, 'sources': sources})}\n\n"

    return StreamingResponse(
        stream_generator(),
        media_type="text/event-stream"
    )

@app.get("/documents")

def get_documents():

    data = collection.get()

    metadatas = data["metadatas"]

    unique_docs = list(set(
        meta["source"]
        for meta in metadatas
    ))

    return {
        "documents": unique_docs
    }