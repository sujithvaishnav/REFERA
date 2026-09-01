from fastapi import FastAPI, UploadFile, File, Form, Header, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from typing import List, Optional
import shutil
import os
import json
import logging

from rag.parser import extract_text_from_pdf
from rag.chunker import chunk_text
from rag.embeddings import generate_embedding
from rag.vectordb import store_chunks, delete_document_by_id, get_user_documents, supabase
from rag.retriever import hybrid_retrieve, build_bm25_index
from rag.generator import generate_answer
from rag.reranker import rerank_documents
from rag.cache import make_answer_cache_key, get_cached_answer, set_cached_answer, get_client

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="ReferA API",
    description="Intelligent Multi-Document Research Assistant with Hybrid RAG, pgvector, and Redis Caching",
    version="2.0.0"
)

# Enable CORS for external frontends and microservices
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_DIR = os.path.join(BASE_DIR, "uploads")
os.makedirs(UPLOAD_DIR, exist_ok=True)

DEFAULT_USER_ID = "00000000-0000-0000-0000-000000000000"

@app.get("/")
def root():
    return {
        "name": "ReferA API",
        "version": "2.0.0",
        "status": "operational",
        "docs_url": "/docs"
    }

@app.get("/health")
def health_check():
    redis_client = get_client()
    redis_healthy = redis_client is not None
    
    return {
        "status": "healthy",
        "services": {
            "redis_cache": "connected" if redis_healthy else "disabled/unreachable",
            "supabase_db": "configured"
        }
    }

@app.post("/upload")
async def upload_pdf(
    file: UploadFile = File(...),
    user_id: Optional[str] = Form(None),
    x_user_id: Optional[str] = Header(None)
):
    """
    Upload a PDF document, extract pages, generate text chunks, embed with MiniLM,
    store into Supabase pgvector, build BM25 index, and return an auto-generated executive summary.
    """
    effective_user_id = user_id or x_user_id or DEFAULT_USER_ID

    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")

    file_path = os.path.join(UPLOAD_DIR, file.filename)

    try:
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        pages = extract_text_from_pdf(file_path)
        if not pages:
            raise HTTPException(status_code=400, detail="No readable text found in the uploaded PDF.")

        chunks = chunk_text(pages)
        if not chunks:
            raise HTTPException(status_code=400, detail="Could not create text chunks from document.")

        summary_text = store_chunks(
            chunks=chunks,
            filename=file.filename,
            generate_embedding=generate_embedding,
            user_id=effective_user_id
        )

        build_bm25_index(effective_user_id)

        return {
            "message": "PDF uploaded, embedded, and indexed successfully.",
            "filename": file.filename,
            "total_pages": len(pages),
            "total_chunks": len(chunks),
            "summary": summary_text
        }

    except Exception as e:
        logger.error(f"Error during upload: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if os.path.exists(file_path):
            try:
                os.remove(file_path)
            except Exception:
                pass

@app.get("/ask")
def ask_question(
    query: str = Query(..., description="The user's question"),
    selected_docs: Optional[str] = Query(None, description="Comma-separated list of filenames to filter by"),
    user_id: Optional[str] = Query(None, description="User ID for multi-tenant isolation"),
    x_user_id: Optional[str] = Header(None)
):
    """
    Ask a question against indexed documents using Hybrid Dense (pgvector) + Sparse (BM25)
    Retrieval with Cross-Encoder Reranking and Server-Sent Event (SSE) streaming.
    """
    effective_user_id = user_id or x_user_id or DEFAULT_USER_ID
    docs_filter = selected_docs.split(",") if selected_docs else None

    # Retrieve recent chat history for context (up to 5 recent turns)
    recent_history = []
    try:
        history_res = supabase.table("chat_messages") \
            .select("role", "content") \
            .eq("user_id", effective_user_id) \
            .order("created_at", desc=True) \
            .limit(10) \
            .execute()
        
        raw_msgs = (history_res.data or [])[::-1]
        for i in range(0, len(raw_msgs) - 1, 2):
            if raw_msgs[i].get("role") == "user" and raw_msgs[i+1].get("role") == "assistant":
                recent_history.append({
                    "question": raw_msgs[i].get("content", ""),
                    "answer": raw_msgs[i+1].get("content", "")
                })
    except Exception:
        recent_history = []

    conversation_context = ""
    for item in recent_history[-5:]:
        conversation_context += f"\nUser: {item['question']}\nAssistant: {item['answer']}\n"

    enhanced_query = f"""
    Previous Conversation:
    {conversation_context}

    Current Question:
    {query}
    """

    # Check cache-aside key
    cache_key = make_answer_cache_key(query, docs_filter, recent_history)
    cached = get_cached_answer(cache_key)

    if cached is not None:
        def cached_stream_generator():
            yield f"data: {json.dumps({'token': cached['answer']})}\n\n"
            yield f"data: {json.dumps({'done': True, 'sources': cached['sources'], 'cached': True})}\n\n"

        return StreamingResponse(
            cached_stream_generator(),
            media_type="text/event-stream"
        )

    # 1. Hybrid Retrieval (Dense pgvector + Sparse BM25 via RRF)
    retrieved_docs = hybrid_retrieve(
        query=enhanced_query,
        user_id=effective_user_id,
        selected_docs=docs_filter
    )

    # 2. Cross-Encoder Reranker
    retrieved_docs = rerank_documents(
        query=enhanced_query,
        retrieved_docs=retrieved_docs,
        top_k=5
    )

    # 3. LLM Completion Generation
    completion, sources = generate_answer(
        query=enhanced_query,
        retrieved_docs=retrieved_docs
    )

    def stream_generator():
        full_answer = ""
        for chunk in completion:
            content = chunk.choices[0].delta.content
            if content:
                full_answer += content
                yield f"data: {json.dumps({'token': content})}\n\n"

        # Cache complete answer and citations
        set_cached_answer(cache_key, full_answer, sources)
        yield f"data: {json.dumps({'done': True, 'sources': sources, 'cached': False})}\n\n"

    return StreamingResponse(
        stream_generator(),
        media_type="text/event-stream"
    )

@app.get("/documents")
def get_documents(
    user_id: Optional[str] = Query(None),
    x_user_id: Optional[str] = Header(None)
):
    """
    Retrieve all document metadata and summaries for the specified user.
    """
    effective_user_id = user_id or x_user_id or DEFAULT_USER_ID
    docs = get_user_documents(effective_user_id)
    return {
        "documents": docs
    }

@app.delete("/documents/{document_id}")
def delete_document(
    document_id: str,
    user_id: Optional[str] = Query(None),
    x_user_id: Optional[str] = Header(None)
):
    """
    Delete a document and all cascaded text chunk embeddings from the knowledge base.
    """
    effective_user_id = user_id or x_user_id or DEFAULT_USER_ID
    delete_document_by_id(document_id, effective_user_id)
    build_bm25_index(effective_user_id)
    return {
        "message": f"Document {document_id} deleted successfully."
    }