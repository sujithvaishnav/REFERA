from supabase import create_client, Client
from dotenv import load_dotenv
import os
from rag.summarizer import generate_summary

load_dotenv()

def get_config_var(key: str, default: str = None) -> str:
    val = os.getenv(key)
    if val:
        return val
    try:
        import streamlit as st
        if hasattr(st, "secrets") and key in st.secrets:
            return str(st.secrets[key])
    except Exception:
        pass
    return default

SUPABASE_URL = get_config_var("SUPABASE_URL")
SUPABASE_KEY = get_config_var("SUPABASE_KEY")

supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY) if (SUPABASE_URL and SUPABASE_KEY) else None

def get_supabase_client(client: Client = None) -> Client:
    global supabase
    if client:
        return client
    if supabase:
        return supabase
    url = get_config_var("SUPABASE_URL")
    key = get_config_var("SUPABASE_KEY")
    if url and key:
        supabase = create_client(url, key)
        return supabase
    raise ValueError("SUPABASE_URL and SUPABASE_KEY must be configured in your .env file or Streamlit Secrets.")

def delete_existing_document(filename: str, user_id: str, client: Client = None):
    """
    Deletes the document metadata and all associated chunks from Supabase by filename and user_id.
    """
    c = get_supabase_client(client)
    if not user_id or not filename:
        return
        
    c.table("documents") \
        .delete() \
        .eq("filename", filename) \
        .eq("user_id", user_id) \
        .execute()

def delete_document_by_id(document_id: str, user_id: str, client: Client = None):
    """
    Deletes the document and cascaded chunks by document UUID and user_id.
    """
    c = get_supabase_client(client)
    if not user_id or not document_id:
        return
        
    c.table("documents") \
        .delete() \
        .eq("id", document_id) \
        .eq("user_id", user_id) \
        .execute()

def get_user_documents(user_id: str = None, client: Client = None):
    """
    Retrieves all document records for a given user.
    """
    c = get_supabase_client(client)
    query = c.table("documents").select("id", "filename", "summary", "created_at")
    if user_id:
        query = query.eq("user_id", user_id)
    response = query.order("created_at", desc=True).execute()
    return response.data or []

def store_chunks(chunks, filename: str, generate_embedding, user_id: str = None, client: Client = None):
    """
    Stores document metadata and text chunks with vector embeddings in Supabase pgvector,
    and automatically generates and attaches an executive summary of the document.
    """
    c = get_supabase_client(client)
    if not user_id:
        raise ValueError("User must be authenticated to upload documents.")

    if not chunks:
        raise ValueError(f"No text chunks found in {filename} to store.")

    # 1. Clear any existing document with the same name for this user
    delete_existing_document(filename, user_id, client=c)

    # 2. Insert document record
    doc_response = c.table("documents").insert({
        "user_id": user_id,
        "filename": filename
    }).execute()

    if not doc_response.data:
        raise RuntimeError(f"Failed to create document record for {filename}")

    document_id = doc_response.data[0]["id"]

    # 3. Prepare chunk records with embeddings
    chunk_records = []
    for chunk in chunks:
        embedding = generate_embedding(chunk["text"])
        chunk_records.append({
            "document_id": document_id,
            "user_id": user_id,
            "content": chunk["text"],
            "metadata": {
                "page": chunk.get("page", 1),
                "source": filename,
                "chunk_length": len(chunk["text"])
            },
            "embedding": embedding
        })

    # 4. Insert chunks in batches of 50 to avoid request payload size limits
    batch_size = 50
    for i in range(0, len(chunk_records), batch_size):
        batch = chunk_records[i:i+batch_size]
        c.table("document_chunks").insert(batch).execute()

    # 5. Generate and save document summary (with graceful fallback if LLM times out)
    try:
        summary_text = generate_summary(chunks)
    except Exception as exc:
        summary_text = f"Summary generation unavailable: {str(exc)}"
    
    try:
        c.table("documents") \
            .update({"summary": summary_text}) \
            .eq("id", document_id) \
            .execute()
    except Exception:
        pass
        
    return summary_text
