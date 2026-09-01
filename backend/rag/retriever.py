from supabase import create_client, Client
from dotenv import load_dotenv
import os
from rank_bm25 import BM25Okapi

load_dotenv()

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")

if not SUPABASE_URL or not SUPABASE_KEY:
    raise ValueError("SUPABASE_URL and SUPABASE_KEY must be set in the .env file")

supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

bm25_index = None
bm25_documents = []
bm25_metadatas = []

def build_bm25_index(user_id=None):
    """
    Fetches all chunks uploaded by the user to build/rebuild the BM25 index in memory.
    """
    global bm25_index
    global bm25_documents
    global bm25_metadatas

    if not user_id:
        bm25_index = None
        bm25_documents = []
        bm25_metadatas = []
        return

    # Fetch all chunks from Supabase for this user
    response = supabase.table("document_chunks") \
        .select("content", "metadata") \
        .eq("user_id", user_id) \
        .execute()

    data = response.data or []
    if not data:
        bm25_index = None
        bm25_documents = []
        bm25_metadatas = []
        return

    bm25_documents = [item["content"] for item in data if item.get("content")]
    bm25_metadatas = [item.get("metadata", {}) for item in data if item.get("content")]

    tokenized_docs = [
        doc.split()
        for doc in bm25_documents
    ]

    if len(tokenized_docs) == 0:
        bm25_index = None
        return

    bm25_index = BM25Okapi(tokenized_docs)

def retrieve(query, user_id=None, top_k=5, selected_docs=None):
    """
    Queries Supabase using pgvector cosine similarity search.
    """
    if not user_id:
        return {"documents": [[]], "metadatas": [[]]}

    from rag.embeddings import generate_embedding
    query_embedding = generate_embedding(query)

    # Call the pgvector similarity search RPC function in Supabase
    params = {
        "query_embedding": query_embedding,
        "match_threshold": -1.0,  # Rank all and sort by distance
        "match_count": top_k,
        "filter_user_id": user_id
    }

    if selected_docs:
        params["filter_filenames"] = selected_docs

    try:
        response = supabase.rpc("match_document_chunks", params).execute()
        data = response.data or []
    except Exception:
        data = []

    documents = []
    metadatas = []

    for row in data:
        documents.append(row.get("content", ""))
        metadatas.append(row.get("metadata", {}))

    return {
        "documents": [documents],
        "metadatas": [metadatas]
    }

def bm25_search(
    query,
    top_k=5,
    selected_docs=None
):
    global bm25_index
    global bm25_documents
    global bm25_metadatas

    if bm25_index is None or not bm25_documents:
        return {
            "documents": [[]],
            "metadatas": [[]]
        }

    tokenized_query = query.split()
    if not tokenized_query:
        return {
            "documents": [[]],
            "metadatas": [[]]
        }

    scores = bm25_index.get_scores(
        tokenized_query
    )

    ranked_indices = sorted(
        range(len(scores)),
        key=lambda i: scores[i],
        reverse=True
    )

    retrieved_docs = []
    retrieved_meta = []

    for idx in ranked_indices:
        metadata = bm25_metadatas[idx] if idx < len(bm25_metadatas) else {}

        source = metadata.get("source", "")
        if selected_docs is not None and source not in selected_docs:
            continue

        retrieved_docs.append(
            bm25_documents[idx]
        )

        retrieved_meta.append(
            metadata
        )

        if len(retrieved_docs) >= top_k:
            break

    return {
        "documents": [retrieved_docs],
        "metadatas": [retrieved_meta]
    }

def reciprocal_rank_fusion(
    dense_docs,
    dense_meta,
    bm25_docs,
    bm25_meta,
    k=60
):
    """
    Combines dense vector search results and sparse BM25 search results
    using Reciprocal Rank Fusion (RRF).
    """
    scores = {}

    # Dense results
    for rank, (doc, meta) in enumerate(
        zip(dense_docs, dense_meta),
        start=1
    ):
        key = (
            doc[:100],
            meta.get("page", 1),
            meta.get("source", "unknown")
        )

        if key not in scores:
            scores[key] = {
                "score": 0.0,
                "doc": doc,
                "meta": meta
            }

        scores[key]["score"] += (
            1.0 / (k + rank)
        )

    # BM25 results
    for rank, (doc, meta) in enumerate(
        zip(bm25_docs, bm25_meta),
        start=1
    ):
        key = (
            doc[:100],
            meta.get("page", 1),
            meta.get("source", "unknown")
        )

        if key not in scores:
            scores[key] = {
                "score": 0.0,
                "doc": doc,
                "meta": meta
            }

        scores[key]["score"] += (
            1.0 / (k + rank)
        )

    ranked = sorted(
        scores.values(),
        key=lambda x: x["score"],
        reverse=True
    )

    return ranked

def hybrid_retrieve(
    query,
    user_id=None,
    top_k_dense=8,
    top_k_bm25=5,
    selected_docs=None
):
    """
    Executes hybrid retrieval by combining dense semantic search (pgvector)
    and lexical search (BM25) via Reciprocal Rank Fusion (RRF).
    """
    dense_results = retrieve(
        query,
        user_id=user_id,
        top_k=top_k_dense,
        selected_docs=selected_docs
    )

    global bm25_index
    if bm25_index is None and user_id:
        build_bm25_index(user_id)

    if bm25_index is None:
        return dense_results

    try:
        bm25_results = bm25_search(
            query,
            top_k=top_k_bm25,
            selected_docs=selected_docs
        )
    except Exception:
        return dense_results

    dense_doc_list = dense_results.get("documents", [[]])[0]
    dense_meta_list = dense_results.get("metadatas", [[]])[0]
    bm25_doc_list = bm25_results.get("documents", [[]])[0]
    bm25_meta_list = bm25_results.get("metadatas", [[]])[0]

    if not dense_doc_list and not bm25_doc_list:
        return {"documents": [[]], "metadatas": [[]]}

    fused_results = reciprocal_rank_fusion(
        dense_doc_list,
        dense_meta_list,
        bm25_doc_list,
        bm25_meta_list
    )

    combined_docs = [
        item["doc"]
        for item in fused_results
    ]

    combined_meta = [
        item["meta"]
        for item in fused_results
    ]

    return {
        "documents": [combined_docs],
        "metadatas": [combined_meta]
    }
