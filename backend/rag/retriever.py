from rag.vectordb import collection
from rag.embeddings import generate_embedding

from rank_bm25 import BM25Okapi

def retrieve(query, top_k=5, selected_docs=None):

    query_embedding = generate_embedding(query)

    if selected_docs:

        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            where={
                "source": {
                    "$in": selected_docs
                }
            },
            include=[
                "documents",
                "metadatas",
                "distances"
            ]
        )

    else:

        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            include=[
                "documents",
                "metadatas",
                "distances"
            ]
        )

    return results

def bm25_search(
    query,
    top_k=5
):

    data = collection.get()

    documents = data["documents"]
    metadatas = data["metadatas"]

    tokenized_docs = [
        doc.split()
        for doc in documents
    ]

    bm25 = BM25Okapi(
        tokenized_docs
    )

    tokenized_query = query.split()

    scores = bm25.get_scores(
        tokenized_query
    )

    ranked_indices = sorted(
        range(len(scores)),
        key=lambda i: scores[i],
        reverse=True
    )[:top_k]

    retrieved_docs = []
    retrieved_meta = []

    for idx in ranked_indices:

        retrieved_docs.append(
            documents[idx]
        )

        retrieved_meta.append(
            metadatas[idx]
        )

    return {
        "documents": [retrieved_docs],
        "metadatas": [retrieved_meta]
    }

def hybrid_retrieve(
    query,
    top_k_dense=8,
    top_k_bm25=5
):

    dense_results = retrieve(
        query,
        top_k=top_k_dense
    )

    bm25_results = bm25_search(
        query,
        top_k=top_k_bm25
    )

    combined_docs = []
    combined_meta = []

    seen = set()

    # Dense Retrieval Results

    for doc, meta in zip(
        dense_results["documents"][0],
        dense_results["metadatas"][0]
    ):

        key = (
            doc[:100],
            meta["page"],
            meta["source"]
        )

        if key not in seen:

            seen.add(key)

            combined_docs.append(doc)
            combined_meta.append(meta)

    # BM25 Results

    for doc, meta in zip(
        bm25_results["documents"][0],
        bm25_results["metadatas"][0]
    ):

        key = (
            doc[:100],
            meta["page"],
            meta["source"]
        )

        if key not in seen:

            seen.add(key)

            combined_docs.append(doc)
            combined_meta.append(meta)

    return {
        "documents": [combined_docs],
        "metadatas": [combined_meta]
    }