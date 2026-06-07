from rag.vectordb import collection
from rag.embeddings import generate_embedding

from rank_bm25 import BM25Okapi

bm25_index = None
bm25_documents = []
bm25_metadatas = []

def build_bm25_index():

    global bm25_index
    global bm25_documents
    global bm25_metadatas

    data = collection.get()

    bm25_documents = data["documents"]
    bm25_metadatas = data["metadatas"]

    tokenized_docs = [
        doc.split()
        for doc in bm25_documents
    ]

    if len(tokenized_docs) == 0:
        bm25_index = None
        return

    bm25_index = BM25Okapi(tokenized_docs)

    print(
        f"BM25 index built with {len(bm25_documents)} chunks"
    )

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
    top_k=5,
    selected_docs=None
):

    global bm25_index
    global bm25_documents
    global bm25_metadatas

    if bm25_index is None:

        build_bm25_index()

    if bm25_index is None:

        return {
            "documents": [[]],
            "metadatas": [[]]
        }

    tokenized_query = query.split()

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

        metadata = bm25_metadatas[idx]

        if (
            selected_docs is not None and
            metadata["source"] not in selected_docs
        ):
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

    scores = {}

    # Dense results
    for rank, (doc, meta) in enumerate(
        zip(dense_docs, dense_meta),
        start=1
    ):

        key = (
            doc[:100],
            meta["page"],
            meta["source"]
        )

        if key not in scores:

            scores[key] = {
                "score": 0,
                "doc": doc,
                "meta": meta
            }

        scores[key]["score"] += (
            1 / (k + rank)
        )

    # BM25 results
    for rank, (doc, meta) in enumerate(
        zip(bm25_docs, bm25_meta),
        start=1
    ):

        key = (
            doc[:100],
            meta["page"],
            meta["source"]
        )

        if key not in scores:

            scores[key] = {
                "score": 0,
                "doc": doc,
                "meta": meta
            }

        scores[key]["score"] += (
            1 / (k + rank)
        )

    ranked = sorted(
        scores.values(),
        key=lambda x: x["score"],
        reverse=True
    )

    return ranked

def hybrid_retrieve(
    query,
    top_k_dense=8,
    top_k_bm25=5,
    selected_docs=None
):

    dense_results = retrieve(
        query,
        top_k=top_k_dense,
        selected_docs=selected_docs
    )

    bm25_results = bm25_search(
        query,
        top_k=top_k_bm25,
        selected_docs=selected_docs
    )

    fused_results = reciprocal_rank_fusion(
        dense_results["documents"][0],
        dense_results["metadatas"][0],
        bm25_results["documents"][0],
        bm25_results["metadatas"][0]
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
