from sentence_transformers import CrossEncoder

reranker_model = CrossEncoder(
    "cross-encoder/ms-marco-MiniLM-L-6-v2"
)

def rerank_documents(
    query,
    retrieved_docs,
    top_k=3
):

    docs = retrieved_docs["documents"][0]
    metas = retrieved_docs["metadatas"][0]

    pairs = []

    for doc in docs:

        pairs.append([query, doc])

    scores = reranker_model.predict(
        pairs
    )

    scored_results = []

    for score, doc, meta in zip(
        scores,
        docs,
        metas
    ):

        scored_results.append({
            "score": float(score),
            "document": doc,
            "metadata": meta
        })

    scored_results = sorted(
        scored_results,
        key=lambda x: x["score"],
        reverse=True
    )

    top_results = scored_results[:top_k]

    reranked_docs = {
        "documents": [[
            item["document"]
            for item in top_results
        ]],

        "metadatas": [[
            item["metadata"]
            for item in top_results
        ]]
    }

    return reranked_docs