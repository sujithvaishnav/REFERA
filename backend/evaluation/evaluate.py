import pandas as pd
import time

from evaluation.sample_dataset import evaluation_data
from rag.reranker import rerank_documents

from evaluation.metrics import (
    semantic_similarity,
    keyword_match,
    retrieval_hit
)

from rag.retriever import hybrid_retrieve
from rag.generator import generate_answer_eval

results = []

for item in evaluation_data:

    question = item["question"]

    start = time.time()

    retrieved_docs = hybrid_retrieve(
            question
        )

    retrieved_docs = rerank_documents(
        question,
        retrieved_docs,
        top_k=5
    )

    response = generate_answer_eval(
        question,
        retrieved_docs
    )

    end = time.time()

    answer = response["answer"]

    contexts = retrieved_docs["documents"][0]

    similarity_score = semantic_similarity(
        item["ground_truth"],
        answer
    )

    keyword_score = keyword_match(
        answer,
        item["expected_keywords"]
    )

    retrieval_score = retrieval_hit(
        contexts,
        item["expected_keywords"]
    )

    latency = end - start

    results.append({

        "question": question,

        "semantic_similarity":
        round(similarity_score, 3),

        "keyword_match":
        round(keyword_score, 3),

        "retrieval_hit":
        retrieval_score,

        "latency_seconds":
        round(latency, 2)
    })

df = pd.DataFrame(results)

print(df)

print("\nAVERAGE SCORES\n")

print(df.mean(numeric_only=True))