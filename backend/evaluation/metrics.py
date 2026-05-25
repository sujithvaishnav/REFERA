from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import time

model = SentenceTransformer(
    "all-MiniLM-L6-v2"
)

# -----------------------------
# Semantic Similarity
# -----------------------------

def semantic_similarity(
    ground_truth,
    answer
):

    emb1 = model.encode([ground_truth])
    emb2 = model.encode([answer])

    score = cosine_similarity(
        emb1,
        emb2
    )[0][0]

    return float(score)

# -----------------------------
# Keyword Coverage
# -----------------------------

def keyword_match(
    answer,
    expected_keywords
):

    answer_lower = answer.lower()

    matched = 0

    for keyword in expected_keywords:

        if keyword.lower() in answer_lower:
            matched += 1

    return matched / len(expected_keywords)

# -----------------------------
# Retrieval Hit
# -----------------------------

def retrieval_hit(
    contexts,
    expected_keywords
):

    combined_context = " ".join(
        contexts
    ).lower()

    for keyword in expected_keywords:

        if keyword.lower() in combined_context:
            return 1

    return 0