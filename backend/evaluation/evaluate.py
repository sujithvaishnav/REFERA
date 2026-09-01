import pandas as pd
import time
import os
import sys

# Ensure backend root is in sys.path
backend_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if backend_dir not in sys.path:
    sys.path.insert(0, backend_dir)

from evaluation.sample_dataset import evaluation_data
from rag.reranker import rerank_documents
from rag.metrics import (
    semantic_similarity,
    keyword_match,
    retrieval_hit
) if os.path.exists(os.path.join(backend_dir, "rag", "metrics.py")) else None

from evaluation.metrics import (
    semantic_similarity,
    keyword_match,
    retrieval_hit
)
from rag.retriever import hybrid_retrieve
from rag.generator import generate_answer_eval

def run_evaluation(user_id=None, selected_docs=None, save_csv=True):
    print("=" * 60)
    print("  REFERA - RAG BENCHMARK & EVALUATION RUNNER")
    print("=" * 60)
    
    results = []
    
    for idx, item in enumerate(evaluation_data, 1):
        question = item["question"]
        print(f"\n[{idx}/{len(evaluation_data)}] Evaluating: {question[:50]}...")
        
        start = time.time()
        
        # 1. Hybrid Retrieve
        retrieved_docs = hybrid_retrieve(
            query=question,
            user_id=user_id,
            selected_docs=selected_docs
        )
        
        # 2. Rerank
        retrieved_docs = rerank_documents(
            query=question,
            retrieved_docs=retrieved_docs,
            top_k=5
        )
        
        # 3. Generate Answer
        response = generate_answer_eval(
            query=question,
            retrieved_docs=retrieved_docs
        )
        
        end = time.time()
        latency = end - start
        
        answer = response.get("answer", "")
        contexts = retrieved_docs.get("documents", [[]])[0]
        
        # 4. Metrics Computation
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
        
        results.append({
            "question": question,
            "semantic_similarity": round(similarity_score, 3),
            "keyword_match": round(keyword_score, 3),
            "retrieval_hit": retrieval_score,
            "latency_seconds": round(latency, 2)
        })

    df = pd.DataFrame(results)
    
    print("\n" + "=" * 60)
    print("  DETAILED EVALUATION RESULTS")
    print("=" * 60)
    print(df.to_string(index=False))
    
    print("\n" + "=" * 60)
    print("  AVERAGE BENCHMARK SCORES")
    print("=" * 60)
    mean_scores = df.mean(numeric_only=True)
    for metric, value in mean_scores.items():
        print(f"  • {metric:<25}: {value:.3f}")
    print("=" * 60)
    
    if save_csv:
        output_file = os.path.join(backend_dir, "evaluation", "evaluation_results.csv")
        df.to_csv(output_file, index=False)
        print(f"\nResults saved to: {output_file}")
        
    return df

if __name__ == "__main__":
    eval_user_id = sys.argv[1] if len(sys.argv) > 1 else None
    run_evaluation(user_id=eval_user_id)