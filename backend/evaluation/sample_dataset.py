evaluation_data = [

    {
        "question":
        "What problem does Retrieval-Augmented Generation (RAG) solve?",

        "ground_truth":
        "RAG solves the limitation of static parametric knowledge in language models by retrieving relevant external documents during generation.",

        "expected_keywords": [
            "external knowledge",
            "retrieval",
            "generation",
            "parametric memory"
        ]
    },

    {
        "question":
        "Why is retrieval important in RAG systems?",

        "ground_truth":
        "Retrieval allows language models to access relevant and up-to-date information from external sources, improving factual accuracy and reducing hallucinations.",

        "expected_keywords": [
            "factual accuracy",
            "external documents",
            "hallucinations",
            "up-to-date information"
        ]
    },

    {
        "question":
        "What are the main components of a RAG pipeline?",

        "ground_truth":
        "A RAG pipeline typically contains a retriever, vector database, embedding model, and generator language model.",

        "expected_keywords": [
            "retriever",
            "vector database",
            "embedding model",
            "generator"
        ]
    },

    {
        "question":
        "What is self-attention in the Transformer architecture?",

        "ground_truth":
        "Self-attention allows each token in a sequence to attend to every other token and compute contextual relationships dynamically.",

        "expected_keywords": [
            "self-attention",
            "contextual relationships",
            "tokens",
            "sequence"
        ]
    },

    {
        "question":
        "Why was the Transformer architecture introduced?",

        "ground_truth":
        "The Transformer architecture was introduced to replace recurrent and convolutional sequence models with a fully attention-based mechanism that enables parallel computation.",

        "expected_keywords": [
            "parallel computation",
            "attention mechanism",
            "recurrent networks",
            "convolutional networks"
        ]
    },

    {
        "question":
        "What is the role of positional encoding in Transformers?",

        "ground_truth":
        "Positional encoding injects information about token order into the Transformer because self-attention alone does not capture sequence order.",

        "expected_keywords": [
            "token order",
            "sequence order",
            "positional encoding",
            "self-attention"
        ]
    },

    {
        "question":
        "What are query, key, and value vectors in attention mechanisms?",

        "ground_truth":
        "Query, key, and value vectors are learned representations used to calculate attention scores and weighted contextual representations.",

        "expected_keywords": [
            "query",
            "key",
            "value",
            "attention scores"
        ]
    },

    {
        "question":
        "What is multi-head attention?",

        "ground_truth":
        "Multi-head attention allows the Transformer to attend to information from multiple representation subspaces simultaneously.",

        "expected_keywords": [
            "multi-head attention",
            "multiple subspaces",
            "parallel attention",
            "representations"
        ]
    },

    {
        "question":
        "How does RAG help reduce hallucinations in LLMs?",

        "ground_truth":
        "RAG reduces hallucinations by grounding responses in retrieved external documents instead of relying only on model parameters.",

        "expected_keywords": [
            "grounding",
            "retrieved documents",
            "hallucinations",
            "external knowledge"
        ]
    },

    {
        "question":
        "What is the advantage of Transformers over RNNs?",

        "ground_truth":
        "Transformers enable parallel processing and better long-range dependency modeling compared to recurrent neural networks.",

        "expected_keywords": [
            "parallel processing",
            "long-range dependencies",
            "RNNs",
            "Transformer"
        ]
    }

]