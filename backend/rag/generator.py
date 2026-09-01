from groq import Groq
from dotenv import load_dotenv
import os
import logging

load_dotenv()

logger = logging.getLogger(__name__)

client = Groq(
    api_key=os.getenv("GROQ_API_KEY")
)

FALLBACK_MODELS = [
    os.getenv("GROQ_MODEL", "qwen/qwen3.8-27b"),
    "qwen/qwen3.8-27b",
    "groq/compound-mini",
    "openai/gpt-oss-120b",
    "openai/gpt-oss-20b",
    "llama-3.3-70b-versatile",
    "llama-3.1-8b-instant"
]

def create_groq_completion(**kwargs):
    """
    Attempts completion with preferred model and gracefully falls back across available Groq models.
    """
    last_exception = None
    # Deduplicate while preserving priority
    seen = set()
    models_to_try = [m for m in FALLBACK_MODELS if m and not (m in seen or seen.add(m))]

    for model_name in models_to_try:
        try:
            kwargs["model"] = model_name
            return client.chat.completions.create(**kwargs)
        except Exception as e:
            err_str = str(e)
            last_exception = e
            if "model_not_found" in err_str or "does not exist" in err_str or "404" in err_str:
                logger.warning(f"Groq model {model_name} not available, trying next fallback...")
                continue
            raise e

    raise last_exception

def generate_answer(query, retrieved_docs):
    context = ""
    sources = []

    docs = retrieved_docs["documents"][0]
    metas = retrieved_docs["metadatas"][0]

    for doc, meta in zip(docs, metas):
        page = meta.get("page", 1)
        source = meta.get("source", "Unknown")

        sources.append({
            "page": page,
            "source": source,
            "snippet": doc[:300]
        })

        context += f"""
        Page: {page}
        Source: {source}
        {doc}
        """

    prompt = f"""
    You are a technical research assistant.

    Answer ONLY using the provided context.

    IMPORTANT RULES:
    - Preserve exact technical terminology from the context
    - Do NOT overly paraphrase
    - Use exact phrases when possible
    - Include important concepts explicitly
    - If multiple sources discuss the topic, combine them carefully
    - Mention page references clearly
    - If answer is not found, say so

    CONTEXT:
    {context}

    QUESTION:
    {query}

    Provide a technically accurate answer.
    """

    completion = create_groq_completion(
        messages=[
            {
                "role": "system",
                "content": "You are a helpful PDF assistant."
            },
            {
                "role": "user",
                "content": prompt
            }
        ],
        temperature=0.2,
        stream=True
    )

    return completion, sources

def generate_answer_eval(query, retrieved_docs):
    context = ""

    docs = retrieved_docs["documents"][0]
    metas = retrieved_docs["metadatas"][0]

    for doc, meta in zip(docs, metas):
        context += f"""
        Page: {meta.get('page', 1)}
        Source: {meta.get('source', 'Unknown')}
        {doc}
        """

    prompt = f"""
    Answer ONLY from provided context.

    CONTEXT:
    {context}

    QUESTION:
    {query}
    """

    completion = create_groq_completion(
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ],
        temperature=0.2,
        stream=False
    )

    answer = completion.choices[0].message.content

    return {
        "answer": answer
    }