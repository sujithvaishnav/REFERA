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
    last_exception = None
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

def generate_summary(chunks):
    """
    Generates a concise summary of the document based on the first few chunks.
    """
    if not chunks:
        return "No content available to summarize."
        
    # Take up to the first 6 chunks, which typically contains the abstract, intro, and metadata.
    summary_chunks = chunks[:6]
    combined_text = "\n\n".join([chunk.get("text", "") for chunk in summary_chunks])
    
    prompt = f"""
    You are an expert research assistant.
    Provide a professional, concise, and structured summary of the following document content (extracted from the beginning of a research paper or PDF).
    
    The summary MUST include:
    1. **Overview**: A brief description of the document's main objective/topic.
    2. **Key Findings / Contributions**: Major points or conclusions.
    3. **Significance**: Why this document is important.
    
    Format the summary nicely in Markdown with clear sections.
    
    CONTENT TO SUMMARIZE:
    {combined_text}
    """
    
    try:
        completion = create_groq_completion(
            messages=[
                {
                    "role": "system",
                    "content": "You are a professional academic summarizer."
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            temperature=0.3
        )
        
        summary = completion.choices[0].message.content
        return summary
    except Exception as e:
        logger.error(f"Error generating summary: {e}")
        return f"Summary generation unavailable: {str(e)}"
