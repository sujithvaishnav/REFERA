from sentence_transformers import SentenceTransformer

from rag.cache import get_cached_embedding, set_cached_embedding

model = SentenceTransformer(
    "all-MiniLM-L6-v2"
)

def generate_embedding(text):

    cached = get_cached_embedding(text)

    if cached is not None:
        return cached

    embedding = model.encode(text).tolist()

    set_cached_embedding(text, embedding)

    return embedding