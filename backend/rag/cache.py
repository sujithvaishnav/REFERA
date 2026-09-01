import os
import json
import hashlib
import logging

import redis

logger = logging.getLogger(__name__)

REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")
CACHE_TTL_SECONDS = int(os.getenv("CACHE_TTL_SECONDS", 3600))  # 1 hour

_client = None
_redis_available = True


def get_client():
    """
    Lazily create a single shared Redis connection.
    If Redis is unreachable, we disable caching rather than crash the app —
    caching is a performance optimization, not a correctness dependency.
    """
    global _client, _redis_available

    if not _redis_available:
        return None

    if _client is None:
        try:
            _client = redis.from_url(
                REDIS_URL,
                socket_connect_timeout=1,
                socket_timeout=1,
                decode_responses=True,
            )
            _client.ping()
        except redis.exceptions.RedisError as exc:
            logger.warning(f"Redis unavailable, caching disabled: {exc}")
            _redis_available = False
            _client = None

    return _client


def make_answer_cache_key(query, selected_docs, recent_history):
    """
    Cache key composition:
    - raw query text (normalized)
    - selected_docs, order-independent
    - a hash of the last few conversation turns, so a cached answer is only
      reused when the conversational context that shaped it is unchanged
    """
    normalized_query = query.strip().lower()

    docs_part = ",".join(sorted(selected_docs)) if selected_docs else "all"

    history_repr = json.dumps(recent_history, sort_keys=True)
    history_hash = hashlib.sha256(history_repr.encode("utf-8")).hexdigest()[:16]

    raw_key = f"{normalized_query}|{docs_part}|{history_hash}"
    digest = hashlib.sha256(raw_key.encode("utf-8")).hexdigest()

    return f"refera:answer:{digest}"


def get_cached_answer(cache_key):
    client = get_client()

    if client is None:
        return None

    try:
        cached = client.get(cache_key)
    except redis.exceptions.RedisError as exc:
        logger.warning(f"Redis GET failed, treating as cache miss: {exc}")
        return None

    if cached is None:
        return None

    try:
        return json.loads(cached)
    except json.JSONDecodeError:
        return None


def set_cached_answer(cache_key, answer, sources, ttl=CACHE_TTL_SECONDS):
    client = get_client()

    if client is None:
        return

    payload = json.dumps({
        "answer": answer,
        "sources": sources,
    })

    try:
        client.setex(cache_key, ttl, payload)
    except redis.exceptions.RedisError as exc:
        logger.warning(f"Redis SET failed, continuing without caching: {exc}")


def make_embedding_cache_key(text):
    digest = hashlib.sha256(text.strip().lower().encode("utf-8")).hexdigest()
    return f"refera:embedding:{digest}"


def get_cached_embedding(text):
    client = get_client()

    if client is None:
        return None

    try:
        cached = client.get(make_embedding_cache_key(text))
    except redis.exceptions.RedisError as exc:
        logger.warning(f"Redis GET failed, treating as cache miss: {exc}")
        return None

    if cached is None:
        return None

    try:
        return json.loads(cached)
    except json.JSONDecodeError:
        return None


def set_cached_embedding(text, embedding, ttl=None):
    """Embeddings for identical text never change, so these are cached
    without expiry by default (ttl=None -> Redis SET with no TTL)."""
    client = get_client()

    if client is None:
        return

    key = make_embedding_cache_key(text)
    payload = json.dumps(embedding)

    try:
        if ttl:
            client.setex(key, ttl, payload)
        else:
            client.set(key, payload)
    except redis.exceptions.RedisError as exc:
        logger.warning(f"Redis SET failed, continuing without caching: {exc}")
