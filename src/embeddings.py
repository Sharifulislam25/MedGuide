"""
embeddings.py

Turns text into vectors (embeddings) using a local Sentence Transformers
model. No API key, and no internet needed after the model's first
download (it's cached locally by the sentence-transformers library).

The same model is used for both document chunks and user questions, so
their vectors land in the same "space" and can be meaningfully compared
via similarity search later (Phase 9+).
"""

from typing import List
from sentence_transformers import SentenceTransformer

from src.config import EMBEDDING_MODEL_NAME

# Loading a SentenceTransformer model takes a few seconds and holds it
# in memory, so we only want to do that once -- not on every single
# call to embed_texts(). This module-level variable acts as a simple
# cache for the lifetime of the running app.
_model = None


def get_embedding_model() -> SentenceTransformer:
    """
    Load (and cache) the embedding model.

    The very first call downloads the model (roughly 90MB) from
    HuggingFace and caches it on disk; every call after that -- even in
    a later run of the app -- loads it from the local cache instead.
    """
    global _model
    if _model is None:
        _model = SentenceTransformer(EMBEDDING_MODEL_NAME)
    return _model


def embed_texts(texts: List[str]) -> List[List[float]]:
    """
    Convert a list of text strings into embedding vectors.

    Parameters
    ----------
    texts : List[str]
        The chunks (or questions) to embed.

    Returns
    -------
    List[List[float]]
        One embedding vector per input text, in the same order. Returns
        an empty list if texts is empty.
    """
    if not texts:
        return []

    model = get_embedding_model()
    vectors = model.encode(texts, show_progress_bar=False)
    return vectors.tolist()


def embed_query(query: str) -> List[float]:
    """
    Convenience wrapper for embedding a single piece of text (typically
    a user's question). Uses the exact same model as embed_texts(), so
    documents and queries always live in the same vector space.
    """
    return embed_texts([query])[0]
