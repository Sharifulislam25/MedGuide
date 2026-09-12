"""
vector_store.py

Wraps ChromaDB so the rest of the app doesn't need to know the details
of its API. Uses a persistent local client -- everything gets written
to disk under CHROMA_DB_PATH, so stored chunks and their embeddings
survive between runs instead of needing to be regenerated every time
the app starts.
"""

from typing import List, Dict, Any
import chromadb

from src.config import CHROMA_DB_PATH
from src.chunker import Chunk

# Reuse a single client for the app's lifetime rather than creating a
# new one on every call (same caching idea as the embedding model).
_client = None


def get_chroma_client():
    """Create (or reuse) the persistent local ChromaDB client."""
    global _client
    if _client is None:
        _client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
    return _client


def get_collection(collection_name: str):
    """Get a ChromaDB collection, creating it if it doesn't exist yet."""
    client = get_chroma_client()
    return client.get_or_create_collection(name=collection_name)


def add_chunks(collection_name: str, chunks: List[Chunk], embeddings: List[List[float]]) -> None:
    """
    Store chunks and their embeddings in a ChromaDB collection.

    Parameters
    ----------
    collection_name : str
        Which collection to store into, e.g. "user_documents".
    chunks : List[Chunk]
        The chunks being stored. Their text and metadata (source, page,
        file_type, chunk_id) get saved alongside each embedding.
    embeddings : List[List[float]]
        One embedding vector per chunk, in the same order as `chunks`.
    """
    if not chunks:
        return

    collection = get_collection(collection_name)

    # ChromaDB requires a unique string ID per item. Combining the
    # source filename with chunk_id keeps IDs unique across uploads.
    ids = [f"{chunk.source}_{chunk.chunk_id}" for chunk in chunks]
    documents = [chunk.text for chunk in chunks]
    metadatas = [
        {
            "source": chunk.source,
            "page": chunk.page,
            "file_type": chunk.file_type,
            "chunk_id": chunk.chunk_id,
        }
        for chunk in chunks
    ]

    # upsert() (rather than add()) means re-uploading the same file
    # just overwrites its old chunks instead of erroring on duplicate IDs.
    collection.upsert(
        ids=ids,
        embeddings=embeddings,
        documents=documents,
        metadatas=metadatas,
    )


def query_collection(collection_name: str, query_embedding: List[float], top_k: int) -> Dict[str, Any]:
    """
    Find the stored chunks most similar to a query embedding.

    Returns
    -------
    dict
        ChromaDB's result dict with keys "documents", "metadatas", and
        "distances" -- each a list-of-lists (one inner list per query;
        we only ever pass one query embedding at a time here, so use
        index [0] to get the actual list of matches).
    """
    collection = get_collection(collection_name)

    # A brand-new or small collection can hold fewer items than top_k;
    # ChromaDB errors if asked for more results than exist, so cap it.
    count = collection.count()
    if count == 0:
        return {"documents": [[]], "metadatas": [[]], "distances": [[]]}

    return collection.query(
        query_embeddings=[query_embedding],
        n_results=min(top_k, count),
    )
