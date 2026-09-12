"""
retriever.py

Searches either the user_documents or medical_knowledge collection for
chunks relevant to a question, and returns them as a list of
RetrievedChunk objects -- a simple, consistent shape the rest of the
app (and the response engine, in Phase 14) can use without needing to
know anything about ChromaDB's raw result format.
"""

from dataclasses import dataclass
from typing import List

from src.embeddings import embed_query
from src.vector_store import query_collection
from src.config import (
    USER_DOCUMENTS_COLLECTION,
    MEDICAL_KNOWLEDGE_COLLECTION,
    TOP_K,
    RELEVANCE_THRESHOLD,
)


@dataclass
class RetrievedChunk:
    """One retrieved chunk, with enough info to show a source and judge relevance."""
    text: str
    source: str
    page: int
    distance: float
    collection: str  # which collection this came from, e.g. "user_documents"


def _run_query(
    collection_name: str,
    query: str,
    top_k: int,
    max_distance: float = RELEVANCE_THRESHOLD,
) -> List[RetrievedChunk]:
    """
    Shared logic used by both retrieval functions below: embed the
    question, search one collection, wrap the raw results into
    RetrievedChunk objects, and drop anything not relevant enough
    (distance above max_distance) to be worth showing or using.
    """
    query_vector = embed_query(query)
    raw_results = query_collection(collection_name, query_vector, top_k)

    documents = raw_results["documents"][0]
    metadatas = raw_results["metadatas"][0]
    distances = raw_results["distances"][0]

    chunks = []
    for doc_text, meta, distance in zip(documents, metadatas, distances):
        if distance > max_distance:
            continue
        chunks.append(
            RetrievedChunk(
                text=doc_text,
                source=meta.get("source", "unknown"),
                page=meta.get("page", 1),
                distance=distance,
                collection=collection_name,
            )
        )
    return chunks


def retrieve_from_user_documents(query: str, top_k: int = TOP_K) -> List[RetrievedChunk]:
    """Search the user's own uploaded documents for chunks relevant to `query`."""
    return _run_query(USER_DOCUMENTS_COLLECTION, query, top_k)


def retrieve_from_medical_knowledge(query: str, top_k: int = TOP_K) -> List[RetrievedChunk]:
    """Search MedGuide's trusted reference material for chunks relevant to `query`."""
    return _run_query(MEDICAL_KNOWLEDGE_COLLECTION, query, top_k)
