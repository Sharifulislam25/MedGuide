"""
rag.py

Combines retrieval from both the user's own documents and MedGuide's
medical knowledge base into a single RagContext -- everything the
response engine (Phase 14) will need to answer a question and cite
its sources, whether that answer comes from what's in the user's
report, what MedGuide knows generally, or both.
"""

from dataclasses import dataclass
from typing import List

from src.retriever import (
    retrieve_from_user_documents,
    retrieve_from_medical_knowledge,
    RetrievedChunk,
)
from src.config import TOP_K


@dataclass
class RagContext:
    """Everything retrieved for one question, split by where it came from."""
    query: str
    user_document_chunks: List[RetrievedChunk]
    medical_knowledge_chunks: List[RetrievedChunk]

    def all_chunks(self) -> List[RetrievedChunk]:
        """Both sets of chunks combined, for convenience."""
        return self.user_document_chunks + self.medical_knowledge_chunks

    def is_empty(self) -> bool:
        return not self.user_document_chunks and not self.medical_knowledge_chunks


def build_context(query: str, top_k: int = TOP_K) -> RagContext:
    """
    Run both retrieval searches for a question and package the results
    together.

    Parameters
    ----------
    query : str
        The user's question.
    top_k : int
        How many chunks to retrieve from each collection.

    Returns
    -------
    RagContext
    """
    user_chunks = retrieve_from_user_documents(query, top_k)
    knowledge_chunks = retrieve_from_medical_knowledge(query, top_k)

    return RagContext(
        query=query,
        user_document_chunks=user_chunks,
        medical_knowledge_chunks=knowledge_chunks,
    )
