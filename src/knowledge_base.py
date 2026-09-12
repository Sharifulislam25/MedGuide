"""
knowledge_base.py

Loads MedGuide's own trusted educational reference texts (the files in
knowledge/) into the "medical_knowledge" ChromaDB collection -- kept
completely separate from whatever the user uploads (the
"user_documents" collection from Phase 9).

This only needs to run once. After that, ChromaDB's persistent storage
already has it, and future app runs skip straight past it.
"""

import os
from src.document_loader import Document
from src.text_processor import clean_text
from src.chunker import chunk_documents
from src.embeddings import embed_texts
from src.vector_store import add_chunks, get_collection, delete_collection
from src.config import MEDICAL_KNOWLEDGE_COLLECTION

KNOWLEDGE_FOLDER = "knowledge"


def build_medical_knowledge_base() -> int:
    """
    Read every .txt file in the knowledge/ folder, clean + chunk + embed
    it, and store the result in the medical_knowledge collection.

    Returns
    -------
    int
        The number of chunks stored (0 if the folder is missing/empty).
    """
    if not os.path.isdir(KNOWLEDGE_FOLDER):
        return 0

    documents = []
    for filename in sorted(os.listdir(KNOWLEDGE_FOLDER)):
        if not filename.endswith(".txt"):
            continue

        filepath = os.path.join(KNOWLEDGE_FOLDER, filename)
        with open(filepath, "r", encoding="utf-8") as f:
            raw_text = f.read()

        documents.append(
            Document(
                text=clean_text(raw_text),
                source=filename,
                file_type="txt",
                page=1,
                metadata={"ocr": False}
            )
        )

    if not documents:
        return 0

    chunks = chunk_documents(documents)
    embeddings = embed_texts([chunk.text for chunk in chunks])
    add_chunks(MEDICAL_KNOWLEDGE_COLLECTION, chunks, embeddings)

    return len(chunks)


def medical_knowledge_base_is_empty() -> bool:
    """Check whether the medical_knowledge collection has anything stored yet."""
    collection = get_collection(MEDICAL_KNOWLEDGE_COLLECTION)
    return collection.count() == 0


def rebuild_medical_knowledge_base() -> int:
    """
    Delete and rebuild the medical_knowledge collection from scratch.

    Use this after adding or editing files in knowledge/ -- the
    "only builds when empty" check in app.py won't notice new files on
    its own, so this gives an explicit way to force a fresh rebuild
    without touching the user_documents collection at all.

    Returns
    -------
    int
        The number of chunks stored after rebuilding.
    """
    delete_collection(MEDICAL_KNOWLEDGE_COLLECTION)
    return build_medical_knowledge_base()
