"""
chunker.py

Splits document text into smaller overlapping chunks, sized right for
embeddings later. Each chunk keeps track of which source file, page,
and file type it came from, plus a running chunk_id -- this is what
lets the app show "Source: blood_report.pdf, Page 2" next to an answer
in later phases.

The algorithm here is intentionally simple (word-based, fixed size with
overlap). It can be swapped for something smarter (sentence-aware,
semantic chunking) in V2 without changing how the rest of the app uses
Chunk objects.
"""

from dataclasses import dataclass
from typing import List

from src.document_loader import Document
from src.config import CHUNK_SIZE, CHUNK_OVERLAP


@dataclass
class Chunk:
    """A single chunk of text, ready to be embedded."""
    text: str
    source: str
    file_type: str
    page: int
    chunk_id: int


def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, chunk_overlap: int = CHUNK_OVERLAP) -> List[str]:
    """
    Split a single string of text into overlapping chunks.

    Works word-by-word (never splits a word in half) and measures size
    in characters. Once a chunk reaches chunk_size, it's closed off and
    the next chunk starts by re-including the last chunk_overlap
    characters' worth of words, so context isn't lost at the boundary.

    Parameters
    ----------
    text : str
        The (already cleaned) text to split.
    chunk_size : int
        Target maximum size of each chunk, in characters.
    chunk_overlap : int
        How much of the previous chunk (in characters) to repeat at the
        start of the next chunk.

    Returns
    -------
    List[str]
        The text split into chunks. Returns an empty list for empty input.
    """
    words = text.split()
    if not words:
        return []

    chunks = []
    current_words = []
    current_length = 0

    for word in words:
        current_words.append(word)
        current_length += len(word) + 1  # +1 accounts for the joining space

        if current_length >= chunk_size:
            chunks.append(" ".join(current_words))

            # Build the overlap for the next chunk: walk backwards from
            # the end of this chunk until we've collected roughly
            # chunk_overlap characters' worth of words.
            overlap_words = []
            overlap_length = 0
            for word_from_end in reversed(current_words):
                overlap_length += len(word_from_end) + 1
                overlap_words.insert(0, word_from_end)
                if overlap_length >= chunk_overlap:
                    break

            current_words = overlap_words
            current_length = overlap_length

    # Add whatever's left as the final chunk.
    if current_words:
        chunks.append(" ".join(current_words))

    return chunks


def chunk_documents(
    documents: List[Document],
    chunk_size: int = CHUNK_SIZE,
    chunk_overlap: int = CHUNK_OVERLAP
) -> List[Chunk]:
    """
    Chunk a list of Documents (e.g. one per PDF page) into Chunk objects,
    carrying over each Document's source/page/file_type and adding a
    chunk_id that's unique across the whole list.

    Parameters
    ----------
    documents : List[Document]
    chunk_size : int
    chunk_overlap : int

    Returns
    -------
    List[Chunk]
    """
    all_chunks = []
    next_chunk_id = 0

    for document in documents:
        pieces = chunk_text(document.text, chunk_size, chunk_overlap)
        for piece in pieces:
            all_chunks.append(
                Chunk(
                    text=piece,
                    source=document.source,
                    file_type=document.file_type,
                    page=document.page,
                    chunk_id=next_chunk_id
                )
            )
            next_chunk_id += 1

    return all_chunks
