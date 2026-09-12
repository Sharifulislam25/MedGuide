"""
config.py

Central place for settings used across the app. As later phases add
embeddings, ChromaDB, and retrieval, add those settings here too rather
than scattering constants through individual files -- this is what
makes V2 upgrades easier later.
"""

# --- Chunking ---
# Measured in characters (not words/tokens) for now -- simplest thing
# that works for a beginner project. CHUNK_OVERLAP lets consecutive
# chunks share some text, so a sentence that gets cut off at a chunk
# boundary still appears in full in the next chunk too.
CHUNK_SIZE = 800
CHUNK_OVERLAP = 150

# --- Embeddings ---
# "all-MiniLM-L6-v2" is small (~90MB), runs fine on CPU, needs no API
# key, and produces 384-dimensional vectors -- a good default for a
# beginner's computer. The exact same model must be used for both
# document chunks and user questions (Phase 11+), so it's defined once
# here rather than being hard-coded in multiple files.
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
