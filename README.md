# MedGuide V1

A local, offline medical document assistant. Runs entirely on your own
computer -- no cloud APIs, no API keys, no internet needed after the
one-time model download.

## Status: V1 complete

- [x] Phase 1 — Streamlit app skeleton
- [x] Phase 2 — TXT upload
- [x] Phase 3 — PDF extraction
- [x] Phase 4 — Image OCR
- [x] Phase 5 — Scanned-PDF OCR fallback
- [x] Phase 6 — Text cleaning
- [x] Phase 7 — Chunking
- [x] Phase 8 — Local embeddings
- [x] Phase 9 — ChromaDB
- [x] Phase 10 — Medical knowledge base
- [x] Phase 11–13 — Retrieval + RAG pipeline
- [x] Phase 14 — Response engine
- [x] Phase 15 — Safety rules (emergency detection + diagnosis-language guard)
- [x] Phase 16 — Full UI

## What it does

Upload a medical document (PDF, scanned PDF, JPG/JPEG/PNG, or TXT).
MedGuide extracts the text (OCR'ing scanned pages/images locally via
Tesseract), cleans it, splits it into chunks, embeds them with a local
Sentence Transformers model, and stores them in a persistent local
ChromaDB database.

Then ask a question in "Ask MedGuide." It searches both your uploaded
documents and a small built-in medical knowledge base (glucose, blood
pressure, hemoglobin, cholesterol, CBC, liver, kidney, thyroid), and
answers using rule-based templates -- no LLM anywhere in this project.
Answers always include a safety disclaimer and never claim to diagnose;
emergency-sounding questions get redirected to urgent guidance instead
of a normal answer.

## Run it

```bash
pip install -r requirements.txt
streamlit run app.py
```

First run will download the embedding model (~90MB) -- needs internet
once, then works fully offline. First run also builds the medical
knowledge base automatically.

## Project structure

```
medguide-v1/
├── app.py                  # Streamlit UI, wires everything together
├── requirements.txt
├── knowledge/               # Reference texts for the medical knowledge base
├── database/chroma_db/      # Persistent vector storage (created on first run)
└── src/
    ├── config.py            # All settings in one place
    ├── document_loader.py    # TXT loading + shared Document type
    ├── pdf_processor.py      # PDF text extraction + scanned-page OCR fallback
    ├── image_processor.py    # Image preprocessing for OCR
    ├── ocr.py                # Tesseract OCR wrapper
    ├── text_processor.py     # Text cleaning
    ├── chunker.py            # Splits text into overlapping chunks
    ├── embeddings.py         # Local Sentence Transformers embeddings
    ├── vector_store.py       # ChromaDB wrapper
    ├── knowledge_base.py     # Builds/rebuilds the medical knowledge collection
    ├── retriever.py          # Searches one collection for relevant chunks
    ├── rag.py                # Combines both collections' retrieval results
    ├── medical_rules.py      # Known test names, units, and reference ranges
    ├── response.py           # Generates the final written answer (no LLM)
    └── safety.py             # Emergency detection + diagnosis-language guard
```

## V2 ideas (not built, by design)

FastAPI instead of Streamlit, PostgreSQL+pgvector instead of ChromaDB,
hybrid search + reranking, a real local LLM for the response engine,
structured extraction instead of regex-based value matching.
