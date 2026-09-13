# 🩺 MedGuide V1 — Architecture & Technical Overview

MedGuide is a 100% local, privacy-first medical document assistant. It enables users to upload lab reports and medical records, inspect extracted data, search against personal records and verified medical knowledge, and receive structured educational interpretations without relying on cloud APIs, external LLMs, or active internet connectivity.

---

## 🏗️ System Architecture

MedGuide implements an offline, deterministic **Dual-Collection Retrieval-Augmented Generation (RAG)** pipeline powered by local embeddings, on-disk vector stores, and a rule-based medical interpretation engine.

```
                                  [ User / Clinician ]
                                           │
                        ┌──────────────────┴──────────────────┐
                        │                                     │
                 [ Document Upload ]                   [ Ask Question ]
                        │                                     │
           ┌────────────┴────────────┐                        │
           │  PDF / Image / TXT Doc  │                        │
           └────────────┬────────────┘                        │
                        │                                     │
           ┌────────────▼────────────┐                        ▼
           │ Ingestion & Processing  │              ┌──────────────────┐
           │ - pdfplumber (PDF text) │              │  Safety Layer 1  │
           │ - Tesseract OCR fallback│              │ Emergency Phrase │──[Emergency?]──► [Urgent Crisis Alert]
           │ - PIL / Grayscale prep  │              │    Detection     │                      (Bypasses RAG)
           │ - Regex text cleaning   │              └────────┬─────────┘
           └────────────┬────────────┘                       │ (Safe query)
                        │                                     │
           ┌────────────▼────────────┐                        │
           │ Character Chunker       │                        │
           │ (800 chars, 150 overlap)│                        │
           └────────────┬────────────┘                        │
                        │                                     │
           ┌────────────▼────────────┐              ┌─────────▼─────────┐
           │ SentenceTransformer     │              │ SentenceTransformer│
           │ all-MiniLM-L6-v2 (384d) │              │ Embed Query Vector│
           └────────────┬────────────┘              └─────────┬─────────┘
                        │                                     │
                        ▼                                     ▼
        ┌─────────────────────────────────────────────────────────────┐
        │                 ChromaDB (Persistent Vector DB)             │
        │                                                             │
        │  ┌───────────────────────────┐ ┌──────────────────────────┐ │
        │  │  user_documents           │ │  medical_knowledge       │ │
        │  │  (Uploaded reports)       │ │  (Built-in reference KB) │ │
        │  └─────────────┬─────────────┘ └────────────┬─────────────┘ │
        └────────────────┼────────────────────────────┼───────────────┘
                         │                            │
                         │ Dual Similarity Search     │ (Cosine / L2 distance
                         │ (Top-K = 3 per collection) │  threshold <= 1.5)
                         ▼                            ▼
                 ┌──────────────────────────────────────────┐
                 │           RagContext Assembler           │
                 │   - User chunks & confidence distance    │
                 │   - Medical reference chunks & distance  │
                 └────────────────────┬─────────────────────┘
                                      │
                                      ▼
                 ┌──────────────────────────────────────────┐
                 │         Rule-Based Response Engine       │
                 │ - Deterministic lab test matching        │
                 │ - Regex extraction of numerical values   │
                 │ - Reference range comparison (Low/Normal/│
                 │   High) without generative hallucination │
                 │ - Context synthesis & mandatory disclaimer│
                 └────────────────────┬─────────────────────┘
                                      │
                                      ▼
                 ┌──────────────────────────────────────────┐
                 │             Safety Layer 2               │
                 │     Diagnosis-Language Guard Validator   │
                 │  - Blocks "you have", "you are diabetic" │
                 │  - Replaces diagnostic phrasing with     │
                 │    cautious clinical guidance            │
                 └────────────────────┬─────────────────────┘
                                      │
                                      ▼
                         [ Streamlit Interactive UI ]
                       (Answer + Sources + Debug View)
```

---

## 🧩 Core Subsystems & Components

### 1. Document Ingestion & Optical Character Recognition (`src/`)
* [document_loader.py](file:///c:/Users/Shariful/Desktop/medguide-v1/src/document_loader.py): Handles plain-text documents and standardized `Document` data structures (`text`, `source`, `page`).
* [pdf_processor.py](file:///c:/Users/Shariful/Desktop/medguide-v1/src/pdf_processor.py): Employs `pdfplumber` for digital text extraction. If page text is below minimal thresholds or empty, falls back to rendering page pixmaps and OCR.
* [image_processor.py](file:///c:/Users/Shariful/Desktop/medguide-v1/src/image_processor.py) & [ocr.py](file:///c:/Users/Shariful/Desktop/medguide-v1/src/ocr.py): Utilizes `pytesseract` and `Pillow` to execute OCR with contrast adjustments and grayscale normalization.
* [text_processor.py](file:///c:/Users/Shariful/Desktop/medguide-v1/src/text_processor.py): Cleans redundant whitespaces, non-standard hyphens, and OCR artifacts.

### 2. Chunking & Local Embeddings
* [chunker.py](file:///c:/Users/Shariful/Desktop/medguide-v1/src/chunker.py): Implements sliding-window chunking (`800` characters, `150` character overlap) with tracking of source document origin and page indices.
* [embeddings.py](file:///c:/Users/Shariful/Desktop/medguide-v1/src/embeddings.py): Encodes text via `sentence-transformers/all-MiniLM-L6-v2`, producing 384-dimensional dense vectors on CPU with zero remote network calls.

### 3. Dual Vector Store (`database/chroma_db/`)
* [vector_store.py](file:///c:/Users/Shariful/Desktop/medguide-v1/src/vector_store.py): Interfaces with persistent `chromadb.PersistentClient`.
* **Two distinct collections:**
  * `user_documents`: User-uploaded reports and lab history.
  * `medical_knowledge`: Built-in reference documents covering blood pressure, cholesterol, complete blood count (CBC), glucose, hemoglobin, kidney function, liver function, and thyroid metrics.
* [knowledge_base.py](file:///c:/Users/Shariful/Desktop/medguide-v1/src/knowledge_base.py): Automatically indexes and maintains reference medical literature located in `knowledge/`.

### 4. Retrieval & Context Synthesis
* [retriever.py](file:///c:/Users/Shariful/Desktop/medguide-v1/src/retriever.py) & [rag.py](file:///c:/Users/Shariful/Desktop/medguide-v1/src/rag.py): Concurrently queries both collections, applies strict distance threshold filtering (`RELEVANCE_THRESHOLD = 1.5`), and builds unified `RagContext`.

### 5. Deterministic Response & Medical Safety
* [medical_rules.py](file:///c:/Users/Shariful/Desktop/medguide-v1/src/medical_rules.py): Maintains codified clinical parameters (normal low/high bounds, standard units, test aliases) and regex extractors for lab values.
* [response.py](file:///c:/Users/Shariful/Desktop/medguide-v1/src/response.py): Synthesizes retrieved evidence into structured answers without an generative LLM, eliminating hallucination risks regarding laboratory numbers or clinical facts.
* [safety.py](file:///c:/Users/Shariful/Desktop/medguide-v1/src/safety.py):
  * **Emergency Detection**: Catches acute crisis phrases (`chest pain`, `cannot breathe`, `stroke`, `severe bleeding`, `suicidal`, etc.) and short-circuits to an urgent crisis helpline warning.
  * **Diagnosis Guard**: Scans outputs to prevent definitive diagnostic assertions (e.g. `you have diabetes`, `you are diabetic`), falling back to consultative clinical guidance.

---

## 🔄 End-to-End Data Flow

1. **Upload**: User uploads a lab report (PDF/Image/TXT) via the Streamlit interface.
2. **Normalize**: File is parsed, OCR'd if scanned, sanitized, and chunked.
3. **Persist**: Chunks are embedded and stored in the `user_documents` ChromaDB collection.
4. **Query**: User asks a query in "Ask MedGuide".
5. **Screen**: Safety engine inspects the question for emergency indicators.
6. **Retrieve**: Dual similarity search retrieves the closest user records and reference literature.
7. **Interpret**: The engine identifies known lab markers, extracts numerical quantities, compares against reference baselines, and generates an educational summary.
8. **Sanitize & Render**: Output is verified by the diagnostic language filter, appends medical disclaimers, and displays final interpretations with source citations.

---

## 🚀 Getting Started

### Prerequisites
- Python 3.10+
- [Tesseract OCR](https://github.com/tesseract-ocr/tesseract) installed on system path (for image/scanned PDF processing).

### Installation & Execution
```bash
# Clone the repository
git clone https://github.com/Sharifulislam25/MedReportSum.git
cd MedReportSum

# Install dependencies
pip install -r requirements.txt

# Run the local application
streamlit run app.py
```
*(On first execution, MedGuide downloads the lightweight ~90MB embedding model and seeds the medical reference database automatically).*
