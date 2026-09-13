import streamlit as st
from src.document_loader import load_txt
from src.pdf_processor import load_pdf
from src.image_processor import load_image
from src.text_processor import clean_text
from src.chunker import chunk_documents
from src.embeddings import embed_texts, embed_query
from src.vector_store import add_chunks, query_collection, get_collection
from src.config import USER_DOCUMENTS_COLLECTION, MEDICAL_KNOWLEDGE_COLLECTION
from src.knowledge_base import (
    build_medical_knowledge_base,
    medical_knowledge_base_is_empty,
    rebuild_medical_knowledge_base,
)
from src.rag import build_context
from src.response import generate_response, get_sources
from src.safety import check_for_emergency, get_emergency_response, enforce_safe_language

st.set_page_config(
    page_title="MedGuide",
    page_icon="🩺",
    layout="wide"
)

# ============================================================
# Sidebar: developer tools -- not part of the everyday user flow,
# kept separate so the main page stays simple and professional.
# ============================================================
with st.sidebar:
    st.header("🛠️ Developer Tools")
    st.caption("Rebuilds the reference knowledge base after editing files in knowledge/.")
    if st.button("🔄 Rebuild medical knowledge base"):
        with st.spinner("Rebuilding medical knowledge base..."):
            num_rebuilt = rebuild_medical_knowledge_base()
        st.success(f"Rebuilt: {num_rebuilt} chunks stored.")

    st.divider()
    st.subheader("Test the knowledge base")
    st.caption("Search medical_knowledge directly with any question.")
    test_query = st.text_input("Question to search for")

    col_search, col_clear = st.columns(2)

    if col_search.button("🔍 Search") and test_query:
        with st.spinner("Searching..."):
            query_vector = embed_query(test_query)
            search_results = query_collection(MEDICAL_KNOWLEDGE_COLLECTION, query_vector, top_k=3)
        st.session_state["kb_search_query"] = test_query
        st.session_state["kb_search_results"] = search_results

    if col_clear.button("🧹 Clear"):
        st.session_state.pop("kb_search_query", None)
        st.session_state.pop("kb_search_results", None)

    if "kb_search_results" in st.session_state:
        st.caption(f'Results for: "{st.session_state["kb_search_query"]}"')
        results = st.session_state["kb_search_results"]
        result_docs = results["documents"][0]
        result_metas = results["metadatas"][0]
        result_distances = results["distances"][0]

        if not result_docs:
            st.write("No results. Has the knowledge base been built yet?")
        else:
            for doc_text, meta, distance in zip(result_docs, result_metas, result_distances):
                st.write(f"**{meta['source']}** (distance: {distance:.3f})")
                st.caption(doc_text[:200] + "...")

# ============================================================
# Header
# ============================================================
st.title("🩺 MedGuide")
st.subheader("Medical Document Assistant")
st.write(
    "Upload a medical document to get started. "
    "(Currently supported: TXT, PDF, JPG, JPEG, PNG.)"
)

# Load the medical knowledge base once, ever -- persisted after that,
# so this check is instant on every run after the very first one.
if medical_knowledge_base_is_empty():
    with st.spinner("Setting up the medical knowledge base (first run only)..."):
        num_knowledge_chunks = build_medical_knowledge_base()
    st.info(
        f"Medical knowledge base loaded: {num_knowledge_chunks} chunks "
        "covering glucose, blood pressure, hemoglobin, cholesterol, CBC, "
        "liver, kidney, and thyroid function."
    )

# ============================================================
# Upload + process a document
# ============================================================
uploaded_file = st.file_uploader(
    "Choose a file", type=["txt", "pdf", "jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    file_name = uploaded_file.name.lower()
    documents = []

    if file_name.endswith(".txt"):
        documents = [load_txt(uploaded_file)]

    elif file_name.endswith(".pdf"):
        with st.spinner("Reading PDF (scanned pages are OCR'd automatically)..."):
            try:
                documents = load_pdf(uploaded_file)
            except RuntimeError as error:
                st.error(str(error))

    elif file_name.endswith((".jpg", ".jpeg", ".png")):
        st.image(uploaded_file, caption="Uploaded Image", use_container_width=True)
        uploaded_file.seek(0)  # reset the read pointer after the preview above

        with st.spinner("Running OCR on the image..."):
            try:
                documents = [load_image(uploaded_file)]
            except RuntimeError as error:
                st.error(str(error))

    else:
        st.error("Unsupported file type. Please upload PDF, JPG, JPEG, PNG, TXT, or MD.")

    if documents:
        # Clean the extracted text right after extraction, so every
        # later step works with the same tidy text, not raw output.
        for doc in documents:
            doc.text = clean_text(doc.text)

        st.divider()
        st.subheader("Document Information")
        st.write(f"**File:** {documents[0].source}")
        st.write(f"**Type:** {documents[0].file_type}")
        st.write(f"**Pages:** {len(documents)}")

        ocr_flags = {doc.metadata.get("ocr") for doc in documents}
        ocr_status = ocr_flags.pop() if len(ocr_flags) == 1 else "Mixed (some pages required OCR)"
        st.write(f"**OCR Used:** {ocr_status}")

        st.divider()
        st.subheader("Extracted Text")

        combined_text = "".join(doc.text for doc in documents)
        if all(doc.metadata.get("ocr") for doc in documents) and len(combined_text.strip()) < 5:
            st.warning(
                "I couldn't extract enough text from this document. "
                "Please try a clearer image or a text-based PDF."
            )

        if len(documents) == 1:
            st.text_area("Text content", documents[0].text, height=300)
        else:
            for doc in documents:
                label = f"Page {doc.page}"
                if doc.metadata.get("ocr"):
                    label += " (OCR)"
                with st.expander(label):
                    st.text_area(
                        f"Page {doc.page} text",
                        doc.text,
                        height=200,
                        key=f"page_{doc.page}"
                    )

        # --- Behind the scenes: chunk, embed, store (Phases 7-9) ---
        # Everyday users see one clean status line; the step-by-step
        # numbers live in the collapsed developer expander below.
        chunks = chunk_documents(documents)
        embeddings = []

        if chunks:
            with st.spinner("Processing document (chunking, embedding, storing)..."):
                chunk_texts = [chunk.text for chunk in chunks]
                embeddings = embed_texts(chunk_texts)
                add_chunks(USER_DOCUMENTS_COLLECTION, chunks, embeddings)

            collection = get_collection(USER_DOCUMENTS_COLLECTION)
            st.success(
                f"✅ Processed into {len(chunks)} chunks and added to your document "
                f"library ({collection.count()} chunks total stored across all uploads)."
            )
        else:
            st.warning("No text was found to process into the document library.")

        with st.expander("🛠️ Developer debug info (chunking / embeddings / vector store)"):
            total_characters = sum(len(doc.text) for doc in documents)
            total_chunks = len(chunks)
            average_chunk_size = (
                sum(len(c.text) for c in chunks) / total_chunks if total_chunks > 0 else 0
            )

            col1, col2, col3 = st.columns(3)
            col1.metric("Total characters", total_characters)
            col2.metric("Total chunks", total_chunks)
            col3.metric("Average chunk size", f"{average_chunk_size:.0f}")

            if chunks:
                st.write("**Chunks:**")
                for chunk in chunks:
                    st.caption(
                        f"Chunk {chunk.chunk_id} — {chunk.source}, "
                        f"page {chunk.page} ({len(chunk.text)} chars)"
                    )
                    st.text(chunk.text)
                    st.markdown("---")

                embedding_dimension = len(embeddings[0]) if embeddings else 0
                st.write(f"**Embeddings generated:** {len(embeddings)}")
                st.write(f"**Embedding dimension:** {embedding_dimension}")
                st.code(str(embeddings[0][:8]) if embeddings else "(none)")

                # Round-trip sanity check: querying with a chunk's own
                # embedding should retrieve that exact same chunk back.
                check_result = query_collection(USER_DOCUMENTS_COLLECTION, embeddings[0], top_k=1)
                retrieved_docs = check_result["documents"][0]
                retrieved_text = retrieved_docs[0] if retrieved_docs else None
                if retrieved_text == chunks[0].text:
                    st.success("Round-trip check passed: a stored chunk was retrieved correctly.")
                else:
                    st.warning("Round-trip check: retrieved text didn't match exactly.")

# ============================================================
# Ask MedGuide (Phases 11-15: retrieval + RAG + response + safety)
# ============================================================
st.divider()
st.subheader("Ask MedGuide")

question = st.text_input("What would you like to know?")

col_ask, col_clear_answer = st.columns(2)

if col_ask.button("Ask") and question:
    if check_for_emergency(question):
        # An emergency-associated phrase was detected in the question
        # itself. Skip the normal RAG pipeline entirely -- a calm,
        # textbook-style answer is the wrong response here.
        st.session_state["emergency"] = True
        st.session_state.pop("rag_context", None)
        st.session_state.pop("answer_text", None)
        st.session_state.pop("sources", None)
    else:
        st.session_state["emergency"] = False
        with st.spinner("Searching your documents and the medical knowledge base..."):
            rag_context = build_context(question)
            answer_text = generate_response(rag_context)
            # Safety-net check: even though response.py's templates are
            # written to avoid it, verify the final text never contains
            # definitive-diagnosis language before showing it.
            answer_text = enforce_safe_language(answer_text)
            sources = get_sources(rag_context)
        st.session_state["rag_context"] = rag_context
        st.session_state["answer_text"] = answer_text
        st.session_state["sources"] = sources

if col_clear_answer.button("🧹 Clear"):
    st.session_state.pop("rag_context", None)
    st.session_state.pop("answer_text", None)
    st.session_state.pop("sources", None)
    st.session_state.pop("emergency", None)

if st.session_state.get("emergency"):
    st.error(get_emergency_response())

elif "answer_text" in st.session_state:
    rag_context = st.session_state["rag_context"]
    st.caption(f'For: "{rag_context.query}"')

    st.subheader("Answer")
    st.write(st.session_state["answer_text"])

    st.subheader("Sources")
    sources = st.session_state["sources"]
    if sources:
        for source in sources:
            st.write(f"- {source}")
    else:
        st.write("No sources -- nothing relevant was found.")

    with st.expander("🛠️ Developer debug info (retrieved context)"):
        st.write("**From your uploaded documents:**")
        if rag_context.user_document_chunks:
            for chunk in rag_context.user_document_chunks:
                st.write(f"- {chunk.source}, page {chunk.page} (distance: {chunk.distance:.3f})")
                st.caption(chunk.text[:250] + ("..." if len(chunk.text) > 250 else ""))
        else:
            st.caption("Nothing relevant found in your uploaded documents.")

        st.write("**From the medical knowledge base:**")
        if rag_context.medical_knowledge_chunks:
            for chunk in rag_context.medical_knowledge_chunks:
                st.write(f"- {chunk.source} (distance: {chunk.distance:.3f})")
                st.caption(chunk.text[:250] + ("..." if len(chunk.text) > 250 else ""))
        else:
            st.caption("Nothing relevant found in the medical knowledge base.")

# ============================================================
# Footer
# ============================================================
st.divider()
st.caption(
    "Educational information only. MedGuide does not provide a medical diagnosis."
)
