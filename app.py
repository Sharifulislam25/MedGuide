import streamlit as st
from src.document_loader import load_txt
from src.pdf_processor import load_pdf
from src.image_processor import load_image
from src.text_processor import clean_text
from src.chunker import chunk_documents
from src.embeddings import embed_texts
from src.vector_store import add_chunks, query_collection, get_collection
from src.config import USER_DOCUMENTS_COLLECTION

st.set_page_config(
    page_title="MedGuide",
    layout="wide"
)

st.title("MedGuide")
st.subheader("Medical Document Assistant")

st.write(
    "Upload a medical document to get started. "
    "(Currently supported: TXT, PDF, JPG, JPEG, PNG. More formats are added in later phases.)"
)

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
        # later step (display, chunking, embedding) works with the same
        # tidy text -- not raw PDF/OCR output.
        for doc in documents:
            doc.text = clean_text(doc.text)

        st.divider()
        st.subheader("Document Information")
        st.write(f"**File:** {documents[0].source}")
        st.write(f"**Type:** {documents[0].file_type}")
        st.write(f"**Pages:** {len(documents)}")

        # A PDF can now have a mix of normal pages and OCR'd pages, so
        # the top-level OCR status might not be a simple True/False.
        ocr_flags = {doc.metadata.get("ocr") for doc in documents}
        if len(ocr_flags) == 1:
            ocr_status = ocr_flags.pop()
        else:
            ocr_status = "Mixed (some pages required OCR)"
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

        # --- Chunking (Phase 7) ---
        # Split the cleaned text into overlapping chunks, ready for the
        # embedding step in Phase 8. Shown here as a debug view for now
        # so it's easy to sanity-check chunk sizes before we start
        # generating embeddings from them.
        chunks = chunk_documents(documents)

        st.divider()
        st.subheader("Chunking (debug view)")

        total_characters = sum(len(doc.text) for doc in documents)
        total_chunks = len(chunks)
        average_chunk_size = (
            sum(len(c.text) for c in chunks) / total_chunks if total_chunks > 0 else 0
        )

        col1, col2, col3 = st.columns(3)
        col1.metric("Total characters", total_characters)
        col2.metric("Total chunks", total_chunks)
        col3.metric("Average chunk size", f"{average_chunk_size:.0f}")

        with st.expander(f"View all {total_chunks} chunks"):
            for chunk in chunks:
                st.caption(
                    f"Chunk {chunk.chunk_id} — {chunk.source}, "
                    f"page {chunk.page} ({len(chunk.text)} chars)"
                )
                st.text(chunk.text)
                st.markdown("---")

        # --- Embeddings (Phase 8) ---
        # Turn each chunk's text into a vector. The first run of this
        # will be slow (downloading the model, ~90MB); after that it's
        # cached locally and loads in a couple of seconds.
        st.divider()
        st.subheader("Embeddings (debug view)")

        if chunks:
            with st.spinner("Loading embedding model and generating embeddings..."):
                chunk_texts = [chunk.text for chunk in chunks]
                embeddings = embed_texts(chunk_texts)

            embedding_dimension = len(embeddings[0]) if embeddings else 0
            st.write(f"**Embeddings generated:** {len(embeddings)}")
            st.write(f"**Embedding dimension:** {embedding_dimension}")

            with st.expander("Preview first embedding vector (first 8 of 384 numbers)"):
                st.code(str(embeddings[0][:8]))
        else:
            st.write("No chunks to embed.")

        # --- Vector Store (Phase 9) ---
        # Persist the chunks + embeddings in a local, on-disk ChromaDB
        # collection, so they survive between runs of the app instead
        # of needing to be re-embedded every time you open MedGuide.
        st.divider()
        st.subheader("Vector Store (debug view)")

        if chunks:
            add_chunks(USER_DOCUMENTS_COLLECTION, chunks, embeddings)
            collection = get_collection(USER_DOCUMENTS_COLLECTION)

            st.write(f"**Collection:** {USER_DOCUMENTS_COLLECTION}")
            st.write(f"**Total chunks stored (across all uploads so far):** {collection.count()}")

            # Round-trip sanity check: querying with a chunk's own
            # embedding should retrieve that exact same chunk back.
            # This is the "chunk -> embedding -> ChromaDB -> retrieve"
            # test the project plan calls for at this phase.
            check_result = query_collection(USER_DOCUMENTS_COLLECTION, embeddings[0], top_k=1)
            retrieved_docs = check_result["documents"][0]
            retrieved_text = retrieved_docs[0] if retrieved_docs else None

            if retrieved_text == chunks[0].text:
                st.success("Round-trip check passed: a stored chunk was retrieved correctly.")
            else:
                st.warning("Round-trip check: retrieved text didn't match exactly -- see details below.")

            with st.expander("Round-trip check details"):
                st.write("Chunk that was stored:")
                st.text(chunks[0].text[:300])
                st.write("Chunk retrieved back from ChromaDB:")
                st.text((retrieved_text or "(nothing retrieved)")[:300])
        else:
            st.write("No chunks to store.")

st.divider()
st.caption(
    "Educational information only. MedGuide does not provide a medical diagnosis."
)
