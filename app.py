import streamlit as st
from src.document_loader import load_txt
from src.pdf_processor import load_pdf
from src.image_processor import load_image

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
        documents = load_pdf(uploaded_file)

    elif file_name.endswith((".jpg", ".jpeg", ".png")):
        # Show the original image so the user can compare it against
        # what OCR actually managed to read.
        st.image(uploaded_file, caption="Uploaded Image", use_container_width=True)
        uploaded_file.seek(0)  # reset the read pointer after the preview above

        with st.spinner("Running OCR on the image..."):
            try:
                documents = [load_image(uploaded_file)]
            except RuntimeError as error:
                st.error(str(error))

    else:
        # Safety net for later phases when more formats exist.
        st.error("Unsupported file type. Please upload PDF, JPG, JPEG, PNG, TXT, or MD.")

    if documents:
        st.divider()
        st.subheader("Document Information")
        st.write(f"**File:** {documents[0].source}")
        st.write(f"**Type:** {documents[0].file_type}")
        st.write(f"**Pages:** {len(documents)}")
        st.write(f"**OCR Used:** {documents[0].metadata.get('ocr')}")

        st.divider()
        st.subheader("Extracted Text")

        if documents[0].metadata.get("ocr") and len(documents[0].text.strip()) < 5:
            st.warning(
                "I couldn't extract enough text from this image. "
                "Please try a clearer photo or scan."
            )

        if len(documents) == 1:
            st.text_area("Text content", documents[0].text, height=300)
        else:
            for doc in documents:
                with st.expander(f"Page {doc.page}"):
                    st.text_area(
                        f"Page {doc.page} text",
                        doc.text,
                        height=200,
                        key=f"page_{doc.page}"
                    )

st.divider()
st.caption(
    "Educational information only. MedGuide does not provide a medical diagnosis."
)
