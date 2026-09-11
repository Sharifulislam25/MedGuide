import streamlit as st
from src.document_loader import load_txt

# st.set_page_config must be the first Streamlit command in the script.
st.set_page_config(
    page_title="MedGuide",
    layout="wide"
)

st.title("MedGuide")
st.subheader("Medical Document Assistant")

st.write(
    "Upload a medical document to get started. "
    "(Currently supported: TXT. More formats are added in later phases.)"
)

# type=["txt"] restricts the uploader to .txt files for now.
# We'll widen this list as PDF/image/MD support gets built in later phases.
uploaded_file = st.file_uploader("Choose a file", type=["txt"])

if uploaded_file is not None:
    document = load_txt(uploaded_file)

    st.divider()
    st.subheader("Document Information")
    st.write(f"**File:** {document.source}")
    st.write(f"**Type:** {document.file_type}")
    st.write(f"**Pages:** {document.page}")
    st.write(f"**OCR Used:** {document.metadata.get('ocr')}")

    st.divider()
    st.subheader("Extracted Text")
    st.text_area("Text content", document.text, height=300)

st.divider()
st.caption(
    "Educational information only. MedGuide does not provide a medical diagnosis."
)
