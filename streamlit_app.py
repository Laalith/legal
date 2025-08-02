import streamlit as st
import docx2txt
import spacy
import os
nlp = spacy.load("en_core_web_sm")

# Try to load SpaCy model safely
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    st.error("SpaCy model 'en_core_web_sm' not found. Please run: `python -m spacy download en_core_web_sm`")
    st.stop()

# Streamlit page setup
st.set_page_config(
    page_title="Clause-wise Legal Analyzer",
    layout="centered",
    initial_sidebar_state="auto"
)

st.title("📄 Clause-wise Legal Document Analyzer")

uploaded_file = st.file_uploader("Upload a .docx file", type="docx")

if uploaded_file is not None:
    # Extract and process text
    with st.spinner("Processing document..."):
        text = docx2txt.process(uploaded_file)

        if not text.strip():
            st.warning("No readable text found in the uploaded file.")
            st.stop()

        doc = nlp(text)

        # Split into sentences (each clause)
        st.subheader("📌 Extracted Clauses:")
        for i, sent in enumerate(doc.sents, 1):
            st.markdown(f"**Clause {i}:** {sent.text.strip()}")

        # Basic entity recognition (optional)
        if st.checkbox("🔍 Show named entities"):
            st.subheader("🔖 Named Entities Found:")
            for ent in doc.ents:
                st.write(f"{ent.text} — *{ent.label_}*")
