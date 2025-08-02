import streamlit as st
import pdfplumber
import docx2txt
import os
import spacy
import pandas as pd
from transformers import pipeline
from spacy.cli import download

# Ensure the model is downloaded
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    download("en_core_web_sm")
    nlp = spacy.load("en_core_web_sm")

# Load spaCy model
nlp = spacy.load("en_core_web_sm")

# Load transformers model (small for memory efficiency)
simplifier = pipeline("text2text-generation", model="t5-small")

# Background color using CSS
st.markdown("""
    <style>
    .stApp {
        background-color: #000000;
        color: #FFFFFF;
    }
    </style>
    """, unsafe_allow_html=True)

st.title("🧠 Legal Document Analyzer")

uploaded_file = st.file_uploader("Upload legal document (PDF, DOCX, TXT)", type=["pdf", "docx", "txt"])

def extract_text(file):
    if file.name.endswith(".pdf"):
        with pdfplumber.open(file) as pdf:
            return "\n".join(page.extract_text() for page in pdf.pages if page.extract_text())
    elif file.name.endswith(".docx"):
        return docx2txt.process(file)
    elif file.name.endswith(".txt"):
        return str(file.read(), "utf-8")
    return ""

if uploaded_file:
    text = extract_text(uploaded_file)
    st.subheader("📄 Extracted Text")
    st.text_area("", text, height=200)

    if st.button("Simplify Clauses"):
        simplified = simplifier(text[:512])[0]['generated_text']  # Limit input for small model
        st.subheader("🧾 Simplified Clauses")
        st.write(simplified)

    if st.button("Named Entity Recognition"):
        doc = nlp(text)
        entities = [(ent.text, ent.label_) for ent in doc.ents]
        st.subheader("🔍 Named Entities")
        st.dataframe(pd.DataFrame(entities, columns=["Entity", "Label"]))

    if st.button("Extract Clauses"):
        clauses = [line.strip() for line in text.split(".") if len(line.strip()) > 20]
        st.subheader("📜 Extracted Clauses")
        for clause in clauses[:10]:
            st.markdown(f"- {clause}")

    if st.button("Classify Document Type"):
        if "lease" in text.lower():
            doc_type = "Lease Agreement"
        elif "nda" in text.lower() or "non-disclosure" in text.lower():
            doc_type = "Non-Disclosure Agreement"
        elif "employment" in text.lower():
            doc_type = "Employment Contract"
        else:
            doc_type = "Service Agreement or Other"
        st.subheader("📂 Document Type")
        st.write(doc_type)
