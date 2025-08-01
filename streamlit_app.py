import streamlit as st
import pdfplumber
from transformers import pipeline

# Load Hugging Face model for simplification
simplifier = pipeline("text2text-generation", model="t5-small")

st.title("📜 Clause-wise Legal Document Analyzer")

# Upload PDF
uploaded_file = st.file_uploader("Upload a legal document (PDF)", type=["pdf"])

def extract_clauses_from_pdf(file):
    clauses = []
    with pdfplumber.open(file) as pdf:
        for page in pdf.pages:
            text = page.extract_text()
            if text:
                for clause in text.split('\n'):
                    if len(clause.strip()) > 30:
                        clauses.append(clause.strip())
    return clauses

if uploaded_file:
    st.success("✅ File uploaded successfully!")
    clauses = extract_clauses_from_pdf(uploaded_file)

    st.subheader("🔍 Extracted Clauses")
    for i, clause in enumerate(clauses):
        with st.expander(f"Clause {i+1}"):
            st.markdown(clause)
            if st.button(f"Simplify Clause {i+1}", key=i):
                with st.spinner("Simplifying..."):
                    simplified = simplifier(f"simplify: {clause}", max_length=150, do_sample=False)[0]['generated_text']
                    st.markdown(f"**📝 Simplified:** {simplified}")
