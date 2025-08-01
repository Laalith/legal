import streamlit as st
import PyPDF2
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM, AutoModelForTokenClassification
import torch
import re

# Load models
@st.cache_resource
def load_models():
    simplifier_model = AutoModelForSeq2SeqLM.from_pretrained("t5-small")
    simplifier_tokenizer = AutoTokenizer.from_pretrained("t5-small")

    ner_pipeline = pipeline("ner", model="dslim/bert-base-NER", grouped_entities=True)
    return simplifier_model, simplifier_tokenizer, ner_pipeline

simplifier_model, simplifier_tokenizer, ner_pipeline = load_models()

# Extract text from uploaded PDF
def extract_text_from_pdf(pdf_file):
    reader = PyPDF2.PdfReader(pdf_file)
    text = ""
    for page in reader.pages:
        text += page.extract_text() + "\n"
    return text

# Extract clauses (simple split by newlines or semicolons)
def extract_clauses(text):
    clauses = re.split(r'\n+|;', text)
    return [clause.strip() for clause in clauses if clause.strip()]

# Simplify a clause
def simplify_clause(clause):
    input_text = "simplify: " + clause
    inputs = simplifier_tokenizer.encode(input_text, return_tensors="pt", max_length=512, truncation=True)
    outputs = simplifier_model.generate(inputs, max_length=150, num_beams=4, early_stopping=True)
    return simplifier_tokenizer.decode(outputs[0], skip_special_tokens=True)

# Display title
st.set_page_config(page_title="Clause-wise Legal Document Analyzer", layout="wide")
st.title("📄 Clause-wise Legal Document Analyzer")

# File upload
uploaded_file = st.file_uploader("Upload a legal document (PDF)", type=["pdf"])

if uploaded_file:
    with st.spinner("Extracting text from document..."):
        full_text = extract_text_from_pdf(uploaded_file)
        clauses = extract_clauses(full_text)

    st.success("Document processed. Number of clauses: {}".format(len(clauses)))
    
    for i, clause in enumerate(clauses):
        with st.expander(f"Clause {i+1}:"):
            st.markdown(f"**Original:** {clause}")

            simplified = simplify_clause(clause)
            st.markdown(f"**Simplified:** {simplified}")

            entities = ner_pipeline(clause)
            if entities:
                st.markdown("**Entities Found:**")
                for ent in entities:
                    st.markdown(f"- {ent['entity_group']}: `{ent['word']}`")
