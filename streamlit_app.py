import os
import streamlit as st
from transformers import pipeline
import pdfplumber
import docx2txt
import tempfile

# Disable TensorFlow to avoid import errors
os.environ["TRANSFORMERS_NO_TF"] = "1"

# Initialize pipelines once (load lightweight models or placeholders)
@st.cache_resource
def load_models():
    simplifier = pipeline("text2text-generation", model="t5-small")  # Clause Simplification
    ner = pipeline("ner", aggregation_strategy="simple")            # Named Entity Recognition
    classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")  # Document Type Classification
    # You can add more or use custom models for clause extraction later
    return simplifier, ner, classifier

simplifier, ner, classifier = load_models()

# Helper functions to extract text from files
def extract_text(file):
    if file.type == "application/pdf":
        with pdfplumber.open(file) as pdf:
            return "\n".join(page.extract_text() or "" for page in pdf.pages)
    elif file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
        return docx2txt.process(file)
    elif file.type == "text/plain":
        return file.getvalue().decode("utf-8")
    else:
        return ""

# Clause extraction example (simple split by semicolons or newlines)
def extract_clauses(text):
    clauses = [c.strip() for c in text.split('\n') if c.strip()]
    return clauses if clauses else [text]

# UI customization: background color
def set_background_color():
    st.markdown(
        """
        <style>
        .stApp {
            background-color: #f0f2f6;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

set_background_color()

st.title("Legal Document Analyzer Prototype")

uploaded_file = st.file_uploader("Upload a legal document (PDF, DOCX, TXT)", 
                                 type=["pdf", "docx", "txt"])

if uploaded_file:
    text = extract_text(uploaded_file)
    
    if not text.strip():
        st.warning("Couldn't extract any text from the uploaded document.")
    else:
        st.subheader("Original Document Text")
        st.write(text[:3000] + ("..." if len(text) > 3000 else ""))  # show first 3000 chars only
        
        # Document Type Classification
        candidate_labels = ["NDA", "Lease", "Employment Contract", "Service Agreement", "Other"]
        classification = classifier(text, candidate_labels)
        st.subheader("Document Classification")
        st.write(classification)
        
        # Clause Extraction
        st.subheader("Clause Extraction")
        clauses = extract_clauses(text)
        st.write(f"Extracted {len(clauses)} clauses.")
        
        # Show clauses with option to simplify and do NER
        for i, clause in enumerate(clauses[:10]):  # limit to first 10 clauses for speed
            st.markdown(f"**Clause {i+1}:**")
            st.write(clause)
            
            if st.button(f"Simplify Clause {i+1}", key=f"simplify_{i}"):
                simplified = simplifier(clause, max_length=150)[0]['generated_text']
                st.info(f"Simplified: {simplified}")
            
            if st.button(f"Extract Entities from Clause {i+1}", key=f"ner_{i}"):
                entities = ner(clause)
                st.json(entities)
