import streamlit as st
import pdfplumber
from transformers import pipeline
import docx2txt

# Initialize pipelines with caching to avoid reload delays
@st.cache_resource(show_spinner=False)
def load_simplifier():
    return pipeline("text2text-generation", model="t5-small")

@st.cache_resource(show_spinner=False)
def load_ner():
    return pipeline("ner", grouped_entities=True, model="dslim/bert-base-NER")

simplifier = load_simplifier()
ner_pipeline = load_ner()

# Utility functions
def extract_text_from_pdf(file):
    text = ""
    with pdfplumber.open(file) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
    return text

def extract_text_from_docx(file):
    # docx2txt expects a path or file-like object
    return docx2txt.process(file)

def extract_text_from_txt(file):
    return file.read().decode("utf-8")

def simplify_text(text):
    # Split long text into chunks (max ~512 tokens) for T5
    max_chunk_size = 500
    simplified_parts = []
    for i in range(0, len(text), max_chunk_size):
        chunk = text[i:i+max_chunk_size]
        result = simplifier(chunk, max_length=512, truncation=True)
        simplified_parts.append(result[0]['generated_text'])
    return "\n\n".join(simplified_parts)

def classify_document(text):
    # Placeholder for document classification
    # Replace with a trained classifier model or rules later
    if "non-disclosure" in text.lower() or "confidential" in text.lower():
        return "NDA (Non-Disclosure Agreement)"
    elif "lease" in text.lower() or "tenant" in text.lower():
        return "Lease Agreement"
    elif "employment" in text.lower() or "employee" in text.lower():
        return "Employment Contract"
    elif "service" in text.lower():
        return "Service Agreement"
    else:
        return "Unknown Document Type"

# --- Streamlit UI ---

st.set_page_config(page_title="Legal Document Analyzer", layout="wide")

st.title("📄 Legal Document Analyzer Prototype")

uploaded_file = st.file_uploader("Upload legal document (PDF, DOCX, TXT)", type=["pdf", "docx", "txt"])

if uploaded_file is not None:
    file_type = uploaded_file.name.split('.')[-1].lower()
    with st.spinner("Extracting text..."):
        if file_type == "pdf":
            text = extract_text_from_pdf(uploaded_file)
        elif file_type == "docx":
            text = extract_text_from_docx(uploaded_file)
        elif file_type == "txt":
            text = extract_text_from_txt(uploaded_file)
        else:
            st.error("Unsupported file type!")
            st.stop()

    st.subheader("Extracted Text")
    st.text_area("Full Document Text", text, height=250)

    # Document classification
    doc_type = classify_document(text)
    st.markdown(f"**Document Type:** {doc_type}")

    # Clause simplification (You can improve clause detection and breakdown later)
    st.subheader("Clause Simplification")
    with st.spinner("Simplifying clauses..."):
        simplified_text = simplify_text(text)
    st.text_area("Simplified Text", simplified_text, height=250)

    # Named Entity Recognition
    st.subheader("Named Entities (NER)")
    with st.spinner("Extracting entities..."):
        entities = ner_pipeline(text)

    if entities:
        for ent in entities:
            st.markdown(f"- **{ent['entity_group']}**: {ent['word']}")
    else:
        st.write("No named entities found.")

else:
    st.info("Upload a legal document in PDF, DOCX, or TXT format to get started.")

