import streamlit as st

# Set page config FIRST
st.set_page_config(
    page_title="Exam Question Generator",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Import other libraries AFTER page config
import os
import tempfile
import base64
import json
import csv
import re
from io import BytesIO
import fitz  # PyMuPDF
import docx
import pandas as pd
import openai
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize
import spacy
import random
from rake_nltk import Rake
import time

# Initialize NLP resources
def initialize_nlp():
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        nltk.download('punkt')
    
    try:
        nltk.data.find('corpora/stopwords')
    except LookupError:
        nltk.download('stopwords')

    try:
        return spacy.load("en_core_web_sm")
    except OSError:
        from spacy.cli import download
        download("en_core_web_sm")
        return spacy.load("en_core_web_sm")

# Rest of the functions remain the same
def extract_text_from_pdf(pdf_file):
    with fitz.open(stream=pdf_file.read(), filetype="pdf") as doc:
        return "".join(page.get_text() for page in doc)

def extract_text_from_txt(txt_file):
    return txt_file.getvalue().decode("utf-8")

# ... [Keep all your existing functions unchanged] ...

def main():
    # Initialize NLP components
    try:
        nlp = initialize_nlp()
    except Exception as e:
        st.error(f"Initialization error: {str(e)}")
        return

    # Custom CSS
    st.markdown("""<style>
    .main-header { font-size: 2.5rem; color: #1E3A8A; text-align: center; }
    /* Keep all your existing CSS styles here */
    </style>""", unsafe_allow_html=True)

    # Main app logic
    st.markdown('<h1 class="main-header">📚 Exam Question Generator</h1>', unsafe_allow_html=True)
    
    # File uploader and processing
    uploaded_file = st.file_uploader("Choose a file", type=["pdf", "txt"])
    
    if uploaded_file:
        with st.spinner("Processing file..."):
            text = extract_text_from_pdf(uploaded_file) if uploaded_file.type == "application/pdf" else extract_text_from_txt(uploaded_file)
            
        # Rest of your original UI logic
        # Question generation and display code
        # ...

if __name__ == "__main__":
    main()
