import streamlit as st
import os
import base64
from io import BytesIO
from datetime import datetime
import google.generativeai as genai

# Set page configuration
st.set_page_config(
    page_title="AI Question Generator",
    page_icon="📚",
    layout="wide"
)

# Simple CSS styling
st.markdown("""
<style>
    .main-header {
        font-size: 2rem;
        color: #1E3A8A;
        text-align: center;
        margin-bottom: 1rem;
    }
    .question-box {
        background-color: #EFF6FF;
        border-left: 5px solid #3B82F6;
        padding: 15px;
        margin-bottom: 15px;
        border-radius: 5px;
    }
    .stButton>button {
        background-color: #2563EB;
        color: white;
    }
</style>
""", unsafe_allow_html=True)

# File processing functions
def extract_text_from_txt(txt_file):
    return txt_file.getvalue().decode("utf-8")

def extract_text_from_docx(docx_file):
    from docx import Document
    doc = Document(BytesIO(docx_file.read()))
    return "\n".join([para.text for para in doc.paragraphs])

def extract_text_from_pptx(pptx_file):
    from pptx import Presentation
    prs = Presentation(BytesIO(pptx_file.read()))
    text = []
    for slide in prs.slides:
        for shape in slide.shapes:
            if hasattr(shape, "text"):
                text.append(shape.text)
    return "\n".join(text)

def extract_text(file):
    if file.type == "text/plain":
        return extract_text_from_txt(file)
    elif file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
        return extract_text_from_docx(file)
    elif file.type == "application/vnd.openxmlformats-officedocument.presentationml.presentation":
        return extract_text_from_pptx(file)
    return ""

# Gemini API integration
def generate_with_gemini(text, question_type, difficulty, api_key):
    genai.configure(api_key=api_key)
    
    # Truncate very long text
    text = text[:10000] if len(text) > 10000 else text
    
    prompt = f"""
    Generate 5 {difficulty.lower()} {question_type} questions based on the following content.
    For each question, provide:
    - The question text
    - Correct answer
    - For MCQs: 3 incorrect options
    - Brief explanation
    
    Content:
    {text}
    """
    
    try:
        model = genai.GenerativeModel('gemini-pro')
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        st.error(f"Gemini API error: {str(e)}")
        return None

def main():
    st.markdown('<h1 class="main-header">📚 AI Question Generator</h1>', unsafe_allow_html=True)
    
    with st.sidebar:
        st.markdown("### Settings")
        api_key = st.text_input("Gemini API Key", type="password", 
                               help="Enter your Gemini API key")
        
        st.markdown("### Upload Lecture Material")
        uploaded_file = st.file_uploader("Choose a file", 
                                        type=["txt", "docx", "pptx"])
        
        if uploaded_file:
            st.markdown("### Question Settings")
            question_type = st.selectbox(
                "Question Type",
                ["MCQ", "Fill-in-the-Blank", "Short Answer"]
            )
            
            difficulty = st.selectbox(
                "Difficulty Level",
                ["Easy", "Medium", "Hard"]
            )
            
            generate_button = st.button("Generate Questions")

    if uploaded_file:
        text = extract_text(uploaded_file)
        
        with st.expander("Preview Extracted Text"):
            st.text(text[:500] + "...")
        
        if generate_button:
            if not api_key:
                st.error("Please enter your Gemini API key")
                return
                
            with st.spinner("Generating questions with Gemini AI..."):
                result = generate_with_gemini(text, question_type, difficulty, api_key)
                
                if result:
                    st.session_state.questions = result
                    st.success("Questions generated successfully!")
                else:
                    st.error("Failed to generate questions")

        if 'questions' in st.session_state:
            st.markdown(f"## Generated {question_type} Questions")
            st.markdown(st.session_state.questions)
            
            # Download as text file
            txt_file = BytesIO(st.session_state.questions.encode('utf-8'))
            st.download_button(
                label="Download Questions",
                data=txt_file,
                file_name=f"questions_{datetime.now().strftime('%Y%m%d')}.txt",
                mime="text/plain"
            )
    else:
        st.info("Please upload a file (TXT, DOCX, or PPTX) to generate questions")

if __name__ == "__main__":
    main()
