import streamlit as st
import os
import tempfile
import base64
import json
import time
import re
from io import BytesIO
from datetime import datetime

# File processing libraries
import fitz  # PyMuPDF
import docx
from docx import Document
from docx.shared import Pt, RGBColor
import pandas as pd

# NLP libraries
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize, word_tokenize
import spacy
from rake_nltk import Rake
import random
import google.generativeai as genai

# Set page configuration
st.set_page_config(
    page_title="Exam Question Generator",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Apply custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1E3A8A;
        text-align: center;
        margin-bottom: 1rem;
        font-weight: bold;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #3B82F6;
        margin-top: 1rem;
        margin-bottom: 0.5rem;
    }
    .custom-box {
        background-color: #F3F4F6;
        border-radius: 10px;
        padding: 20px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    .question-box {
        background-color: #EFF6FF;
        border-left: 5px solid #3B82F6;
        padding: 15px;
        margin-bottom: 15px;
        border-radius: 5px;
    }
    .option-correct {
        background-color: #ECFDF5;
        border-left: 5px solid #10B981;
        padding: 8px;
        margin-bottom: 8px;
        border-radius: 3px;
    }
    .option-incorrect {
        background-color: #F9FAFB;
        border-left: 5px solid #9CA3AF;
        padding: 8px;
        margin-bottom: 8px;
        border-radius: 3px;
    }
    .stButton>button {
        background-color: #2563EB;
        color: white;
        font-weight: bold;
    }
    .info-box {
        background-color: #DBEAFE;
        padding: 10px;
        border-radius: 5px;
        margin-bottom: 10px;
    }
    .stProgress .st-eb {
        background-color: #3B82F6;
    }
</style>
""", unsafe_allow_html=True)

# Initialize NLP resources
@st.cache_resource
def initialize_nlp():
    # Download required NLTK resources
    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True)
    nltk.download('averaged_perceptron_tagger', quiet=True)
    
    # Initialize Rake for keyword extraction
    rake = Rake()
    
    # Initialize spaCy
    try:
        nlp = spacy.load("en_core_web_sm")
    except OSError:
        spacy.cli.download("en_core_web_sm")
        nlp = spacy.load("en_core_web_sm")
    
    return rake, nlp

# Extract text from PDF files
def extract_text_from_pdf(pdf_file):
    with fitz.open(stream=pdf_file.read(), filetype="pdf") as doc:
        text = ""
        for page in doc:
            text += page.get_text()
        return text

# Extract text from TXT files
def extract_text_from_txt(txt_file):
    return txt_file.getvalue().decode("utf-8")

# Function to generate questions using OpenAI API
def generate_questions_with_gemini(text, question_type, difficulty, api_key):
    # Set up Gemini client
    if not api_key:
        return None
    
    genai.configure(api_key=api_key)
    
    # Truncate text if too long (limit to ~4000 tokens)
    max_chars = 12000  # Approximate for 4000 tokens
    if len(text) > max_chars:
        text = text[:max_chars]
    
    # Create prompt based on question type and difficulty
    type_descriptions = {
        "MCQ": "multiple-choice questions with 1 correct answer and 3 plausible but incorrect options",
        "Fill-in-the-Blank": "fill-in-the-blank questions where key terms or concepts are removed",
        "Short Answer": "short answer questions that require brief explanations of concepts"
    }
    
    difficulty_descriptions = {
        "Easy": "basic recall of information",
        "Medium": "understanding of concepts and relationships",
        "Hard": "application, analysis, and synthesis of information"
    }
    
    # Build the prompt
    prompt = f"""
You are an expert educator creating exam questions based on lecture notes. 
Create 5 {difficulty.lower()} {question_type} questions about the following lecture content.
Question difficulty should be {difficulty.lower()}, focusing on {difficulty_descriptions[difficulty]}.

For each question:
"""

    if question_type == "MCQ":
        prompt += """
- Create a clear question stem
- Provide 4 options labeled A, B, C, D
- Exactly one option should be correct
- The other 3 options should be plausible but incorrect
- Mark the correct answer
- Provide a brief explanation why the correct answer is right
"""
    elif question_type == "Fill-in-the-Blank":
        prompt += """
- Create a sentence with an important concept or term removed and replaced with _____
- Provide the correct answer
- Make sure the missing term is significant to understanding the content
"""
    else:  # Short Answer
        prompt += """
- Create a question that requires explaining a concept, process, or relationship
- Provide a model short answer (2-3 sentences)
- Focus on key concepts from the lecture
"""

    prompt += f"""
Format your response as a JSON object with the following structure:
{{
  "questions": [
    {{
      "question": "Question text",
      "type": "{question_type}",
"""

    if question_type == "MCQ":
        prompt += """
      "options": ["A. option1", "B. option2", "C. option3", "D. option4"],
      "correct_answer": "A",
      "explanation": "Explanation text"
"""
    elif question_type == "Fill-in-the-Blank":
        prompt += """
      "answer": "correct word or phrase"
"""
    else:  # Short Answer
        prompt += """
      "model_answer": "Sample correct answer"
"""

    prompt += """
    }
  ]
}


    try:
        model = genai.GenerativeModel('gemini-pro')
        response = model.generate_content(prompt)
        
        # Extract response text
        result = response.text
        
        # Extract JSON from the response
        try:
            json_match = re.search(r'({[\s\S]*})', result)
            if json_match:
                json_str = json_match.group(1)
                return json.loads(json_str)
            return json.loads(result)
        except json.JSONDecodeError:
            st.error("Failed to parse Gemini response as JSON. Using fallback method instead.")
            return None
            
    except Exception as e:
        st.error(f"Gemini API error: {str(e)}")
        return None

# Fallback method for generating questions without OpenAI
def generate_questions_fallback(text, question_type, difficulty, rake_nlp, spacy_nlp):
    # Process the text
    sentences = sent_tokenize(text)
    if len(sentences) < 10:
        return {"error": "Text is too short to generate meaningful questions."}
    
    # Extract keywords and phrases
    rake_nlp.extract_keywords_from_text(text)
    keywords = rake_nlp.get_ranked_phrases()[:20]
    
    # Process with spaCy for entity recognition
    doc = spacy_nlp(text)
    entities = [ent.text for ent in doc.ents]
    
    # Generate questions based on type
    questions = []
    
    if question_type == "MCQ":
        # Find sentences with important keywords
        keyword_sentences = []
        for sentence in sentences:
            for keyword in keywords[:10]:
                if keyword.lower() in sentence.lower():
                    keyword_sentences.append((sentence, keyword))
                    break
        
        # Take up to 5 sentences
        selected_pairs = random.sample(keyword_sentences, min(5, len(keyword_sentences)))
        
        for sentence, keyword in selected_pairs:
            # Create question
            question = f"Which of the following best describes {keyword}?"
            
            # Create correct answer
            correct_answer = sentence
            
            # Create distractors (incorrect options)
            distractors = []
            for _ in range(3):
                random_sentence = random.choice(sentences)
                while random_sentence == sentence or random_sentence in distractors:
                    random_sentence = random.choice(sentences)
                distractors.append(random_sentence)
            
            # Randomize options
            options = [correct_answer] + distractors
            random.shuffle(options)
            
            # Find index of correct answer
            correct_index = options.index(correct_answer)
            correct_letter = chr(65 + correct_index)  # A, B, C, or D
            
            # Format options with letters
            formatted_options = [f"{chr(65+i)}. {opt}" for i, opt in enumerate(options)]
            
            questions.append({
                "question": question,
                "type": "MCQ",
                "options": formatted_options,
                "correct_answer": correct_letter,
                "explanation": f"This option correctly describes {keyword}."
            })
            
    elif question_type == "Fill-in-the-Blank":
        # Find sentences with important keywords
        good_sentences = []
        for sentence in sentences:
            if len(sentence.split()) > 8 and len(sentence.split()) < 25:
                for keyword in keywords:
                    if keyword.lower() in sentence.lower() and keyword not in entities:
                        good_sentences.append((sentence, keyword))
                        break
        
        # Take up to 5 sentences
        selected_pairs = random.sample(good_sentences, min(5, len(good_sentences)))
        
        for sentence, keyword in selected_pairs:
            # Create fill-in-the-blank by replacing the keyword
            blank_sentence = re.sub(re.escape(keyword), "_____", sentence, flags=re.IGNORECASE)
            
            questions.append({
                "question": blank_sentence,
                "type": "Fill-in-the-Blank",
                "answer": keyword
            })
            
    else:  # Short Answer
        # Find sentences with important concepts
        doc = spacy_nlp(text)
        important_sentences = []
        
        for sentence in sentences:
            if len(sentence.split()) > 10:
                sent_doc = spacy_nlp(sentence)
                if any(ent.label_ in ["ORG", "PERSON", "GPE", "LOC", "EVENT", "WORK_OF_ART"] for ent in sent_doc.ents):
                    important_sentences.append(sentence)
                for keyword in keywords[:10]:
                    if keyword.lower() in sentence.lower():
                        important_sentences.append(sentence)
                        break
        
        # Remove duplicates
        important_sentences = list(set(important_sentences))
        
        # Take up to 5 sentences
        selected_sentences = random.sample(important_sentences, min(5, len(important_sentences)))
        
        for sentence in selected_sentences:
            # Create a question by transforming the sentence
            doc = spacy_nlp(sentence)
            main_subject = ""
            
            # Try to find the subject of the sentence
            for token in doc:
                if token.dep_ == "nsubj" or token.dep_ == "nsubjpass":
                    main_subject = token.text
                    break
            
            if main_subject:
                question = f"Explain the significance of {main_subject} as discussed in the lecture."
            else:
                # Fallback to a generic question
                question = f"Explain the following concept from the lecture: '{sentence[:50]}...'"
            
            questions.append({
                "question": question,
                "type": "Short Answer",
                "model_answer": f"Based on the lecture notes, {sentence}"
            })
    
    # Format the return object to match OpenAI's format
    return {"questions": questions[:5]}  # Limit to 5 questions

# Function to create downloadable DOCX file
def create_docx(questions, question_type):
    doc = Document()
    
    # Add title
    title = doc.add_heading(f"{question_type} Questions", level=1)
    
    # Add date
    doc.add_paragraph(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    doc.add_paragraph()  # Add space
    
    # Add questions
    for i, q in enumerate(questions, 1):
        # Add question number and text
        question_para = doc.add_paragraph()
        question_para.add_run(f"Question {i}: ").bold = True
        question_para.add_run(q["question"])
        
        # Format based on question type
        if q["type"] == "MCQ":
            # Add options
            for option in q["options"]:
                option_para = doc.add_paragraph()
                option_para.add_run(f"    {option}")
                
            # Add correct answer
            answer_para = doc.add_paragraph()
            answer_para.add_run("    Correct Answer: ").bold = True
            answer_para.add_run(q["correct_answer"])
            
            # Add explanation if available
            if "explanation" in q:
                exp_para = doc.add_paragraph()
                exp_para.add_run("    Explanation: ").italic = True
                exp_para.add_run(q["explanation"])
                
        elif q["type"] == "Fill-in-the-Blank":
            # Add answer
            answer_para = doc.add_paragraph()
            answer_para.add_run("    Answer: ").bold = True
            answer_para.add_run(q["answer"])
            
        else:  # Short Answer
            # Add model answer
            model_para = doc.add_paragraph()
            model_para.add_run("    Model Answer: ").bold = True
            model_para.add_run(q["model_answer"])
        
        # Add space between questions
        doc.add_paragraph()
    
    # Save the document to a BytesIO object
    docx_bytes = BytesIO()
    doc.save(docx_bytes)
    docx_bytes.seek(0)
    
    return docx_bytes

# Function to create downloadable CSV file
def create_csv(questions, question_type):
    output = BytesIO()
    
    # Create CSV writer
    writer = csv.writer(output)
    
    # Write header
    if question_type == "MCQ":
        writer.writerow(["Question", "Option A", "Option B", "Option C", "Option D", "Correct Answer", "Explanation"])
    elif question_type == "Fill-in-the-Blank":
        writer.writerow(["Question", "Answer"])
    else:  # Short Answer
        writer.writerow(["Question", "Model Answer"])
    
    # Write questions
    for q in questions:
        if q["type"] == "MCQ":
            # Extract options without A., B., etc.
            options = [opt.split(". ", 1)[1] if ". " in opt else opt for opt in q["options"]]
            while len(options) < 4:
                options.append("")  # Ensure we have 4 options
                
            writer.writerow([
                q["question"],
                options[0] if len(options) > 0 else "",
                options[1] if len(options) > 1 else "",
                options[2] if len(options) > 2 else "",
                options[3] if len(options) > 3 else "",
                q["correct_answer"],
                q.get("explanation", "")
            ])
        elif q["type"] == "Fill-in-the-Blank":
            writer.writerow([q["question"], q["answer"]])
        else:  # Short Answer
            writer.writerow([q["question"], q["model_answer"]])
    
    output.seek(0)
    return output

# Function to create download link
def get_download_link(file_bytes, file_name, file_type):
    b64 = base64.b64encode(file_bytes.read()).decode()
    return f'<a href="data:application/{file_type};base64,{b64}" download="{file_name}" class="download-button">Download {file_name}</a>'

# Main app function
def main():
    # Initialize NLP components
    rake_nlp, spacy_nlp = initialize_nlp()
    
    # App title and header
    st.markdown('<h1 class="main-header">📚 Exam Question Generator from Lecture Notes</h1>', unsafe_allow_html=True)
    
    # Create sidebar for settings
    with st.sidebar:
        st.markdown("### Settings")
        
        # OpenAI API key input
        api_key = st.text_input("AIzaSyC0gjQQgnOUfjEowLmFB3crn-UzWIBXQHg", type="password", 
                               help="Enter your OpenAI API key. If not provided, the app will use a fallback method.")
        
        st.markdown("---")
        
        # File upload section
        st.markdown("### Upload Lecture Notes")
        uploaded_file = st.file_uploader("Choose a file", type=["pdf", "txt"])
        
        # Question type and difficulty selection (only shown when file is uploaded)
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
            
            generate_button = st.button("Generate Questions", type="primary")
            
            # Export format selection
            st.markdown("### Export Settings")
            export_format = st.selectbox(
                "Export Format",
                ["DOCX", "CSV"]
            )
    
    # Main content area
    if uploaded_file:
        # Show file information
        col1, col2 = st.columns(2)
        with col1:
            st.markdown(f"**File Name:** {uploaded_file.name}")
        with col2:
            file_type = "PDF" if uploaded_file.type == "application/pdf" else "TXT"
            st.markdown(f"**File Type:** {file_type}")
        
        # Text extraction
        if uploaded_file.type == "application/pdf":
            text = extract_text_from_pdf(uploaded_file)
        else:
            text = extract_text_from_txt(uploaded_file)
        
        # Show a sample of extracted text
        with st.expander("Preview Extracted Text"):
            st.markdown(f"```\n{text[:500]}...\n```")
            st.markdown(f"*Total characters: {len(text)}*")
        
        # Question generation
        if 'questions' not in st.session_state:
            st.session_state.questions = None
        
        if generate_button:
            with st.spinner("Generating questions..."):
                # Try OpenAI first if API key is provided
                if api_key:
                    try:
                        questions_data = generate_questions_with_gemini(
                            text, question_type, difficulty, api_key
                        )
                    except Exception as e:
                        st.error(f"Error using OpenAI API: {str(e)}")
                        questions_data = None
                else:
                    questions_data = None
                
                # Fall back to rule-based method if OpenAI fails or no API key
                if questions_data is None:
                    with st.spinner("Using fallback method to generate questions..."):
                        questions_data = generate_questions_fallback(
                            text, question_type, difficulty, rake_nlp, spacy_nlp
                        )
                
                # Store questions in session state
                st.session_state.questions = questions_data
        
        # Display questions if available
        if st.session_state.questions and "questions" in st.session_state.questions:
            st.markdown(f'<h2 class="sub-header">Generated {question_type} Questions ({difficulty})</h2>', unsafe_allow_html=True)
            
            # Display questions based on type
            for i, q in enumerate(st.session_state.questions["questions"], 1):
                with st.container():
                    st.markdown(f'<div class="question-box">', unsafe_allow_html=True)
                    st.markdown(f"**Question {i}:** {q['question']}")
                    
                    if q["type"] == "MCQ":
                        for option in q["options"]:
                            option_letter = option[0]
                            is_correct = option_letter == q["correct_answer"]
                            
                            if is_correct:
                                st.markdown(f'<div class="option-correct">{option} ✓</div>', unsafe_allow_html=True)
                            else:
                                st.markdown(f'<div class="option-incorrect">{option}</div>', unsafe_allow_html=True)
                        
                        with st.expander("Explanation"):
                            st.write(q.get("explanation", "No explanation available."))
                            
                    elif q["type"] == "Fill-in-the-Blank":
                        st.markdown(f"**Answer:** {q['answer']}")
                        
                    else:  # Short Answer
                        with st.expander("Model Answer"):
                            st.write(q["model_answer"])
                    
                    st.markdown('</div>', unsafe_allow_html=True)
            
            # Export options
            st.markdown(f'<h2 class="sub-header">Export Questions</h2>', unsafe_allow_html=True)
            
            col1, col2 = st.columns(2)
            
            with col1:
                if export_format == "DOCX":
                    # Create DOCX
                    docx_bytes = create_docx(st.session_state.questions["questions"], question_type)
                    
                    # Download button
                    st.download_button(
                        label="Download as DOCX",
                        data=docx_bytes,
                        file_name=f"{question_type}_questions_{difficulty.lower()}.docx",
                        mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document"
                    )
                else:  # CSV
                    # Create CSV
                    csv_bytes = create_csv(st.session_state.questions["questions"], question_type)
                    
                    # Download button
                    st.download_button(
                        label="Download as CSV",
                        data=csv_bytes,
                        file_name=f"{question_type}_questions_{difficulty.lower()}.csv",
                        mime="text/csv"
                    )
            
    else:
        # Show welcome message when no file is uploaded
        st.markdown('<div class="custom-box">', unsafe_allow_html=True)
        st.markdown("### 👋 Welcome to the Exam Question Generator!")
        st.markdown("""
        This app helps you create exam questions from your lecture notes. Here's how to use it:
        
        1. Upload your lecture notes (PDF or TXT format)
        2. Select the type of questions you want to generate
        3. Choose the difficulty level
        4. Click on "Generate Questions"
        5. Download the questions in your preferred format
        
        For best results, provide an OpenAI API key in the sidebar.
        """)
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Show sample questions
        with st.expander("See sample outputs"):
            st.markdown("### Multiple Choice Question (MCQ) Example")
            st.markdown("""
            **Question:** Which of the following best describes photosynthesis?
            
            A. A process where plants release carbon dioxide and consume oxygen
            B. A process where plants convert light energy into chemical energy ✓
            C. A process only occurring in animals to produce energy
            D. A process of breaking down glucose in the absence of oxygen
            
            **Explanation:** Photosynthesis is the process used by plants to convert light energy from the sun into chemical energy stored in glucose molecules.
            """)
            
            st.markdown("### Fill-in-the-Blank Example")
            st.markdown("""
            **Question:** During cellular respiration, glucose is broken down into carbon dioxide and water, releasing _____ as a byproduct.
            
            **Answer:** energy (or ATP)
            """)
            
            st.markdown("### Short Answer Example")
            st.markdown("""
            **Question:** Explain the significance of DNA replication in cell division.
            
            **Model Answer:** DNA replication is crucial for cell division because it ensures each daughter cell receives an identical copy of the genetic material. During this process, the DNA double helix unwinds, and each strand serves as a template for the synthesis of a new complementary strand, resulting in two identical DNA molecules before the cell divides.
            """)

# Run the app
if __name__ == "__main__":
    main()
