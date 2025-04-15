# app.py
import streamlit as st
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

# Download necessary NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')
    
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

# Load spaCy NLP model
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    st.warning("Downloading spaCy model. This might take a while...")
    os.system("python -m spacy download en_core_web_sm")
    nlp = spacy.load("en_core_web_sm")

# Initialize RAKE algorithm
rake = Rake()

# Set page config
st.set_page_config(
    page_title="Exam Question Generator",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better UI
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1E3A8A;
        text-align: center;
        margin-bottom: 1rem;
        font-weight: 700;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #3B82F6;
        margin-bottom: 1rem;
        font-weight: 600;
    }
    .info-text {
        color: #4B5563;
        font-size: 1rem;
    }
    .highlight {
        background-color: #EFF6FF;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 5px solid #3B82F6;
    }
    .stButton button {
        background-color: #3B82F6;
        color: white;
        font-weight: bold;
        padding: 0.5rem 1rem;
        border-radius: 0.5rem;
        transition: all 0.3s;
    }
    .stButton button:hover {
        background-color: #1E3A8A;
        color: white;
    }
    .question-box {
        background-color: #F3F4F6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
        border-left: 5px solid #3B82F6;
    }
    .option-box {
        padding: 0.5rem;
        margin: 0.5rem 0;
        border-radius: 0.3rem;
    }
    .correct-option {
        border-left: 5px solid #10B981;
    }
    .download-btn {
        text-align: center;
        margin-top: 2rem;
    }
    .file-upload {
        border: 2px dashed #CBD5E1;
        border-radius: 0.5rem;
        padding: 2rem;
        text-align: center;
        margin-bottom: 1rem;
    }
    .settings-panel {
        background-color: #F9FAFB;
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
    }
</style>
""", unsafe_allow_html=True)

# Functions for text extraction
def extract_text_from_pdf(pdf_file):
    with fitz.open(stream=pdf_file.read(), filetype="pdf") as doc:
        text = ""
        for page in doc:
            text += page.get_text()
    return text

def extract_text_from_txt(txt_file):
    return txt_file.getvalue().decode("utf-8")

# Function to create a download link
def create_download_link(content, filename, link_text):
    b64 = base64.b64encode(content).decode()
    href = f'<a href="data:application/octet-stream;base64,{b64}" download="{filename}">{link_text}</a>'
    return href

# Function to export questions to DOCX
def export_to_docx(questions, question_type):
    doc = docx.Document()
    doc.add_heading(f"{question_type} Questions", level=1)
    
    for i, q in enumerate(questions, 1):
        doc.add_paragraph(f"{i}. {q['question']}", style='Heading 2')
        
        if question_type == "MCQ":
            # Add options with correct answer highlighted
            options = q['options']
            correct_answer = q['answer']
            for j, option in enumerate(options):
                p = doc.add_paragraph(f"    {chr(65+j)}) {option}")
                if option == correct_answer:
                    p.runs[0].bold = True
        elif question_type == "Fill-in-the-Blank":
            doc.add_paragraph(f"Answer: {q['answer']}")
        elif question_type == "Short Answer":
            doc.add_paragraph(f"Sample Answer: {q['answer']}")
            
        doc.add_paragraph()  # Add some space between questions
    
    # Save the document to a BytesIO object
    f = BytesIO()
    doc.save(f)
    f.seek(0)
    return f.getvalue()

# Function to export questions to CSV
def export_to_csv(questions, question_type):
    output = BytesIO()
    fieldnames = ['Question Number', 'Question', 'Answer']
    
    if question_type == "MCQ":
        fieldnames.extend(['Option A', 'Option B', 'Option C', 'Option D'])
    
    writer = csv.DictWriter(output, fieldnames=fieldnames)
    writer.writeheader()
    
    for i, q in enumerate(questions, 1):
        row = {
            'Question Number': i,
            'Question': q['question'],
            'Answer': q['answer']
        }
        
        if question_type == "MCQ":
            for j, option in enumerate(q['options']):
                row[f'Option {chr(65+j)}'] = option
                
        writer.writerow(row)
    
    output.seek(0)
    return output.getvalue()

# Fallback question generation (when OpenAI API is not available)
def generate_mcq_fallback(text, num_questions=5, difficulty="Medium"):
    questions = []
    # Extract entities to use as answers
    doc = nlp(text)
    entities = []
    
    # Look for named entities and important noun phrases
    for ent in doc.ents:
        if ent.label_ in ["PERSON", "ORG", "GPE", "DATE", "EVENT", "WORK_OF_ART", "LAW", "NORP", "FAC"]:
            entities.append(ent.text)
    
    # Extract noun phrases as potential answers
    for chunk in doc.noun_chunks:
        if len(chunk.text.split()) <= 3 and chunk.text.lower() not in [e.lower() for e in entities]:
            entities.append(chunk.text)
    
    # Use RAKE for keyword extraction as additional source
    rake.extract_keywords_from_text(text)
    keywords = rake.get_ranked_phrases()[:20]
    entities.extend([k for k in keywords if len(k.split()) <= 3])
    
    # Remove duplicates and limit
    entities = list(set(entities))[:15]
    
    # Get sentences containing these entities
    sentences = sent_tokenize(text)
    entity_sentences = {}
    
    for entity in entities:
        for sentence in sentences:
            if entity.lower() in sentence.lower():
                entity_sentences[entity] = sentence
                break
    
    # Create questions
    count = 0
    for entity, sentence in entity_sentences.items():
        if count >= num_questions:
            break
        
        # Create a question by replacing the entity with a blank
        question = sentence.replace(entity, "________")
        
        # Generate distractors (other entities)
        distractors = [e for e in entities if e != entity and e.lower() != entity.lower()][:3]
        if len(distractors) < 3:  # If we don't have enough distractors
            # Create some fake distractors
            distractors.extend([f"Option {i+1}" for i in range(3-len(distractors))])
        
        # Create a list with the correct answer and distractors
        options = [entity] + distractors[:3]
        random.shuffle(options)  # Shuffle to randomize position of correct answer
        
        questions.append({
            "question": question,
            "options": options,
            "answer": entity
        })
        count += 1
    
    # If we couldn't generate enough questions, create generic ones
    while len(questions) < num_questions:
        sample_sentence = random.choice(sentences)
        words = sample_sentence.split()
        if len(words) > 5:
            # Choose a random word to blank out
            blank_idx = random.randint(2, len(words) - 2)
            blank_word = words[blank_idx]
            words[blank_idx] = "________"
            question = " ".join(words)
            
            # Generate options
            options = [blank_word]
            other_words = [w for w in re.findall(r'\b\w+\b', text) if len(w) > 4]
            options.extend(random.sample(other_words, 3))
            random.shuffle(options)
            
            questions.append({
                "question": question,
                "options": options,
                "answer": blank_word
            })
    
    return questions

def generate_fill_in_blank_fallback(text, num_questions=5, difficulty="Medium"):
    questions = []
    sentences = sent_tokenize(text)
    
    # Identify important terms and concepts
    doc = nlp(text)
    
    # Extract noun phrases and entities as potential blanks
    important_terms = []
    for chunk in doc.noun_chunks:
        if len(chunk.text.split()) <= 3:
            important_terms.append(chunk.text)
    
    for ent in doc.ents:
        important_terms.append(ent.text)
    
    # Use RAKE for keyword extraction as additional source
    rake.extract_keywords_from_text(text)
    important_terms.extend(rake.get_ranked_phrases()[:20])
    
    # Remove duplicates and sort by length (prefer multi-word terms)
    important_terms = sorted(list(set(important_terms)), key=len, reverse=True)
    
    # Create fill-in-the-blank questions
    count = 0
    for term in important_terms:
        if count >= num_questions:
            break
            
        # Find a sentence containing this term
        for sentence in sentences:
            if term in sentence and len(sentence.split()) > 5:
                # Replace the term with a blank
                question = sentence.replace(term, "________________")
                
                questions.append({
                    "question": question,
                    "answer": term
                })
                count += 1
                break
    
    # If we couldn't generate enough questions, use random sentences
    random.shuffle(sentences)
    for sentence in sentences:
        if len(questions) >= num_questions:
            break
            
        words = sentence.split()
        if len(words) > 5:
            # Choose a random word to blank out
            blank_idx = random.randint(2, len(words) - 2)
            blank_word = words[blank_idx]
            
            # Skip short or common words
            if len(blank_word) < 4 or blank_word.lower() in stopwords.words('english'):
                continue
                
            words[blank_idx] = "________________"
            question = " ".join(words)
            
            questions.append({
                "question": question,
                "answer": blank_word
            })
    
    return questions

def generate_short_answer_fallback(text, num_questions=5, difficulty="Medium"):
    questions = []
    sentences = sent_tokenize(text)
    
    # Filter for factual sentences (longer sentences often contain more info)
    factual_sentences = [s for s in sentences if len(s.split()) > 8 and len(s.split()) < 25]
    
    # Sort by potential information content
    factual_sentences.sort(key=len, reverse=True)
    
    # Extract key points from top sentences
    for sentence in factual_sentences[:min(num_questions*2, len(factual_sentences))]:
        if len(questions) >= num_questions:
            break
            
        doc = nlp(sentence)
        
        # Check if sentence has a subject and a verb
        has_subj = False
        has_verb = False
        
        for token in doc:
            if token.dep_ in ["nsubj", "nsubjpass"]:
                has_subj = True
            if token.pos_ == "VERB":
                has_verb = True
        
        if has_subj and has_verb:
            # Convert to question
            if sentence.lower().startswith(("the ", "a ", "an ")):
                # For sentences starting with articles, try to make a "what" question
                question = f"What {sentence.lower()}?"
            else:
                # Create a question by changing word order or adding interrogative
                if "is" in sentence or "are" in sentence:
                    parts = sentence.split(" is ", 1)
                    if len(parts) > 1:
                        question = f"What is {parts[1].rstrip('.')}?"
                    else:
                        parts = sentence.split(" are ", 1)
                        if len(parts) > 1:
                            question = f"What are {parts[1].rstrip('.')}?"
                        else:
                            question = f"Explain the following: {sentence}"
                else:
                    question = f"Explain the following: {sentence}"
            
            questions.append({
                "question": question,
                "answer": sentence
            })
    
    # If we still need more questions, create generic ones
    if len(questions) < num_questions:
        # Extract key concepts using RAKE
        rake.extract_keywords_from_text(text)
        key_concepts = rake.get_ranked_phrases()[:10]
        
        for concept in key_concepts:
            if len(questions) >= num_questions:
                break
                
            question = f"Explain the concept of {concept} as described in the lecture notes."
            
            # Find sentences related to this concept for the sample answer
            related_sentences = []
            for sentence in sentences:
                if concept.lower() in sentence.lower():
                    related_sentences.append(sentence)
            
            if related_sentences:
                answer = " ".join(related_sentences[:2])
            else:
                answer = f"The concept of {concept} is an important topic covered in the lecture notes."
            
            questions.append({
                "question": question,
                "answer": answer
            })
    
    return questions

# Function to generate questions using OpenAI API
def generate_questions_with_openai(text, question_type, difficulty, num_questions=5):
    # Truncate text if too long (OpenAI has token limits)
    max_tokens = 4000
    if len(text) > max_tokens * 4:  # Rough estimate of 4 chars per token
        text = text[:max_tokens * 4]
    
    try:
        # Set up the API key
        api_key = st.session_state.get('openai_api_key', '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')
        if not api_key:
            st.warning("⚠️ OpenAI API key not provided. Using fallback question generation method.")
            raise ValueError("API key not provided")
            
        openai.api_key = api_key
        
        # Craft the prompt based on question type
        if question_type == "MCQ":
            prompt = f"""
            Create {num_questions} multiple-choice questions based on these lecture notes. 
            Each question should have 4 options with exactly 1 correct answer.
            Difficulty level: {difficulty}
            
            Format each question as follows:
            {{
                "question": "Question text here?",
                "options": ["Option A", "Option B", "Option C", "Option D"],
                "answer": "Correct option text here"
            }}
            
            Return a valid JSON array containing {num_questions} question objects.
            
            Lecture notes:
            {text}
            """
        elif question_type == "Fill-in-the-Blank":
            prompt = f"""
            Create {num_questions} fill-in-the-blank questions based on these lecture notes.
            Remove key terms/concepts/facts and replace them with blank spaces.
            Difficulty level: {difficulty}
            
            Format each question as follows:
            {{
                "question": "Sentence with __________ for the missing term.",
                "answer": "Correct term that goes in the blank"
            }}
            
            Return a valid JSON array containing {num_questions} question objects.
            
            Lecture notes:
            {text}
            """
        else:  # Short Answer
            prompt = f"""
            Create {num_questions} short-answer questions based on these lecture notes.
            Questions should test understanding of key concepts, facts, and relationships.
            Difficulty level: {difficulty}
            
            Format each question as follows:
            {{
                "question": "Question text here?",
                "answer": "Sample answer here (1-2 sentences)"
            }}
            
            Return a valid JSON array containing {num_questions} question objects.
            
            Lecture notes:
            {text}
            """
        
        # Make the API call
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are an educational expert who creates exam questions from lecture materials."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
            max_tokens=2000
        )
        
        # Extract the content from the response
        content = response.choices[0].message.content.strip()
        
        # Find and extract the JSON part
        json_pattern = r'\[.*\]'
        json_match = re.search(json_pattern, content, re.DOTALL)
        
        if json_match:
            json_str = json_match.group()
            questions = json.loads(json_str)
        else:
            # If JSON pattern not found, try the whole content
            questions = json.loads(content)
        
        return questions
        
    except Exception as e:
        st.error(f"Error generating questions with OpenAI: {str(e)}")
        
        # Fall back to rule-based methods
        if question_type == "MCQ":
            return generate_mcq_fallback(text, num_questions, difficulty)
        elif question_type == "Fill-in-the-Blank":
            return generate_fill_in_blank_fallback(text, num_questions, difficulty)
        else:  # Short Answer
            return generate_short_answer_fallback(text, num_questions, difficulty)

# Main App Logic
def main():
    # App Header
    st.markdown('<h1 class="main-header">📚 Exam Question Generator from Lecture Notes</h1>', unsafe_allow_html=True)
    
    # Sidebar for settings
    with st.sidebar:
        st.markdown('<h2 class="sub-header">⚙️ Settings</h2>', unsafe_allow_html=True)
        
        # API Key input
        st.markdown('<p class="info-text">Optional: Enter OpenAI API key for better quality questions</p>', unsafe_allow_html=True)
        openai_api_key = st.text_input("OpenAI API Key", type="password", help="Enter your OpenAI API key for better question generation")
        if openai_api_key:
            st.session_state['openai_api_key'] = openai_api_key
            st.success("✅ API Key saved!")
        
        st.divider()
        
        # About section
        st.markdown('<h3>About</h3>', unsafe_allow_html=True)
        st.markdown('''
        This app generates exam questions from your lecture notes.
        
        Upload your notes in PDF or TXT format, select the type of questions and difficulty level, and let the app do the rest!
        
        Questions can be downloaded as Word document (.docx) or CSV file.
        ''')
        
        st.divider()
        
        # Instructions
        with st.expander("How to use"):
            st.markdown('''
            1. Upload your lecture notes (PDF or TXT)
            2. Select the type of questions you want to generate
            3. Choose the difficulty level
            4. Click "Generate Questions"
            5. Download the generated questions in your preferred format
            ''')
    
    # Main content area
    st.markdown('<div class="highlight">', unsafe_allow_html=True)
    st.markdown('<p class="info-text">Upload your lecture notes to generate exam questions. Supported formats: PDF, TXT</p>', unsafe_allow_html=True)
    
    # File uploader
    st.markdown('<div class="file-upload">', unsafe_allow_html=True)
    uploaded_file = st.file_uploader("Choose a file", type=["pdf", "txt"], help="Upload your lecture notes in PDF or TXT format")
    st.markdown('</div>', unsafe_allow_html=True)
    
    if uploaded_file is not None:
        with st.spinner("Processing file..."):
            # Extract text from the uploaded file
            if uploaded_file.type == "application/pdf":
                text = extract_text_from_pdf(uploaded_file)
            else:  # txt file
                text = extract_text_from_txt(uploaded_file)
            
            # Show a preview of the extracted text
            with st.expander("Preview of extracted text"):
                st.text(text[:500] + "..." if len(text) > 500 else text)
        
        # Settings for question generation
        st.markdown('<div class="settings-panel">', unsafe_allow_html=True)
        st.markdown('<h3 class="sub-header">Question Settings</h3>', unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        with col1:
            question_type = st.selectbox(
                "Select the type of questions",
                ["MCQ", "Fill-in-the-Blank", "Short Answer"],
                help="Choose the type of questions you want to generate"
            )
        
        with col2:
            difficulty = st.selectbox(
                "Select difficulty level",
                ["Easy", "Medium", "Hard"],
                help="Choose the difficulty level of the questions"
            )
        
        num_questions = st.slider("Number of questions to generate", min_value=3, max_value=10, value=5, step=1)
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Button to generate questions
        if st.button("🚀 Generate Questions"):
            with st.spinner(f"Generating {question_type} questions at {difficulty} difficulty..."):
                # Start timing
                start_time = time.time()
                
                # Generate questions
                questions = generate_questions_with_openai(text, question_type, difficulty, num_questions)
                
                # End timing
                elapsed_time = time.time() - start_time
                
                # Store questions in session state for later use
                st.session_state['generated_questions'] = questions
                st.session_state['question_type'] = question_type
                
                # Success message
                st.success(f"✅ Generated {len(questions)} questions in {elapsed_time:.2f} seconds!")
    
    # Display questions if they exist in session state
    if 'generated_questions' in st.session_state:
        questions = st.session_state['generated_questions']
        question_type = st.session_state['question_type']
        
        st.markdown('<h2 class="sub-header">Generated Questions</h2>', unsafe_allow_html=True)
        
        # Display each question
        for i, q in enumerate(questions, 1):
            st.markdown(f'<div class="question-box">', unsafe_allow_html=True)
            st.markdown(f"**Question {i}:** {q['question']}")
            
            if question_type == "MCQ":
                # Get the options and correct answer
                options = q['options']
                correct_answer = q['answer']
                
                # Display each option
                for j, option in enumerate(options):
                    if option == correct_answer:
                        st.markdown(f'<div class="option-box correct-option">🔘 {chr(65+j)}) {option} ✓</div>', unsafe_allow_html=True)
                    else:
                        st.markdown(f'<div class="option-box">🔘 {chr(65+j)}) {option}</div>', unsafe_allow_html=True)
            else:
                st.markdown(f"**Answer:** {q['answer']}")
            
            st.markdown('</div>', unsafe_allow_html=True)
        
        # Download buttons
        st.markdown('<div class="download-btn">', unsafe_allow_html=True)
        col1, col2 = st.columns(2)
        
        with col1:
            # Download as DOCX
            docx_data = export_to_docx(questions, question_type)
            st.download_button(
                label="📄 Download as DOCX",
                data=docx_data,
                file_name=f"{question_type}_questions.docx",
                mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            )
        
        with col2:
            # Download as CSV
            csv_data = export_to_csv(questions, question_type)
            st.download_button(
                label="📊 Download as CSV",
                data=csv_data,
                file_name=f"{question_type}_questions.csv",
                mime="text/csv",
            )
        st.markdown('</div>', unsafe_allow_html=True)
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Footer
    st.markdown("---")
    st.markdown('<p style="text-align:center">Made with ❤️ by the Exam Question Generator Team</p>', unsafe_allow_html=True)

if __name__ == "__main__":
    main()
