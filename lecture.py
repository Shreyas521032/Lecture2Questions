import streamlit as st
import random
import re
from datetime import datetime
from io import BytesIO
import csv

# Set page configuration
st.set_page_config(
    page_title="Simple Exam Question Generator",
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

def extract_text_from_txt(txt_file):
    return txt_file.getvalue().decode("utf-8")

def generate_questions_simple(text, question_type, difficulty):
    sentences = [s.strip() for s in re.split(r'[.!?]', text) if len(s.split()) > 5]
    if len(sentences) < 5:
        return {"error": "Text is too short to generate meaningful questions."}
    
    questions = []
    
    if question_type == "MCQ":
        for i in range(5):
            if i >= len(sentences):
                break
                
            sentence = sentences[i]
            words = sentence.split()
            keyword = random.choice([w for w in words if len(w) > 5 and w.isalpha()])
            
            question = f"What is the meaning of '{keyword}' in this context?"
            
            # Create options
            correct = f"The term '{keyword}' refers to {random.choice(['a concept', 'an idea', 'a process'])} described in the text."
            incorrects = [
                f"The term '{keyword}' is not mentioned in the text.",
                f"The term '{keyword}' refers to something completely unrelated.",
                f"The term '{keyword}' is used as an example of something else."
            ]
            
            options = [correct] + incorrects
            random.shuffle(options)
            correct_letter = chr(65 + options.index(correct))
            
            questions.append({
                "question": question,
                "options": [f"{chr(65+i)}. {opt}" for i, opt in enumerate(options)],
                "correct_answer": correct_letter,
                "explanation": f"Option {correct_letter} correctly explains the term '{keyword}'."
            })
    
    elif question_type == "Fill-in-the-Blank":
        for i in range(5):
            if i >= len(sentences):
                break
                
            sentence = sentences[i]
            words = sentence.split()
            keyword = random.choice([w for w in words if len(w) > 5 and w.isalpha()])
            
            blank_sentence = sentence.replace(keyword, "_____")
            questions.append({
                "question": blank_sentence,
                "answer": keyword
            })
    
    else:  # Short Answer
        for i in range(5):
            if i >= len(sentences):
                break
                
            sentence = sentences[i]
            question = f"Explain the following in your own words: '{sentence}'"
            questions.append({
                "question": question,
                "model_answer": f"A good answer would summarize: {sentence[:100]}..."
            })
    
    return {"questions": questions}

def create_docx_simple(questions, question_type):
    from docx import Document
    doc = Document()
    doc.add_heading(f"{question_type} Questions", level=1)
    
    for i, q in enumerate(questions, 1):
        doc.add_paragraph(f"Question {i}: {q['question']}")
        
        if question_type == "MCQ":
            for option in q["options"]:
                doc.add_paragraph(f"    {option}")
            doc.add_paragraph(f"Correct Answer: {q['correct_answer']}")
        elif question_type == "Fill-in-the-Blank":
            doc.add_paragraph(f"Answer: {q['answer']}")
        else:
            doc.add_paragraph(f"Model Answer: {q['model_answer']}")
        
        doc.add_paragraph()
    
    docx_bytes = BytesIO()
    doc.save(docx_bytes)
    docx_bytes.seek(0)
    return docx_bytes

def create_csv_simple(questions, question_type):
    output = BytesIO()
    writer = csv.writer(output)
    
    if question_type == "MCQ":
        writer.writerow(["Question", "Option A", "Option B", "Option C", "Option D", "Correct Answer"])
        for q in questions:
            options = [opt.split(". ", 1)[1] for opt in q["options"]]
            writer.writerow([q["question"]] + options + [q["correct_answer"]])
    elif question_type == "Fill-in-the-Blank":
        writer.writerow(["Question", "Answer"])
        for q in questions:
            writer.writerow([q["question"], q["answer"]])
    else:
        writer.writerow(["Question", "Model Answer"])
        for q in questions:
            writer.writerow([q["question"], q["model_answer"]])
    
    output.seek(0)
    return output

def main():
    st.markdown('<h1 class="main-header">📚 Simple Exam Question Generator</h1>', unsafe_allow_html=True)
    
    with st.sidebar:
        st.markdown("### Upload Lecture Notes")
        uploaded_file = st.file_uploader("Choose a TXT file", type=["txt"])
        
        if uploaded_file:
            question_type = st.selectbox("Question Type", ["MCQ", "Fill-in-the-Blank", "Short Answer"])
            difficulty = st.selectbox("Difficulty", ["Easy", "Medium"])
            generate_button = st.button("Generate Questions")
            export_format = st.selectbox("Export Format", ["DOCX", "CSV"])

    if uploaded_file:
        text = extract_text_from_txt(uploaded_file)
        
        with st.expander("Preview Text"):
            st.text(text[:500] + "...")
        
        if generate_button:
            questions_data = generate_questions_simple(text, question_type, difficulty)
            
            if "error" in questions_data:
                st.error(questions_data["error"])
            else:
                st.session_state.questions = questions_data
        
        if 'questions' in st.session_state:
            st.markdown(f"## Generated {question_type} Questions")
            
            for i, q in enumerate(st.session_state.questions["questions"], 1):
                st.markdown(f'<div class="question-box">', unsafe_allow_html=True)
                st.markdown(f"**Question {i}:** {q['question']}")
                
                if question_type == "MCQ":
                    for option in q["options"]:
                        st.markdown(option)
                    st.markdown(f"**Correct Answer:** {q['correct_answer']}")
                elif question_type == "Fill-in-the-Blank":
                    st.markdown(f"**Answer:** {q['answer']}")
                else:
                    st.markdown(f"**Model Answer:** {q['model_answer']}")
                
                st.markdown('</div>', unsafe_allow_html=True)
            
            # Export buttons
            col1, col2 = st.columns(2)
            
            with col1:
                if export_format == "DOCX":
                    docx_bytes = create_docx_simple(st.session_state.questions["questions"], question_type)
                    st.download_button(
                        label="Download DOCX",
                        data=docx_bytes,
                        file_name=f"questions_{question_type}.docx",
                        mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document"
                    )
                else:
                    csv_bytes = create_csv_simple(st.session_state.questions["questions"], question_type)
                    st.download_button(
                        label="Download CSV",
                        data=csv_bytes,
                        file_name=f"questions_{question_type}.csv",
                        mime="text/csv"
                    )
    else:
        st.info("Please upload a text file to generate questions")

if __name__ == "__main__":
    main()
