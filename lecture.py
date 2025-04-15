import streamlit as st
from io import BytesIO
import google.generativeai as genai
import pandas as pd
from datetime import datetime
import plotly.express as px

# Set page configuration
st.set_page_config(
    page_title="Lecture2Exam",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Enhanced CSS styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.8rem;
        color: #2563EB;
        text-align: center;
        margin-bottom: 2rem;
        font-weight: bold;
        text-shadow: 1px 1px 3px rgba(0,0,0,0.15);
        padding: 1rem;
        background: linear-gradient(90deg, #EFF6FF 0%, #DBEAFE 100%);
        border-radius: 12px;
    }
    .question-box {
        background-color: #F0F9FF;
        border-left: 5px solid #0EA5E9;
        padding: 20px;
        margin-bottom: 25px;
        border-radius: 8px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.08);
        transition: transform 0.2s ease;
    }
    .stats-card {
        background: linear-gradient(145deg, #DBEAFE 0%, #E0F2FE 100%);
        padding: 20px;
        border-radius: 12px;
        margin-bottom: 20px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.05);
    }
    .file-item {
        display: flex;
        justify-content: space-between;
        padding: 8px;
        margin-bottom: 5px;
        background-color: white;
        border-radius: 6px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.1);
    }
    /* Full CSS remains the same as previous version */
</style>
""", unsafe_allow_html=True)

# Session state initialization
if 'generation_history' not in st.session_state:
    st.session_state.generation_history = []

if 'uploaded_files' not in st.session_state:
    st.session_state.uploaded_files = []

def extract_text(file):
    # Existing extract_text function remains the same
    # ...
    try:
        if file.type == "text/plain":
            return file.getvalue().decode("utf-8")
        
        elif file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
            from docx import Document
            doc = Document(BytesIO(file.read()))
            return "\n".join([para.text for para in doc.paragraphs])
        
        elif file.type == "application/vnd.openxmlformats-officedocument.presentationml.presentation":
            from pptx import Presentation
            prs = Presentation(BytesIO(file.read()))
            text = []
            for slide in prs.slides:
                for shape in slide.shapes:
                    if hasattr(shape, "text"):
                        text.append(shape.text)
            return "\n".join(text)
        
        elif file.type == "application/pdf":
            import PyPDF2
            pdf_reader = PyPDF2.PdfReader(BytesIO(file.read()))
            text = ""
            for page in pdf_reader.pages:
                text += page.extract_text() + "\n"
            return text
        
        else:
            st.error("Unsupported file type")
            return ""
    
    except Exception as e:
        st.error(f"Error processing file: {str(e)}")
        return ""

def format_questions(raw_questions):
    # Existing format_questions function remains the same
    # ...
    """Format the questions with better spacing and styling"""
    if not raw_questions:
        return ""
    
    # Split by numbered questions
    import re
    questions = re.split(r'\n\s*(\d+)\.\s+', raw_questions)
    
    if len(questions) <= 1:
        return raw_questions
    
    formatted = ""
    
    # Skip the first empty element if it exists
    start_idx = 1 if questions[0].strip() == "" else 0
    
    for i in range(start_idx, len(questions), 2):
        if i+1 < len(questions):
            question_num = questions[i]
            question_content = questions[i+1].strip()
            
            # Add extra spacing between parts
            question_content = question_content.replace("Question:", "\nQuestion:")
            question_content = question_content.replace("Options:", "\nOptions:")
            question_content = question_content.replace("Correct Answer:", "\n\nCorrect Answer:")
            question_content = question_content.replace("Explanation:", "\n\nExplanation:")
            
            formatted += f'<div class="question-box">\n'
            formatted += f'<span class="question-number">🔍 {question_num}.</span>\n'
            formatted += f'{question_content}\n'
            formatted += f'</div>\n\n'
    
    return formatted

def generate_with_gemini(text, question_type, difficulty, api_key, num_questions=5):
    # Existing generate_with_gemini function remains the same
    # ...
    try:
        genai.configure(api_key=api_key)
        
        # Use the correct model name - gemini-1.5-pro-latest or gemini-1.0-pro
        model = genai.GenerativeModel('gemini-1.5-pro-latest')
        
        prompt = f"""
        Generate {num_questions} {difficulty.lower()} {question_type} questions based on this content.
        Format should be:
        
        1. Question: [question text]
           Options (if MCQ): A) [option1] B) [option2] C) [option3] D) [option4]
           Correct Answer: [correct answer]
           Explanation: [brief explanation]
        
        Content:
        {text[:12000]}  # Using first 12k chars to avoid token limits
        """
        
        response = model.generate_content(prompt)
        
        if not response.text:
            raise ValueError("Empty response from API")
            
        return response.text
    
    except Exception as e:
        st.markdown(f'<div class="error-message">⚠️ API error: {str(e)}</div>', unsafe_allow_html=True)
        st.info("🔑 Please ensure you're using the correct API key and model name")
        st.info("💡 Try using 'gemini-1.0-pro' if this model isn't available")
        return None

def main():
    st.markdown('<h1 class="main-header">🧠 Lecture2Exam ✨</h1>', unsafe_allow_html=True)
    
    # File removal handling
    if 'remove_index' not in st.session_state:
        st.session_state.remove_index = -1
    
    # Create tabs
    tab1, tab2 = st.tabs(["📝 Generate Questions", "📊 History & Analytics"])
    
    with tab1:
        col1, col2 = st.columns([1, 2])
        
        with col1:
            # API Configuration
            st.markdown('<div class="config-section">', unsafe_allow_html=True)
            api_key = st.text_input("🔑 API Key", type="password", 
                                   help="Get your key from Google AI Studio")
            st.markdown('</div>', unsafe_allow_html=True)
            
            # File Uploader
            st.markdown('<div class="file-uploader">', unsafe_allow_html=True)
            uploaded_files = st.file_uploader("📄 Upload Learning Material", 
                                            type=["txt", "docx", "pptx", "pdf"],
                                            accept_multiple_files=True)
            
            # Store uploaded files
            if uploaded_files:
                for file in uploaded_files:
                    if file.name not in [f['name'] for f in st.session_state.uploaded_files]:
                        st.session_state.uploaded_files.append({
                            'name': file.name,
                            'size': f"{len(file.getvalue()) / 1024:.1f} KB",
                            'file_obj': file
                        })
            
            # Display uploaded files with removal
            if st.session_state.uploaded_files:
                st.markdown("### 📂 Selected Files")
                st.markdown('<div class="file-list">', unsafe_allow_html=True)
                for i, file in enumerate(st.session_state.uploaded_files):
                    st.markdown(
                        f'<div class="file-item">'
                        f'<span class="file-name">{file["name"]}</span>'
                        f'<span class="file-size">{file["size"]}</span>'
                        f'<span class="remove-btn" onclick="removeFile({i})">🗑️</span>'
                        f'</div>',
                        unsafe_allow_html=True
                    )
                st.markdown('</div>', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)
            
            # Question Configuration
            if st.session_state.uploaded_files:
                st.markdown('<div class="config-section">', unsafe_allow_html=True)
                question_type = st.selectbox("📋 Question Type", ["MCQ", "Fill-in-the-Blank", "Short Answer"])
                difficulty = st.select_slider("🎯 Difficulty Level", options=["Easy", "Medium", "Hard"])
                num_questions = st.slider("🔢 Number of Questions", 3, 15, 5)
                st.markdown('</div>', unsafe_allow_html=True)
                
                # Generation controls
                gen_col1, gen_col2 = st.columns(2)
                with gen_col1:
                    generate_btn = st.button("✨ Generate Questions", use_container_width=True)
                with gen_col2:
                    if st.button("🗑️ Clear Results", use_container_width=True):
                        if 'questions' in st.session_state:
                            del st.session_state.questions
                
                if generate_btn and api_key:
                    with st.spinner("🧠 Generating intelligent questions..."):
                        combined_text = ""
                        for file_info in st.session_state.uploaded_files:
                            text = extract_text(file_info['file_obj'])
                            combined_text += f"\n\n--- {file_info['name']} ---\n\n{text}"
                        
                        if combined_text:
                            questions = generate_with_gemini(combined_text[:12000], question_type, 
                                                           difficulty, api_key, num_questions)
                            if questions:
                                st.session_state.questions = questions
                                st.session_state.formatted_questions = format_questions(questions)
                                st.session_state.generation_history.append({
                                    'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M"),
                                    'file': ", ".join([f['name'] for f in st.session_state.uploaded_files]),
                                    'type': question_type,
                                    'difficulty': difficulty,
                                    'count': num_questions,
                                    'questions': questions
                                })

        with col2:
            # Results display
            if 'questions' in st.session_state and st.session_state.questions:
                st.markdown(f"<h2 class='subheader'>✅ Generated {question_type} Questions</h2>", 
                           unsafe_allow_html=True)
                st.markdown(st.session_state.formatted_questions, unsafe_allow_html=True)
                
                # Download controls
                col_a, col_b = st.columns(2)
                with col_a:
                    st.download_button(
                        label="📥 Download Questions",
                        data=st.session_state.questions.encode('utf-8'),
                        file_name=f"questions_{datetime.now().strftime('%Y%m%d%H%M')}.txt",
                        mime="text/plain"
                    )
                with col_b:
                    if st.button("🔄 Regenerate Questions"):
                        del st.session_state.questions
    
    with tab2:
        # Analytics Tab
        st.markdown("<h2 class='subheader'>📊 Generation Analytics</h2>", unsafe_allow_html=True)
        
        if st.session_state.generation_history:
            history_df = pd.DataFrame(st.session_state.generation_history)
            
            # Stats Cards
            col1, col2, col3 = st.columns(3)
            with col1:
                st.markdown(f"""
                <div class="stats-card">
                <h4>🔄 Total Generations</h4>
                <h2>{len(history_df)}</h2>
                </div>
                """, unsafe_allow_html=True)
            with col2:
                st.markdown(f"""
                <div class="stats-card">
                <h4>📑 Files Processed</h4>
                <h2>{len(history_df['file'].unique())}</h2>
                </div>
                """, unsafe_allow_html=True)
            with col3:
                st.markdown(f"""
                <div class="stats-card">
                <h4>🧩 Total Questions</h4>
                <h2>{history_df['count'].sum()}</h2>
                </div>
                """, unsafe_allow_html=True)
            
            # Visualizations
            history_df['date'] = pd.to_datetime(history_df['timestamp']).dt.date
            time_series = history_df.groupby('date')['count'].sum().reset_index()
            
            fig1 = px.line(time_series, x='date', y='count', 
                          title='Questions Generated Over Time',
                          markers=True)
            st.plotly_chart(fig1, use_container_width=True)
            
            fig2 = px.pie(history_df, names='difficulty', 
                         title='Difficulty Distribution',
                         color_discrete_sequence=px.colors.qualitative.Pastel)
            st.plotly_chart(fig2, use_container_width=True)
            
            # History Table
            st.markdown("### 📜 Recent Activity")
            st.dataframe(history_df[['timestamp', 'file', 'type', 'difficulty', 'count']]
                        .tail(10)
                        .rename(columns={
                            'timestamp': '⏰ Time',
                            'file': '📄 File',
                            'type': '📋 Type',
                            'difficulty': '🎯 Difficulty',
                            'count': '🔢 Count'
                        }), use_container_width=True)
            
            if st.button("🗑️ Clear History"):
                st.session_state.generation_history = []
                st.experimental_rerun()
        else:
            st.info("📭 No generation history available")

    # File removal JavaScript
    st.markdown("""
    <script>
    function removeFile(index) {
        window.parent.postMessage({
            type: 'streamlit:setComponentValue',
            value: index
        }, '*');
    }
    </script>
    """, unsafe_allow_html=True)
    
    # File removal handler
    remove_index = st.number_input('', min_value=-1, value=-1, key='remove_index')
    if remove_index >= 0 and remove_index < len(st.session_state.uploaded_files):
        st.session_state.uploaded_files.pop(remove_index)
        st.session_state.remove_index = -1
        st.experimental_rerun()

if __name__ == "__main__":
    main()
