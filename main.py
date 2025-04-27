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

st.markdown("""
<style>
    /* Enhanced Main Header with Animation */
    .main-header {
        font-size: 3.2rem;
        color: #2563EB;
        text-align: center;
        margin-bottom: 2rem;
        font-weight: 700;
        background: linear-gradient(120deg, #3B82F6 0%, #1E40AF 100%);
        background-clip: text;
        -webkit-background-clip: text;
        color: transparent;
        padding: 1.2rem 0;
        position: relative;
        overflow: hidden;
    }
    .main-header::after {
        content: "";
        position: absolute;
        bottom: 0;
        left: 0;
        width: 100%;
        height: 4px;
        background: linear-gradient(90deg, #3B82F6, #60A5FA, #3B82F6);
        background-size: 200% 100%;
        animation: gradient-shift 3s ease infinite;
    }
    @keyframes gradient-shift {
        0% {background-position: 0% 50%}
        50% {background-position: 100% 50%}
        100% {background-position: 0% 50%}
    }
    
    /* Interactive File Cards */
    .file-card {
        background: white;
        border-radius: 10px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.08);
        padding: 15px;
        margin-bottom: 15px;
        display: flex;
        justify-content: space-between;
        align-items: center;
        transition: all 0.3s ease;
        border-left: 4px solid #3B82F6;
    }
    .file-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 8px 15px rgba(0,0,0,0.12);
    }
    
    /* Improved Question Box with Animation */
    .question-box {
        background: linear-gradient(to right, #F0F9FF 0%, #EFF6FF 100%);
        border-left: 5px solid #0EA5E9;
        padding: 20px;
        margin-bottom: 25px;
        border-radius: 8px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.08);
        transition: all 0.3s cubic-bezier(0.25, 0.8, 0.25, 1);
        position: relative;
        overflow: hidden;
    }
    .question-box:hover {
        transform: translateY(-3px) scale(1.01);
        box-shadow: 0 8px 20px rgba(0,0,0,0.12);
    }
    .question-box::before {
        content: "";
        position: absolute;
        top: 0;
        left: 0;
        width: 100%;
        height: 100%;
        background: linear-gradient(120deg, rgba(59, 130, 246, 0.1) 0%, rgba(14, 165, 233, 0.05) 100%);
        transform: translateX(-100%);
        transition: transform 0.5s ease;
    }
    .question-box:hover::before {
        transform: translateX(0);
    }
    
    /* Pulsing Action Button */
    .action-button button {
        position: relative;
        overflow: hidden;
    }
    .action-button button::after {
        content: "";
        position: absolute;
        top: 50%;
        left: 50%;
        width: 5px;
        height: 5px;
        background: rgba(255, 255, 255, 0.5);
        opacity: 0;
        border-radius: 100%;
        transform: scale(1, 1) translate(-50%, -50%);
        transform-origin: 50% 50%;
    }
    .action-button button:hover::after {
        animation: ripple 1s ease-out;
    }
    @keyframes ripple {
        0% {
            transform: scale(0, 0);
            opacity: 0.5;
        }
        100% {
            transform: scale(20, 20);
            opacity: 0;
        }
    }
</style>
""", unsafe_allow_html=True)

# Enhanced file management
def handle_multiple_files():
    # File uploading interface
    st.markdown('<div class="file-uploader">', unsafe_allow_html=True)
    st.markdown("### 📄 Upload Learning Materials")
    
    uploaded_files = st.file_uploader("Choose files", 
                                   type=["txt", "docx", "pptx", "pdf"],
                                   accept_multiple_files=True)
    
    # Process newly uploaded files
    if uploaded_files:
        for file in uploaded_files:
            file_hash = hash(file.name + str(file.size))
            if file_hash not in [f['hash'] for f in st.session_state.uploaded_files]:
                # Extract preview text for display
                preview = extract_text(file)[:100] + "..." if len(extract_text(file)) > 100 else extract_text(file)
                
                st.session_state.uploaded_files.append({
                    'name': file.name,
                    'size': f"{len(file.getvalue()) / 1024:.1f} KB",
                    'hash': file_hash,
                    'file_obj': file,
                    'preview': preview,
                    'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M"),
                    'type': file.type
                })
    
    # Display uploaded files with interactive cards
    if st.session_state.uploaded_files:
        st.markdown("### 📂 Your Materials")
        
        for i, file in enumerate(st.session_state.uploaded_files):
            col1, col2 = st.columns([4, 1])
            
            with col1:
                expander = st.expander(f"📄 {file['name']} ({file['size']})")
                with expander:
                    st.write(f"**Type:** {file['type'].split('/')[-1].upper()}")
                    st.write(f"**Added:** {file['timestamp']}")
                    st.write("**Preview:**")
                    st.markdown(f"<div style='background:#f5f5f5;padding:10px;border-radius:5px;'>{file['preview']}</div>", unsafe_allow_html=True)
            
            with col2:
                if st.button(f"🗑️ Remove", key=f"remove_{i}"):
                    st.session_state.uploaded_files.pop(i)
                    st.experimental_rerun()
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Return true if files are available
    return len(st.session_state.uploaded_files) > 0

def generate_questions(api_key, question_type, difficulty, num_questions):
    with st.spinner("🧙‍♂️ Creating intelligent questions..."):
        # Combine content from all files with better organization
        combined_text = ""
        for file_info in st.session_state.uploaded_files:
            file = file_info['file_obj']
            text = extract_text(file)
            if text:
                # Add better file context separation
                combined_text += f"\n\n==== CONTENT FROM: {file.name} ====\n\n{text}\n\n"
        
        if not combined_text:
            st.error("⚠️ Could not extract text from the files")
            return None
            
        # Configure Gemini AI
        genai.configure(api_key=api_key)
        
        try:
            # Try with latest model first
            model = genai.GenerativeModel('gemini-1.5-pro-latest')
            
            # Enhanced prompt for better question quality
            prompt = f"""
            Generate {num_questions} {difficulty.lower()} {question_type} questions based on the content I'm providing.
            
            Guidelines:
            - Focus on key concepts and important details
            - Ensure questions test understanding, not just memorization
            - Include a mix of recall and application questions
            - Make explanations clear and educational
            
            Format each question as:
            
            [#]. Question: [question text]
                Options (for MCQ): A) [option1] B) [option2] C) [option3] D) [option4]
                Correct Answer: [correct answer]
                Explanation: [concise explanation]
            
            Content:
            {combined_text[:15000]}
            """
            
            response = model.generate_content(prompt)
            
            if not response.text:
                # Fallback to gemini-1.0-pro if needed
                model = genai.GenerativeModel('gemini-1.0-pro')
                response = model.generate_content(prompt)
                
            return response.text
                
        except Exception as e:
            st.markdown(f'<div class="error-message">⚠️ API error: {str(e)}</div>', unsafe_allow_html=True)
            return None

def main():
    # Initialize session state
    if 'generation_history' not in st.session_state:
        st.session_state.generation_history = []
    if 'uploaded_files' not in st.session_state:
        st.session_state.uploaded_files = []
    if 'questions' not in st.session_state:
        st.session_state.questions = None

    # App header with animated effects
    st.markdown('<h1 class="main-header">🧠 Lecture2Exam ✨</h1>', unsafe_allow_html=True)
    
    # Create tabs with improved UX
    tab1, tab2 = st.tabs(["📝 Create Questions", "📊 History & Analytics"])
    
    with tab1:
        col1, col2 = st.columns([1, 2])
        
        with col1:
            # API Configuration
            with st.container():
                st.markdown("### 🔑 API Configuration")
                api_key = st.text_input("API Key", type="password", 
                                      help="Get your key from Google AI Studio")
            
            # Handle file uploads
            files_available = handle_multiple_files()
            
            # Question settings
            if files_available:
                with st.container():
                    st.markdown("### ⚙️ Question Settings")
                    
                    question_type = st.selectbox("Question Type", 
                                              ["MCQ", "Fill-in-the-Blank", "Short Answer"])
                    
                    difficulty = st.select_slider(
                        "Difficulty Level",
                        options=["Easy", "Medium", "Hard"],
                        format_func=lambda x: f"{'🟢' if x == 'Easy' else '🟡' if x == 'Medium' else '🔴'} {x}"
                    )
                    
                    num_questions = st.slider("Number of Questions", 
                                           min_value=3, max_value=15, value=5)
                
                # Action buttons
                st.markdown('<div class="action-button">', unsafe_allow_html=True)
                generate_btn = st.button("✨ Generate Questions", use_container_width=True)
                st.markdown('</div>', unsafe_allow_html=True)
                
                if generate_btn:
                    if not api_key:
                        st.error("⚠️ Please enter your API key")
                    else:
                        questions = generate_questions(api_key, question_type, difficulty, num_questions)
                        if questions:
                            st.session_state.questions = questions
                            st.session_state.formatted_questions = format_questions(questions)
                            
                            # Add to history
                            file_names = ", ".join([f['name'] for f in st.session_state.uploaded_files])
                            st.session_state.generation_history.append({
                                'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M"),
                                'file': file_names,
                                'type': question_type,
                                'difficulty': difficulty,
                                'count': num_questions,
                                'questions': questions
                            })
        
        with col2:
            if files_available and st.session_state.questions:
                st.markdown(f"<h2 class='subheader'>✅ Generated {question_type} Questions</h2>", unsafe_allow_html=True)
                
                # Show formatted questions
                st.markdown(st.session_state.formatted_questions, unsafe_allow_html=True)
                
                col_a, col_b = st.columns(2)
                with col_a:
                    st.download_button(
                        label="📥 Download Questions",
                        data=st.session_state.questions,
                        file_name=f"{question_type.lower().replace(' ', '_')}_questions.txt",
                        mime="text/plain",
                        use_container_width=True
                    )
                with col_b:
                    if st.button("🔄 Regenerate", use_container_width=True):
                        questions = generate_questions(api_key, question_type, difficulty, num_questions)
                        if questions:
                            st.session_state.questions = questions
                            st.session_state.formatted_questions = format_questions(questions)
            else:
                # Welcome message with animation
                st.markdown("""
                <div style="background: linear-gradient(135deg, #EFF6FF 0%, #DBEAFE 100%); 
                           padding: 30px; border-radius: 12px; text-align: center; 
                           margin-top: 50px; box-shadow: 0 4px 15px rgba(0,0,0,0.05);
                           animation: pulse 2s infinite ease-in-out;">
                    <h3>🚀 Ready to Create Exam Questions?</h3>
                    <p>Upload your lecture materials and let AI transform them into quality assessment questions.</p>
                    <div style="font-size: 3.5rem; margin: 25px 0; opacity: 0.85;">
                        📚 → 🧠 → 📝
                    </div>
                    <p>Perfect for educators, students, and learning professionals.</p>
                </div>
                <style>
                @keyframes pulse {
                    0% {transform: scale(1);}
                    50% {transform: scale(1.02);}
                    100% {transform: scale(1);}
                }
                </style>
                """, unsafe_allow_html=True)
    
    # History tab implementation remains similar but more compact
    with tab2:
        display_history_analytics()
        
    # Footer with animation
    st.markdown("""
    <div class="footer" style="text-align: center; margin-top: 40px; padding: 20px; 
                             border-top: 1px solid #E2E8F0; animation: fadeIn 1s ease-in;">
        🧠 Lecture2Exam<br>
        Making assessment creation effortless with AI ✨<br>
        <small>Made with ❤️ by Shreyas, Shaurya and Mahati</small>
    </div>
    <style>
    @keyframes fadeIn {
        from {opacity: 0;}
        to {opacity: 1;}
    }
    </style>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
  main()
