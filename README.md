# Lecture2Exam - AI-Powered Assessment Generator 🧠✨

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://lecture2exam.streamlit.app/)

Transform learning materials into intelligent exam questions using AI. Perfect for educators, students, and professionals!

---

## 🚀 Features

- **Multi-Format Support**: Upload TXT, DOCX, PPTX, and PDF files  
- **AI-Powered Generation**: Google's Gemini API creates context-aware questions  
- **Question Types**:
  - Multiple Choice (MCQ)
  - Fill-in-the-Blanks
  - Short Answer
- **Smart Analytics**:
  - Generation timeline charts
  - Difficulty distribution visualization
  - Historical performance tracking
- **IF-TDF Integration**: Intelligent term weighting for better question relevance
- **Export Capabilities**: Download generated questions as text files

---

## 🛠️ Technologies Used

**Frontend**:  
- Streamlit (Python framework)  
- Plotly for visualizations  

**Backend**:  
- Google Generative AI (Gemini API)  
- IF-TDF (Inverse Frequency - Term Document Frequency) for content analysis  

**Processing**:  
- PyPDF2 (PDF text extraction)  
- python-docx (Word document parsing)  
- python-pptx (PowerPoint parsing)  

---

## 📋 Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/Lecture2Exam.git
cd Lecture2Exam
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Get Google Gemini API key:
- Visit [Google AI Studio](https://aistudio.google.com)
- Create API key and add to `.env` file:
```env
GOOGLE_API_KEY=your_api_key_here
```

---

## 🎮 Usage

- Upload learning materials (multiple files supported)
- Configure settings:
  - Question type (MCQ/Fill-in-the-Blank/Short Answer)
  - Difficulty level (Easy/Medium/Hard)
  - Number of questions
- Generate questions with AI
- Explore analytics in History tab
- Download questions or view in-app

---

## 🌐 Live Demo

Access the deployed application:  
[Open in Streamlit](https://lecture2exam.streamlit.app/)

---

## 📊 IF-TDF Implementation

Our custom Inverse Frequency - Term Document Frequency implementation:

```python
def calculate_if_tdf(content):
    """
    Calculate IF-TDF weights for document terms
    """
    # Preprocess content
    terms = preprocess_text(content)

    # Calculate term frequencies
    tf = Counter(terms)

    # Calculate inverse document frequency
    idf = {term: log(len(documents)/(doc_freq[term]+1)) for term in tf}

    # Combine using IF-TDF formula
    if_tdf = {term: (1/tf[term]) * idf[term] for term in tf}

    return sorted(if_tdf.items(), key=lambda x: x[1], reverse=True)
```

This helps in:
- Identifying key concepts from documents
- Prioritizing important terms for question generation
- Reducing bias towards common words

---

## 📊 Future Roadmap

- Interactive quiz mode
- Question difficulty validation
- Multi-language support
- Collaborative question editing
- Automatic answer validation

---

## 🤝 Contributing

1. Fork the project
2. Create your feature branch:
```bash
git checkout -b feature/AmazingFeature
```
3. Commit changes:
```bash
git commit -m 'Add some AmazingFeature'
```
4. Push to branch:
```bash
git push origin feature/AmazingFeature
```
5. Open a Pull Request

---

## 📜 License

Distributed under MIT License. See `LICENSE` for more information.

---
