# Arabic Lesson Evaluator 

An interactive app for Arabic students to practice **active recall** using AI evaluation based on Bayyinah Institute's Dream textbook.

Stack:
- OpenAI GPT
- PDF textbook embedding using FAISS
- Streamlit frontend
- LangChain for vector search

---

## Functionality

1. Displays Arabic language lessons.
2. Users type what they remember about a lesson.
3. The app retrieves relevant content using RAG.
4. AI scores the response and highlights:
    - Correct points
    - Incorrect understanding
    - Missed concepts
    - Feedback

---

## Live Demo
[Launch on Streamlit Cloud](https://arabic-leappn-eval-bo4juht5vqqzqtgfsayval.streamlit.app/)

---

## File Structure

arabic-lesson-eval/
├── streamlit_app.py # Main app
├── rag/
│ ├── textbook.pdf # Dream textbook
│ ├── ingest_pdf.py # Converts PDF to vector DB
│ └── vector_store/ # FAISS vector index
├── requirements.txt
├── .streamlit/
│ └── secrets.toml #
└── README.md