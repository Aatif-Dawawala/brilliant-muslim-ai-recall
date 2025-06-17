# Arabic Lesson Evaluator 

An interactive app for Arabic students to practice **active recall** using AI evaluation based on Bayyinah Institute's Dream textbook.

Stack:
- Gemini 2.5 Pro & OpenAI GPT 4.1
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

## Local Setup

### Clone the Repo
```
bash git clone https://github.com/aatif-dawawala/arabic-lesson-eval.git
cd arabic-lesson-eval
```

### Install Dependencies
```pip install -r requirements.txt```

### Add API Key
Create a ```.env``` file or ```.streamlist/secrets.toml```:
```
    OPENAI_API_KEY=your-key-here
    GEMINI_API_KEY=your-key-here
```

### Run Locally
```streamlit run streamlit_app.py```

## License 
MIT

## Contributing
Pull requests are welcome. Let's help Arabic learners around the world!
