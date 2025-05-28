import streamlit as st
import requests
import json
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
import openai
import os
from dotenv import load_dotenv
from typing import List

load_dotenv()
client = openai.OpenAI(
    api_key = os.getenv("OPENAI_API_KEY")
)

VECTOR_PATH = "./rag/vector_store"

def retrieve_chunks(user_response, k=4):
    db = FAISS.load_local(
        VECTOR_PATH, 
        OpenAIEmbeddings(),
        allow_dangerous_deserialization=True
    )
    results = db.similarity_search(user_response, k=k)
    return "\n---\n".join([doc.page_content for doc in results])

def evaluate_response_with_rag(user_response: str) -> dict:
    retrieved_text = retrieve_chunks(user_response)


    prompt = f"""
You are an Arabic language tutor.

Here is the relevant content retrieved from the textbook
<<<
{retrieved_text}
>>>

The student wrote:
\"\"\"{user_response}\"\"\"

Instructions:
- Identify what the student got right, wrong, and left out.
- Give a score out of 100.
- Provide a brief feedback paragraph.

Response in this JSON format:
{{
    "score": <number>,
    "correct_points": [...],
    "incorrect_points": [...],
    "missed_points": [...],
    "generated_feedback": "..."
}}
"""
    response = client.chat.completions.create(
        model="gpt-4.1",
        messages=[
            {"role": "system", "content": "You are an expect Arabic language tutor."},
            {"role": "user", "content": prompt}
        ],
        temperature=.2
    )

    content = response.choices[0].message.content
    return json.loads(content)

API_URL = "http://127.0.0.1:8000/evaluate"

LESSONS = {
    "lesson1": {
        "title": "اسم الإشارة",
        "content": """
### اسم الإشارة
The اسماء الإشارة include:
- هذا
- هذه
- ذلك
- تلك

They are used as pointer words, and must match in gender and number with what they are pointing at.
""",
    }
}

st.set_page_config(page_title="Arabic Lesson Recall", layout="wide")
st.title("Arabic Lesson Recall")

# Select
lesson_id = st.selectbox("Choose a lesson:", options=list(LESSONS.keys()), format_func=lambda k: LESSONS[k]["title"])
lesson = LESSONS[lesson_id]

st.markdown(lesson["content"], unsafe_allow_html=True)

# Input
st.subheader("What do you remember?")
user_input = st.text_area("Type everything you recall from this lesson", height=200)

if st.button("Evaluate Response"):
    with st.spinner("Evaluating with AI..."):
        try: 
            result = evaluate_response_with_rag(user_input)

            st.success(f"Your Score: {result['score']}/100")
            st.subheader("Correct Poitns")
            for pt in result["correct_points"]:
                st.markdown(f"- {pt}")
            
            st.subheader("Incorrect Points")
            for pt in result["incorrect_points"]:
                st.markdown(f"- {pt}")

            st.subheader("Feedback")
            st.info(result["generated_feedback"])
        except Exception as e:
            st.error(f"Error: {e}")