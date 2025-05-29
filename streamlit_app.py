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
- Rewrite the student's answer to be more complete and accurate.

Response in this JSON format:
{{
    "score": <number>,
    "correct_points": [...],
    "incorrect_points": [...],
    "missed_points": [...],
    "generated_feedback": "...",
    "rewritten_answer": "..."
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
            
            # Score + Performance
            score = result['score']
            if score >= 90:
                level = "Excellent"
                color = "green"
            elif score >= 70:
                level = "Good"
                color = "blue"
            else:
                level = "Needs review"
                color = "red"

            st.success(f"Score: {score}/100")
            st.markdown(f"### Performance Level: <span style='color:{color}'>{level}</span>", unsafe_allow_html=True)

            # Feedback paragraph
            st.markdown("---")
            st.markdown("### Feedback Summary")
            st.success(result["generated_feedback"])

            # Detailed analysis
            st.markdown("---")
            st.markdown("### Detailed Evaluation")

            with st.expander("Correct Points"):
                if result["correct_points"]:
                    for pt in result["correct_points"]:
                        st.markdown(f"- {pt}")
                else:
                    st.markdown("_None detected._")

            with st.expander("Incorrect Points"):
                if result["incorrect_points"]:
                    for pt in result["incorrect_points"]:
                        st.markdown(f"- {pt}")
                else:
                    st.markdown("_No misunderstandings identified._")

            with st.expander("Missed Points"):
                if result["missed_points"]:
                    for pt in result["missed_points"]:
                        st.markdown(f"- {pt}")
                else:
                    st.markdown("_No major points were missed._")

            # Rewrite 
            st.markdown("---")
            st.markdown("### Suggested Improved Answer")
            st.info(result["rewritten_answer"])

        except Exception as e:
            st.error(f"Error: {e}")