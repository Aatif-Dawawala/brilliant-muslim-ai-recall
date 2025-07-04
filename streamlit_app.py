import streamlit as st
import requests
import json
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
import os
from dotenv import load_dotenv
from typing import List
from prompt_templates import build_rag_prompt
from evaluation_logger import append_example
from model_switcher import evaluate, OutputFormat

load_dotenv()

VECTOR_PATH = "./rag/vector_store"

def retrieve_chunks(user_response, k=4):
    db = FAISS.load_local(
        VECTOR_PATH, 
        OpenAIEmbeddings(),
        allow_dangerous_deserialization=True
    )
    results = db.similarity_search(user_response, k=k)
    return "\n---\n".join([doc.page_content for doc in results])

def evaluate_response_with_rag(user_response: str, lesson, model_choice: str) -> dict:
    retrieved_text = retrieve_chunks(user_response)
    key_points = lesson["key_points"]

    prompt = build_rag_prompt(user_response, retrieved_text, key_points)
    result = evaluate(prompt, model_choice)
    append_example(prompt, result)
    return result

API_URL = "http://127.0.0.1:8000/evaluate"

LESSONS = {
    "lesson1": {
        "title": "الإعراب",
        "content": """
            ### الإعراب
            The Arabic cases are:
            - الرفع
                - Primarily used for the subject, predicate, and doer.
            - النصب
                - Primarily used for the done-to and after حروف which trigger it's use.
            - الجر
                - Primarily used after prepositions.

            Cases in Arabic are applied through case markers at the end of words. They help signify what role a word plays in a sentence.
        """,
        "key_points": [
            "الرفع is primarily used for the subject, predicate, and doer.",
            "النصب is primarily used for the done-to and after حروف which trigger its use.",
            "الجر is primarily used after prepositions."
        ]
    },
        "lesson2": {
        "title": "How to tell status",
        "content": """
            ### Determining Status
            Status is generally determined by the ending sound or combination.
            
            - Singular words that end with a ضمة or ضمتان generally are الرفع.
            - Singular words that end with a كسرة or كسرتان generally are الجر.
            - Singular words that end with a فتحة or فتحتان generally are النصب. 
        """,
        "key_points": [
            "Singular words that end with a ضمة or ضمتان generally are الرفع.",
            "Singular words that end with a كسرة or كسرتان generally are الجر.",
            "Singular words that end with a فتحة or فتحتان generally are النصب."         ]
    },
        "lesson3": {
        "title": "Light vs. Heavy",
        "content": """
            ### Understanding Light and Heavy Words
            Words can either be light or heavy, with heavy being the default.
            In order to make a word light, simply remove the ن at the end.

            For example:
            - مسلمٌ <- مسلمُ
            - مسلمان <- مسلما
            - مسلمون <- مسلمو

            - Words are never light unless there is a specific reason for them to be.
        """,
        "key_points": [
            "Words are heavy by default",
            "To make a word light, they ن must be removed."
            "Words are never light unless there is a specific reason for them to be."
        ]
    },
        "lesson4": {
        "title": "Flexibility",
        "content": """
            ### Flexibility
            Flexibility is a sub-category of status, and only pertains to words that have an ending sound (as opposed to ending combination).
            - This means that flexibility only pertains to singular words.

            There are three types of flexibility:
            - Fully-flexible
            - Partly-flexible
            - Non-flexible
        """,
        "key_points": [
                "Flexibility is a sub-category of status",
                "Flexibility only pertains to singular words",
                "Words may only be fully-flexible, partly-flexible, or non-flexible."
        ]
    },
        "lesson5": {
        "title": "Pronouns",
        "content": """
            ### Pronouns in Arabic
            Arabic has 1st person, 2nd person, and 3rd person pronouns. The 1st person pronouns have the singular and plural form, and the 2nd/3rd person pronouns have singular, dual, and plural forms.   

            Pronouns may take three forms:
            - الضمير المستتر
                - Pronouns within أفعال.
            - الضمير المنفصل
                - Pronouns independently standing by themselves.
            - الضمير المتصل
                - Attached pronouns.
            """,
        "key_points": [
                "Arabic has 1st person, 2nd person, and 3rd person pronouns",
                "The 1st person pronouns have singular and plural forms.",
                "The 2nd/3rd person pronouns have singular, dual, and plural forms.",
                "There are three forms that pronouns can take: الضمير المستتر, الضمير المنفصل, الضمير المتصل"
        ]
    }
}

st.set_page_config(page_title="Arabic Lesson Recall", layout="wide")
st.title("Arabic Lesson Recall")

# Initialize recall flag
if "show_recall" not in st.session_state:
    st.session_state.show_recall = False

# Select
model_choice = st.selectbox(
    "Choose evaluation model:",
    options=["Gemini", "OpenAI"],
    index=0
)
lesson_id = st.selectbox("Choose a lesson:", options=list(LESSONS.keys()), format_func=lambda k: LESSONS[k]["title"])
lesson = LESSONS[lesson_id]

st.markdown(lesson["content"], unsafe_allow_html=True)

if not st.session_state.show_recall:
    if st.button("Start Recall"):
        st.session_state.show_recall = True
        st.rerun()
else:
    # Input
    st.subheader("What do you remember?")
    user_input = st.text_area(
        "Type everything you recall from this lesson", height=200
    )

    if st.button("Evaluate Response"):
        with st.spinner("Evaluating..."):
            try:
                result = evaluate_response_with_rag(user_input, lesson, model_choice)
                print(result)
            
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
                st.markdown(
                    f"### Performance Level: <span style='color:{color}'>{level}</span>",
                    unsafe_allow_html=True,
                )

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
