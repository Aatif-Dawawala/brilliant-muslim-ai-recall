import streamlit as st
import requests
import json
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
import openai
import os
from dotenv import load_dotenv
from typing import List
from google import genai
from pydantic import BaseModel

load_dotenv()
openaiClient = openai.OpenAI(
    api_key = os.getenv("OPENAI_API_KEY")
)

geminiClient = genai.Client(
    api_key = os.getenv("GEMINI_API_KEY")
)

class outputFormat(BaseModel):
    score: int
    correct_points: list[str]
    incorrect_points: list[str]
    missed_points: list[str]
    generated_feedback: str
    rewritten_answer: str


useGemini = True
useOpenAI = False

VECTOR_PATH = "./rag/vector_store"

def retrieve_chunks(user_response, k=4):
    db = FAISS.load_local(
        VECTOR_PATH, 
        OpenAIEmbeddings(),
        allow_dangerous_deserialization=True
    )
    results = db.similarity_search(user_response, k=k)
    return "\n---\n".join([doc.page_content for doc in results])

def evaluate_response_with_rag(user_response: str, lesson) -> dict:
    retrieved_text = retrieve_chunks(user_response)
    key_points = lesson["key_points"]

    key_points_text = "\n".join(f"- {pt}" for pt in key_points)


    prompt = f"""
You are an Arabic language tutor.

Instructions:
- Identify what the student got right, wrong, and left out.
- Give a score out of 100.
- Provide a brief feedback paragraph.
- Rewrite the student's answer to be more complete and accurate.
- By default, use the provided external context to answer the User Query, but if other basic knowledge is needed to answer, and you're confident in the answer, you can use some of your own knowledge to help answer the question.


Here is the relevant content retrieved from the textbook
<<<
{retrieved_text}
>>>

And here are the key points the student should recall:
{key_points_text}

The student wrote:
\"\"\"{user_response}\"\"\"

Instructions:
- Identify what the student got right, wrong, and left out.
- Give a score out of 100.
- Provide a brief feedback paragraph.
- Rewrite the student's answer to be more complete and accurate.
- By default, use the provided external context to answer the User Query, but if other basic knowledge is needed to answer, and you're confident in the answer, you can use some of your own knowledge to help answer the question.

# Examples:
Example 1: [
Relevant retrieved text:
            Nominative:
            The subject of a verbal sentence is usually in the nominative case. For example, "الولد يدرس" (al-walad yadrus - The boy studies). 
            Accusative:
            The direct object of a verb is in the accusative case. For example, "أنا أقرأ الكتاب" (Ana a'qra' al-kitab - I read the book). 
            Genitive:
            The genitive case is used in several contexts, including after prepositions, when indicating ownership, or when the noun is used with a second noun in a descriptive context (known as idhaafa). 

Key Points:
            "الرفع is primarily used for the subject, predicate, and doer.",
            "النصب is primarily used for the done-to and after حروف which trigger its use.",
            "الجر is primarily used after prepositions."

User recall:
            Arabic has three cases. One of the cases is رفع, another is جر, and finally we have نصب. Raf' is used for subjects, jarr is used for prepositions, and nasb is used for the done-to. 
            These cases help us understand the role of certain words in Arabic

Your response:
            {{
                "score": 70,
                "correct_points": ["الرفع is primarily used for the subject", "النصب is primarily used for the done-to"],
                "incorrect_points": ["You incorrectly stated that الجر is used for prepositions. The correct usage of الجر is after prepositions"],
                "missed_points": ["You didn't mention that النصب is used after حروف which trigger its use", "You didn't mention that الرفع is used for the predicate and doer"],
                "generate_feedback": "You have a solid idea of how cases work in Arabic. You primarily need to focus on the specific details of when the cases are used. I would suggest that you through the lesson once more and focus on the finer details, then try recalling everything you remember a second time. Good luck!",
                "rewritten_answer": "Arabic has three cases. One of the cases is رفع, another is جر, and finally we have نصب.  Raf' is used for the subject, predicate, and done-to; jarr is used after prepositions; nasb is used for the done-to and after حروف which trigger its use. 
                                     These cases help us understand the role of certain words in Arabic"
            }}
]

Response in this JSON format :
{{
    "score": <number>,
    "correct_points": [...],
    "incorrect_points": [...],
    "missed_points": [...],
    "generated_feedback": "...",
    "rewritten_answer": "..."
}}
"""
    if useOpenAI:
        response = openaiClient.chat.completions.create(
            model="gpt-4.1",
            messages=[
                {"role": "system", "content": "You are an expect Arabic language tutor."},
                {"role": "user", "content": prompt}
            ],
            temperature=.2
        )

        content = response.choices[0].message.content
        return json.loads(content)
    
    if useGemini:
        response = geminiClient.models.generate_content(
            model="gemini-2.0-flash",
            contents=prompt,
            config={
                "response_mime_type": "application/json",
                "response_schema": outputFormat,
            }
        )
        content = response.text
        return json.loads(content)

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
        "lesson2": {
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
        "lesson3": {
        "title": "Flexibility",
        "content": """
            ### Flexibility
            Flexibilty is a sub-category of status, and only pertains to words that have an ending sound (as opposed to ending combination).
            - This means that flexibility only pertains to singular words.

            There are three types of flexibility:
            - Fully-flexible
            - Party-flexible
            - Non-flexible
        """,
        "key_points": [
                "Flexbility is a sub-category of status",
                "Flexbility only pertains to singular words",
                "Words may only be fully-flexible, partly-flexible, or non-flexible."
        ]
    },
        "lesson4": {
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
            result = evaluate_response_with_rag(user_input, lesson)
            
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