import streamlit as st
import requests
import json
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
import os
from dotenv import load_dotenv
from typing import List
from prompt_templates import build_rag_prompt
from evaluation_logger import append_example_mongo
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
    append_example_mongo(prompt, result, user_response, lesson["title"])
    return result

API_URL = "http://127.0.0.1:8000/evaluate"

LESSONS = {
    "lesson1": {
        "title": "الإعراب",
        "content": """
            ### الإعراب
            #### Introduction
            Status is the first of the four properties. Status has to do with the role an اسم is playing in a sentence. In
            Arabic, an اسم can have one of three statuses. The status depends on the role the اسم is playing.

            #### الإعراب
            Status is the first of the four properties. Status has to do with the role an اسم is playing in a sentence. In
            Arabic, an اسم can have one of three statuses. The status depends on the role the اسم is playing.

            ##### رفع
            The doer is the one who carries out the action. Take a look at the following examples. 

            **I ate too much chocolate.**

            *The action here is “ate”. Now ask yourself who it was who ate. It is the speaker “I” who did the 
            action. In this sentence “I” is the doer.*

            **My tooth in aching.**

            *The action here is “aching”. Now asking yourself what is doing the aching. It is the tooth. In this 
            sentence “tooth” is the doer.*

            **The dentist gave me a filling.**

            *The action here is “gave”. Now ask yourself who is the one who gave. It is the dentist. In this 
            sentence, “dentist” is the doer.*

            When searching for the doer in a sentence, follow a two-step process: 
            1. Identify the action
            2. Ask yourself “Who is doing the action?”

            Note that it is possible for the doer to be non-human. 

            رفع is also known as the default status. If there is no reason for an اسم to be put in another status, it
            remains in the رفع status.

            The doer is always in the رَفْعٌ  status. The way you say “in the رَفْع status” in Arabic is مرفوع. *Memorize* this 
            term and use it.

            #### نصب
            The detail refers to additional information about the action. When looking for a detail in a sentence, 
            follow a two-step process: 
            1. Find the action and the doer
            2. Everything else in the sentence is a detail

            The detail is always in the نصب status. The way you say “in the نَصْب status” in Arabic is منصوب. *Memorize* 
            this term and use it. 

            #### جر
            This is the status of words that come after "of".
            """,

        "key_points": [
            "الرفع is primarily used for the doer (the one who carries out the action).",
            "النصب is primarily used for the the detail in a sentence. Once the action and the doer are found, everything else in the sentence is a detail.",
            "الجر is primarily used after 'of'."
        ]
    },
        "lesson2": {
        "title": "How to tell status",
        "content": """
            ### How to Tell Status
            In English, we were able to determine the status based on the meaning. In Arabic, however, status is 
            determined by a marker or sign at the end of the word.  

            As you know, there are three statuses in Arabic. There are, however, more than three status markers or 
            signs. In other words, there are more than three ways that the status of a word can show. This is 
            because each status can show in different ways depending on the number and the gender of the word.  

            It is important to keep in mind that whenever you are trying to figure out the status of an Ism you must 
            look at the ending of the word. There are two types of endings we will see, **ending sounds** (vowel 
            change at the end) and **ending combinations** (letters added to the end of a word). 

            The number/gender variations are singular, pair, masculine plural, and feminine plural.  Take a look at 
            the charts below. Notice how each status looks different depending on the number and the gender of 
            the word.   

            The word مسلم is the base. Anything beyond the last letter (in this case, the م) is part of the status marker.

            | Plural | Pair   | Singular |     |
            |--------|--------|----------|-----|
            | مسلمون | مسلمان | مسلم     | رفع |
            | مسلمين | مسلمين | مسلم     | نصب |
            | مسلمين | مسلمين | مسلم     | جر  |

            | Plural Feminine | Pair Feminine | Singular Feminine |     |
            |-----------------|---------------|-------------------|-----|
            | مسلمات          | مسلمتان       | مسلمة             | رفع |
            | مسلمات          | مسلمتين       | مسلمة             | نصب |
            | مسلمات          | مسلتين        | مسلمة             | جر  |
        """,
        "key_points": [
            "In Arabic, status is determined by a marker or sign at the end of words",
            "While there are 3 cases, there are more than 3 case markers or signs.",
            "When finding the status of a word, there are two things to pay attention to, ending sounds and ending combinations.",
            "The number/gender variations consist of singular, pair, masculine plural, and feminine plural."           
            ]
    },
        "lesson3": {
        "title": "Light vs. Heavy",
        "content": """
            ### Understanding Light and Heavy Words
            Lightness and heaviness are not from among the four properties of the اسم. Rather, the discussion of
            light and heavy is a sub-topic that falls under status. Now that we have learned about the different
            markers that we can use to determine status, we will learn about different variations and forms that
            these markers can take.

            Notice that every word in the مسلم chart ends in an ‘n’ sound, whether it be an ending sound or
            combination. These words are considered heavy. **Heavy** is the **default**. To make a word light, all you
            have to do is remove the ‘n’ sound at the end.

            | Plural 	| Pair   	| Singular 	|     	|
            |--------	|--------	|----------	|-----	|
            |  مسلمو 	|  مسلما 	|  مسلم    	| رفع 	|
            |  مسلمي 	|  مسلمي 	|  مسلم    	| نصب 	|
            |  مسلمي 	|  مسلمي 	|  مسلم    	| جر  	|

            To get rid of the ن sound in Arabic, use the following rules.

            1. If the word ends in a double accent (ْالتَّنْوِين), replace the double accent with a single حَرَكَة. For
            instance, the word مسلمٌ would become مسلمُ. The word مسلمات would become مسلماتِ .
            2. If the word ends in the letter ن, all you have to do is drop the ن. For instance, the word مسلمون
            becomes مسلمو.

            | Plural 	| Pair   	| Singular 	|     	|
            |--------	|--------	|----------	|-----	|
            | مسلمات 	| مسلمتا 	| مسلمة    	| رفع 	|
            | مسلمات 	| مسلمتي 	| مسلمة    	| نصب 	|
            | مسلمات 	| مسلمتي 	| مسلمة    	| جر  	|
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
    },
        "lesson6": {
        "title": "Introduction to الفعل المبني للمجهول",
        "content": """
            ### Introduction to الفعل المبني للمجهول
            The passive فعل, or الفعل لمبنيُ للمجهول, is a فعل-form that is used to express the occurrence of an action
            while keeping the doer of that action anonymous. In Arabic, مجهول literally means “unknown” or
            “anonymous”. The sentence, “The cake was eaten,” for example, is considered مجهول, because the one 
            who ate the cake is not known. The same goes for the sentence “The cake is eaten”. Both ماض and 
            مضارع can be made مجهول.

            To determine whether something is passive in either Arabic or English:
            1. Find the action
            2. Ask yourself: "Do I know who performed the action?"

            If the answer is no, it is passive. Otherwise, it is active.
""",
        "key_points": [
                "A مجهول word is considered anonymous or unknown in a sentance",
                "To determine if a word is passive, you must first find the action, then ask yourself: \"Do I know who performed the action?\""
        ]
    },
        "lesson7": {
        "title": "Introduction to الأفعال الناقصة",
        "content": """
            ### Introduction to الأفعال الناقصة
            الأفعالُ الناقصة are a set of أفعال that are incomplete in meaning. الأفعالُ الناقصة are also known as كانُ و أخواتها,
            or “كان and her sisters”. This is because كان is the most commonly used فعلُ ناقص. Take a look at the list
            below. Pay attention to the definitions.

            1. كان, يكون&nbsp;&nbsp;To be...
            2. أصبح, يصبح&nbsp;&nbsp;To become...
            3. ظل, يظل&nbsp;&nbsp;To remain...
            4. ما زال/لا يزال&nbsp;&nbsp;To still be...
            5. ما دام&nbsp;&nbsp;As long as...
            6. ليس&nbsp;&nbsp;Is not...

            Notice that the أفعال above do not convey a complete thought. For example, were you to hear someone
            say “كانَُ” or “He was...” you would be left with the questions “What/who was he?” Compare this to a
            normal فعل, like “أَكَلَُ” or “He ate”. This is a complete sentence as it conveys a complete thought.

            Because these أفعال are incomplete in meaning, they do not operate like a normal فعل. In fact, a sentence
            that contains a فعلُ ناقص is not even considered a جملةُ فعلية. It is considered a جملةُ اسمية.

            Just as we defined the part before “is” as a مبتدأ and the part after “is” as the خبر orمتعلقُ بالخبر in a regular
            جملةُاسمية, in this new type of جملةُ اسمية that we are learning about, the part before “was” (or any of the
            other sisters of كان) is the مبتدأ and the part after it is the خبر orمتعلقُ بالخبر.

            A key difference, however, is that while the “is” in a regular جملةُ اسمية is invisible, the “was” (or any of the
            other أفعالُ ناقصة) is not. It is considered part of the مبتدأ.
""",
        "key_points": [
                "الأفعال الناقصة are incomplete in meaning",
                "الأفعال الناقصة do not leave you with a complete thought, unlike normal الأفعال.",
                "Sentences with الأفعال الناقصة are considered جمل اسمية",
                "The part before 'was' (or any of the other sisters of كان) is مبتد, and the part after it is خبر/متعق بالخبر",
                "While the 'is' is invisible in regular جمل اسمية, the 'was' (or any of the other أفعال الناقصة) is not."
        ]
    }
}

st.set_page_config(page_title="Arabic Lesson Recall", layout="wide")
st.title("Arabic Lesson Recall")

# Initialize recall flag
if "show_recall" not in st.session_state:
    st.session_state.show_recall = False

if "hide_lesson" not in st.session_state:
    st.session_state.hide_lesson = False

# Select
model_choice = "Gemini"

if not st.session_state.hide_lesson:
    lesson_id = st.selectbox("Choose a lesson:", options=list(LESSONS.keys()), format_func=lambda k: LESSONS[k]["title"])
    st.session_state.lesson = lesson = LESSONS[lesson_id]
    st.markdown(lesson["content"], unsafe_allow_html=True)


if not st.session_state.show_recall:
    if st.button("Start Recall"):
        st.session_state.show_recall = True
        st.session_state.hide_lesson = True
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
                result = evaluate_response_with_rag(user_input, st.session_state.lesson, model_choice)
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
    
    if st.button("Show lesson"):
        st.session_state.hide_lesson = False
        st.session_state.show_recall = False
        st.rerun()
