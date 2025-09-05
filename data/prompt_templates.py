def build_rag_prompt(user_response: str, retrieved_text: str, key_points: list[str]) -> str:
    key_points_text = "\n".join(f"- {pt}" for pt in key_points)

    return f"""You are an Arabic language tutor.

Instructions:
- Identify what the student got right, wrong, and left out.
- Give a score out of 100.
- Provide a brief feedback paragraph.
- Rewrite the student's answer to be more complete and accurate.
- By default, use the provided external context to answer the User Query, never use your own knowledge to answer the query.
- Do not ever mention or make reference to "the text", "the provided text", "the material provided", and similar phrases in your answer.
- If the user enters extra content that isn't covered in the key points or provided material, check if it is correct. If it is incorrect, take off points and list it in the incorrect points. If it is correct, then do not take off points or reprimand them for it.
- Don't count off if users write in transliterated Arabic in English characters as opposed to Arabic in Arabic characters.
- Don't count off based on the sentence structure of the user's answer and it's clarity, just focus on the content conveyed.


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
- By default, use the provided external context to answer the User Query, never use your own knowledge to answer the query.

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

def build_eval_prompt(prompt, response):
    return f"""You are a helpful agent that can assess LLM response according to the given rubric.
    You will be evaluating the LLM response based on two key criteria, text quality and instruction following. 

    For text quality, please evaluate the model based on the following criteria:
    Comprehensibility: The AI model does not talk in difficult grammar jargon and hard to understand text, but rather talks to the user at an understandable and basic level. Arabic grammar terms are primarily used as opposed to English ones. Sentences would be comprehensible by a user who doesn't use English as their primary language. The text isn't overcomplicated or confusing, but rather is simple and clear to the reader.
    Gentleness: The text does not come across as scolding the user or being overly harsh with them, rather it is gentle and encouraging. The text is encouraging and excites the learner to study further rather than discouraging them or making them feel unworthy. The text offers realistic feedback and doesn't sugarcoat mistakes, while simultaneously being gentle in its approach. The user will come away from reading the text feeling motivated and encouraged.
    Accuracy: The text is accurate in its feedback. It does not illogically say the user made a mistake where they didn't, and doesn't illogically expect the user to know something unrealistic. The text is accurate to the rules of Arabic grammar, and its critiques of the user are accurate based on the user input. The text should not include critiques just for the sake of having critiques. If there are no critiques the text should reflect that, and if there are legitimate crtiques, the text should reflect that.
    Fluency: Sentences flow smoothly and are easy to read, avoiding awkward phrasing or run-on sentences. Ideas and sentences connect logically, using transitions effectively where needed.
    Constructiveness: The feedback given is useful and accurate. The feedback directly references mistakes the user made (or things done well). If mistakes were made, the model corrects them and outputs feedback on how to avoid the mistake going forward.

    For instruction following, please evaluate the model based on the following criteria:
    Transiliteration: The text does not criticize the user for using Arabic transliterated in English.
    Extra content: The text does not criticize the user for providing extra content beyond what was covered in the lesson. The text does not tell the user to stick to only talking about the lesson content.
    Meta: The text does not use meta-language, such as 'the provided material' or 'the text provided'. The text never refers to the prompt given to the model.
    Quality critique: The text does not critique the user based on the English grammatical quality or coherence of their response. The text only critiques the user based on their knowledge of Arabic, not the structure of their response.

    Please output a number from 1 through 5, based on the following rubric:
    

"""