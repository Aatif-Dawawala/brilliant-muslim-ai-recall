prompt = """
You are an Arabic language tutor.

Instructions:
- Identify what the student got right, wrong, and left out.
- Give a score out of 100.
- Provide a brief feedback paragraph.
- Rewrite the student's answer to be more complete and accurate.
- By default, use the provided external context to answer the User Query, never use your own knowledge to answer the query.


Here is the relevant content retrieved from the textbook
<<<
flexible words as you are exposed to more vocabulary.  However, there are a few categories of partly-
flexible words we can familiarize ourselves with to make identification of partly-flexible words easier.  
A. NON-ARAB NAMES
Non-Arab names are partly-flexible. Arab names are fully-flexible. There are four Arab prophet’s names 
mentioned in the Quran. They are:  ٌ ,صالِحٌ  ,هُوْدٌ  ,ُمُ َمَّدand  ٌ .شُ عَيْبTake a look at the chart below. Take note of
---
نصب
   مُسْ لِمات َ ْمُسْ لِمِي ِ ْ مُسْ لِمَي  مِلْسُم
جر
There are three forms of flexibility. They are: 
1)
fully-flexible ( )مُنَْصَ ِف/مُعْرَب
2)
partly-flexible ( ِ)مَمْنُوْعٌ مِن الَصَّ ْف
3)
non-flexible (  ِ )مَبْن
We will discuss partly-flexible last, because it requires the most attention. 
FULLY-FLEXIBLE (بَرْعُم/فِ ََصْنُ )مis the default state for a word. It is the broadest category. A fully-flexible
---
جر
نصب
رفع
  عَدْن
عَدْنًا ٌ عَدْن
There are certain word patterns that are partly-flexible. We will look at these patterns in our study of 
Sarf.  
Note: In your vocabulary, partly flexible words will be denoted by having a single vowel ending, not a 
 .تنوينFor now, as you memorize new vocabulary, make a note of which words are partly-flexible.
➢ DRILL 5    Are the following words fully, partly, or non-flexible? Why?  
REASON 
FLEXIBILITY WORD
REASON 
FLEXIBILITY 
WORD
F / P / N
---
the differences between how the partly-flexible names and the fully-flexible names look in each status. 
جر
نصب
رفع
 َيُوْسُف َيُوْسُف ُيُوْسُف
 َإِسْماعِيْل َإِسْماعِيْل ُإِسْماعِيْل
 َإِبْراهِيْم َإِبْراهِيْم ُإِبْراهِيْم
 َيَعْقُوْب َيَعْقُوْب ُيَعْقُوْب
 َمَرْيَم َمَرْيَم ُمَرْيَم
  ُمُ َمَّد
ُمُ َمَّدًا ٌ ُمُ َمَّد
  هُوْد
هُوْدًا ٌ هُوْد
  صالِح
صاِلِ ًا ٌ صالِح
  شُعَيْب
شُعَيْبًا ٌ شُعَيْب
>>>

And here are the key points the student should recall:
- Flexibility is a sub-category of status
- Flexibility only pertains to singular words
- Words may only be fully-flexible, partly-flexible, or non-flexible.

The student wrote:
""There are three types of flexibility in Arabic: fully-flexible, partly-flexible, and non-flexible.
Certain words have different levels of flexibility, such as non-Arab names.""

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
            {
                "score": 70,
                "correct_points": ["الرفع is primarily used for the subject", "النصب is primarily used for the done-to"],
                "incorrect_points": ["You incorrectly stated that الجر is used for prepositions. The correct usage of الجر is after prepositions"],
                "missed_points": ["You didn't mention that النصب is used after حروف which trigger its use", "You didn't mention that الرفع is used for the predicate and doer"],
                "generate_feedback": "You have a solid idea of how cases work in Arabic. You primarily need to focus on the specific details of when the cases are used. I would suggest that you through the lesson once more and focus on the finer details, then try recalling everything you remember a second time. Good luck!",
                "rewritten_answer": "Arabic has three cases. One of the cases is رفع, another is جر, and finally we have نصب.  Raf' is used for the subject, predicate, and done-to; jarr is used after prepositions; nasb is used for the done-to and after حروف which trigger its use. 
                                     These cases help us understand the role of certain words in Arabic"
            }
]

Response in this JSON format :
{
    "score": <number>,
    "correct_points": [...],
    "incorrect_points": [...],
    "missed_points": [...],
    "generated_feedback": "...",
    "rewritten_answer": "..."
}

"""

response = """
"{\n  \"score\": 60,\n  \"correct_points\": [\n    \"Identified the three types of flexibility: fully-flexible, partly-flexible, and non-flexible.\",\n    \"Correctly stated that non-Arab names are an example of a category of words with a specific flexibility type.\"\n  ],\n  \"incorrect_points\": [],\n  \"missed_points\": [\n    \"You did not mention that flexibility is a sub-category of status.\",\n    \"You did not mention that flexibility only pertains to singular words.\"\n  ],\n  \"generated_feedback\": \"You have correctly identified the three categories of flexibility and provided a good example. That's a great start! However, your answer is missing some crucial context. To make your understanding more complete, remember to include that flexibility is a sub-category of status and that this concept applies specifically to singular words. Reviewing the definition of what flexibility entails would be beneficial.\",\n  \"rewritten_answer\": \"Flexibility is a sub-category of status and it only applies to singular words. Words in Arabic can be one of three types: fully-flexible, partly-flexible, or non-flexible. Fully-flexible is the default for words. Partly-flexible words, such as non-Arab names, have some limitations on how they show status. Non-flexible words do not change their endings at all.\"\n}"
"""