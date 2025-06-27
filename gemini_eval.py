import vertexai
import pandas as pd
from vertexai.evaluation import EvalTask, PointwiseMetric, PointwiseMetricPromptTemplate
from google.cloud import aiplatform

PROJECT_ID = "model-eval-463217"

vertexai.init(
    project=PROJECT_ID
)

custom_text_quality = PointwiseMetric(
    metric="custom_text-quality",
    metric_prompt_template=PointwiseMetricPromptTemplate(
        criteria={
            "comprehensibility": (
                
                    "The AI model does not talk in difficult grammar jargon and hard to understand text, but rather talks to the user "
                    "at an understandable and basic level. Arabic grammar terms are primarily used as opposed to English ones. Sentences would be comprehensible by a user "
                    "who doesn't use English as their primary language. The text isn't overcomplicated or confusing, but rather is simple and clear to the reader." 
                
            ),
            "gentleness": (
                 
                    "The text does not come across as scolding the user or being overly harsh with them, rather it is gentle and encouraging. The text is encouraging and excites the learner to "
                    "study further rather than discouraging them or making them feel unworthy. The text offers realistic feedback and doesn't sugarcoat mistakes, while simultaneously being gentle in its "
                    "approach. The user will come away from reading the text feeling motivated and encouraged."
                
            ),
            "accuracy": (
                 
                    "The text is accurate in its feedback. It does not illogically say the user made a mistake where they didn't, and doesn't illogically "
                    "expect the user to know something unrealistic. The text is accurate to the rules of Arabic grammar, and its critiques of the user "
                    "are accurate based on the user input. The text should not include critiques just for the sake of having critiques. If there are no "
                    "critiques the text should reflect that, and if there are legitimate crtiques, the text should reflect that."
                
            ),
        },
        rating_rubric={
            
                "5": "(Very good). Exceptionally clear, coherent, fluent, and concise. Fully adheres to instructions and stays grounded.",
                "4": "(Good). Well-written, coherent, and fluent. Mostly adheres to instructions and stays grounded. Minor room for improvement.",
                "3": "(Ok). Adequate writing with decent coherence and fluency. Partially fulfills instructions and may contain minor ungrounded information. Could be more concise.",
                "2": "(Bad). Poorly written, lacking coherence and fluency. Struggles to adhere to instructions and may include ungrounded information. Issues with conciseness.",
                "1": "(Very bad). Very poorly written, incoherent, and non-fluent. Fails to follow instructions and contains substantial ungrounded information. Severely lacking in conciseness."
            
        },
    ),
)

responses = [
    """
    {'score': 100, 'correct_points': ["You correctly identified the three cases in Arabic: Raf', Nasb, and Jarr.", "You correctly stated that Raf' is used for the subject, predicate, and doer.", 'You correctly stated that Jarr is used after prepositions.', 'You correctly stated that Nasb is used for the done-to and after certain huruf.'], 'incorrect_points': [], 'missed_points': [], 'generated_feedback': 'Excellent work! You have perfectly recalled all the key information about the three Arabic cases and their primary uses. Your answer is complete and accurate based on the provided material. Keep up the great work!', 'rewritten_answer': "Arabic has 3 cases: Raf', nasb, and jarr. Raf' is used for the subject, the predicate, and the doer. Jarr is used after prepositions. Nasb is used for the done-to and after certain huruf."}
    """,
]

eval_dataset = pd.read_csv("eval_dataset.csv")

eval_task = EvalTask(
    dataset=eval_dataset,
    metrics=[custom_text_quality],
    experiment="myexperiment"
)

pointwise_result = eval_task.evaluate()

print(pointwise_result.metrics_table)

with open("eval_results.txt", "w") as f:
    f.write(pointwise_result.metrics_table.to_string())

aiplatform.ExperimentRun(
    run_name=pointwise_result.metadata["experiment_run"],
    experiment = pointwise_result.metadata["experiment"],
).delete()