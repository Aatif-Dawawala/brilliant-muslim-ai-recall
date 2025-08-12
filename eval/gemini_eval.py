import argparse
import os
import sys

import vertexai
import pandas as pd
from vertexai.evaluation import EvalTask, PointwiseMetric, PointwiseMetricPromptTemplate
from google.cloud import aiplatform
import csv
import json

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from utilities.print_average import get_average

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
            "fluency": (

                    "Sentences flow smoothly and are easy to read, avoiding awkward"
                    " phrasing or run-on sentences. Ideas and sentences connect"
                    " logically, using transitions effectively where needed."    
                            
            ),
            "constructiveness": (

                    "The feedback given is useful and accurate. The feedback directly"
                    " references mistakes the user made (or things done well). If mistakes"
                    " were made, the model corrects them and outputs feedback on how to "
                    "avoid the mistake going forward."

            )
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

metrics_path = "eval/metrics.json"

with open(metrics_path, 'r') as f:
    metric_data = json.load(f)


def main(dataset_path: str, project_id: str, results_path: str) -> None:
    """Run evaluation on a dataset and store the results.

    Args:
        dataset_path: Path to the CSV dataset file.
        project_id: Google Cloud project identifier used for Vertex AI.
        results_path: Destination path for the evaluation results CSV.
    """

    if not os.path.isfile(dataset_path):
        raise FileNotFoundError(f"Dataset not found: {dataset_path}")

    try:
        vertexai.init(project=project_id)
    except Exception as exc:  # pragma: no cover - depends on external service
        raise RuntimeError(
            f"Could not initialize Vertex AI for project '{project_id}': {exc}"
        ) from exc

    eval_dataset = pd.read_csv(dataset_path)

    eval_task = EvalTask(
        dataset=eval_dataset,
        metrics=[custom_text_quality],
        experiment="myexperiment",
    )

    pointwise_result = eval_task.evaluate()
    print(pointwise_result.metrics_table)

    try:
        print(os.listdir())
        print(os.getcwd())
        with open(results_path, "w") as f:
            f.write(pointwise_result.metrics_table.to_csv(index=False))
    except OSError as exc:
        raise RuntimeError(
            f"Could not write results to {results_path}: {exc}"
        ) from exc

    aiplatform.ExperimentRun(
        run_name=pointwise_result.metadata["experiment_run"],
        experiment=pointwise_result.metadata["experiment"],
    ).delete()

    print(f"The average score is: {round(get_average(), 2)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate model responses using Vertex AI",
    )
    parser.add_argument(
        "--dataset",
        dest="dataset_path",
        default=os.getenv("DATASET_PATH", "data/eval_dataset.csv"),
        help="Path to the evaluation dataset CSV",
    )
    parser.add_argument(
        "--project",
        dest="project_id",
        default=os.getenv("PROJECT_ID"),
        help="Google Cloud project ID",
    )
    parser.add_argument(
        "--results",
        dest="results_path",
        default=os.getenv("RESULTS_PATH", "data/eval_results.csv"),
        help="File path to write evaluation results",
    )

    args = parser.parse_args()

    if not args.project_id:
        parser.error("A project ID must be provided via --project or PROJECT_ID")

    try:
        main(args.dataset_path, args.project_id, args.results_path)
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)

