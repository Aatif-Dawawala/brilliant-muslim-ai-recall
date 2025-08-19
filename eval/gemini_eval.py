import argparse
import os
import sys

import vertexai
import pandas as pd
from vertexai.evaluation import EvalTask, PointwiseMetric, PointwiseMetricPromptTemplate
from google.cloud import aiplatform
import csv
import json
from pathlib import Path

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from utilities.print_average import get_average
from services.generate_metrics import generate_metrics


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

    eval_dataset = pd.read_csv(dataset_path, encoding="utf-8")

    eval_task = EvalTask(
        dataset=eval_dataset,
        metrics=generate_metrics(),
        experiment="myexperiment",
    )

    pointwise_result = eval_task.evaluate()
    print(pointwise_result.metrics_table)


    Path(os.path.dirname(results_path)).mkdir(parents=True, exist_ok=True)
    with open(results_path, "w", encoding="utf-8", newline="") as f:
        pointwise_result.metrics_table.to_csv(f, index=False)

    aiplatform.ExperimentRun(
        run_name=pointwise_result.metadata["experiment_run"],
        experiment=pointwise_result.metadata["experiment"],
    ).delete()

    print(f"The average score is: {get_average()}")


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