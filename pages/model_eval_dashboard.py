"""Streamlit dashboard for viewing and running model evaluations.

This module exposes a ``run_eval`` function that wraps the evaluation entry
point so it can be triggered from the dashboard as well as imported in unit
tests.
"""

import os

import pandas as pd
import streamlit as st

from utilities.print_average import get_average
from eval.gemini_eval import main as gemini_main
from data.prompt_templates import build_rag_prompt


def run_eval(dataset_path: str, project_id: str, results_path: str) -> None:
    """Run the evaluation script and write results to ``results_path``.

    Parameters
    ----------
    dataset_path:
        Path to the dataset containing prompts and responses.
    project_id:
        Google Cloud project identifier used for Vertex AI.
    results_path:
        Destination file to which evaluation results are written.
    """

    gemini_main(dataset_path, project_id, results_path)


def render_dashboard() -> None:
    """Display the evaluation dashboard in Streamlit."""

    st.title("Model Evaluation Dashboard")
    st.header("Gemini 2.5 Pro")

    results_path = st.text_input(
        "Results file", value="data/eval_results.csv", key="results_path"
    )

    dataset_path = st.text_input(
        "Dataset file", value="data/eval_dataset.csv", key="dataset_path"
    )
    project_id = st.text_input(
        "Project ID", value=os.getenv("PROJECT_ID", ""), key="project_id"
    )

    if st.button("Run Evaluation"):
        if not project_id:
            st.error("A project ID must be provided to run the evaluation.")
        else:
            with st.spinner("Running evaluation..."):
                try:
                    run_eval(dataset_path, project_id, results_path)
                    st.success("Evaluation completed successfully.")
                except Exception as exc:  # pragma: no cover - UI feedback only
                    st.error(f"Evaluation failed: {exc}")

    if os.path.exists(results_path):
        df = pd.read_csv(results_path)

        st.subheader("Prompt")

        st.text_area("", value=build_rag_prompt("{user_response}", "{retrieved_text}", ["{key_points}"]), height=340)

        st.subheader("Raw Evaluation Results")
        st.dataframe(df)

        st.subheader("Average Scores")
        averages = get_average(results_path)
        st.json(averages)
    else:
        st.warning(f"Evaluation results file not found at {results_path}.")
        st.info("Run evaluations to generate the results file.")


if __name__ == "__main__":  # pragma: no cover - UI entry point
    render_dashboard()
