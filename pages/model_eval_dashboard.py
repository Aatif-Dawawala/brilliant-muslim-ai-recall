import os
import pandas as pd
import streamlit as st
from utilities.print_average import get_average

st.title("Model Evaluation Dashboard")

results_path = "data/eval_results.csv"

if os.path.exists(results_path):
    df = pd.read_csv(results_path)
    st.subheader("Raw Evaluation Results")
    st.dataframe(df)

    st.subheader("Average Scores")
    averages = get_average(results_path)
    st.json(averages)
else:
    st.warning(f"Evaluation results file not found at {results_path}.")
    st.info("Run evaluations to generate the results file.")
