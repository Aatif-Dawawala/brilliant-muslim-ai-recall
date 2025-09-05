import csv
import os
import sys 

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from data.prompt_templates import build_rag_prompt
from services.model_switcher import evaluate
from eval.evaluation_logger import append_example

prompt_dataset_path = "data/prompt_dataset.csv"
eval_dataset_path = "data/eval_dataset_new.csv"
with open(prompt_dataset_path, "r", encoding="utf-8") as file:
    dict_reader = csv.DictReader(file)
    csv_rows = list(dict_reader)
    for row_dict in csv_rows:
        prompt = build_rag_prompt(row_dict["user_response"], row_dict["retrieved text"], row_dict["key_points_text"].replace("[", "").replace("]", "").split(","))
        response = evaluate(prompt, "Gemini")
        append_example(prompt, response, eval_dataset_path)

with open(eval_dataset_path, "r", encoding="utf-8") as file:
    dict_reader = csv.DictReader(file)
    csv_rows = list(dict_reader)
    for row_dict in csv_rows:
        

        
        

