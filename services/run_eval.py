import csv
import os
import sys 
from pydantic import BaseModel
from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIModel
from pydantic_ai.providers.openai import OpenAIProvider

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from data.prompt_templates import build_rag_prompt
from services.model_switcher import evaluate
from eval.evaluation_logger import append_example

prompt_dataset_path = "data/prompt_dataset.csv"
eval_dataset_path = "data/eval_dataset.csv"

def run_eval(prompt, response_input):
    openai_llm = OpenAIModel(
    'gpt-4o',
    provider=OpenAIProvider(api_key=os.getenv("OPENAI_API_KEY"))
    )
    instructions = f"You are an LLM judge model that is tasked with evaluating the response of an AI model to a prompt. Please give the response a rating from 1 to 5. Here is the prompt supplied to the model: {prompt}. Here is the model's output: {response_input} Simply output the number rating and nothing more please."
    test_prompt = "What is the capital of the U.S"
    openaiAgent = Agent(openai_llm)
    eval_output = openaiAgent.run_sync(user_prompt=instructions)
    return eval_output.output

# with open(prompt_dataset_path, "r", encoding="utf-8") as file:
#     dict_reader = csv.DictReader(file)
#     csv_rows = list(dict_reader)
#     for row_dict in csv_rows:
#         prompt = build_rag_prompt(row_dict["user_response"], row_dict["retrieved text"], row_dict["key_points_text"].replace("[", "").replace("]", "").split(","))
#         response = evaluate(prompt, "Gemini")
#         append_example(prompt, response, eval_dataset_path)

with open(eval_dataset_path, "r", encoding="utf-8") as file:
    dict_reader = csv.DictReader(file)
    csv_rows = list(dict_reader)
    count = 0;
    for row_dict in csv_rows:
        output = run_eval(row_dict["prompt"], row_dict["response"]);
        print(output)
        if (count == 3):
            break
        count += 1

    

        
        
