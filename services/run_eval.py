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
from eval.evaluation_logger import append_example, append_result

prompt_dataset_path = "data/prompt_dataset.csv"
eval_dataset_path = "data/eval_datasetnew.csv"
results_path = "data/eval_resultsnew.csv"

def run_eval(prompt, response_input):
    openai_llm = OpenAIModel(
    'gpt-4o',
    provider=OpenAIProvider(api_key=os.getenv("OPENAI_API_KEY"))
    )
    instructions = f"""You are an LLM judge model that is tasked with evaluating the response of an AI model to a prompt. Here is the prompt supplied to the model: {prompt}. Here is the model's output: {response_input}. 
    You will be judging the model's output based on two rubrics, the first of which is text quality. Here is the rubric:
    Comprehensibility: The AI model does not talk in difficult grammar jargon and hard to understand text, but rather talks to the user at an understandable and basic level. Arabic grammar terms are primarily used as opposed to English ones. Sentences would be comprehensible by a user who doesn't use English as their primary language. The text isn't overcomplicated or confusing, but rather is simple and clear to the reader.
    Gentleness: The text does not come across as scolding the user or being overly harsh with them, rather it is gentle and encouraging. The text is encouraging and excites the learner to study further rather than discouraging them or making them feel unworthy. The text offers realistic feedback and doesn't sugarcoat mistakes, while simultaneously being gentle in its approach. The user will come away from reading the text feeling motivated and encouraged.
    Accuracy: The text is accurate in its feedback. It does not illogically say the user made a mistake where they didn't, and doesn't illogically expect the user to know something unrealistic. The text is accurate to the rules of Arabic grammar, and its critiques of the user are accurate based on the user input. The text should not include critiques just for the sake of having critiques. If there are no critiques the text should reflect that, and if there are legitimate crtiques, the text should reflect that.
    Fluency: Sentences flow smoothly and are easy to read, avoiding awkward phrasing or run-on sentences. Ideas and sentences connect logically, using transitions effectively where needed.
    Constructiveness: The feedback given is useful and accurate. The feedback directly references mistakes the user made (or things done well). If mistakes were made, the model corrects them and outputs feedback on how to avoid the mistake going forward.
    
    Please give a rating from 1-5 based on the above rubric.
    
    Simply output the number rating and nothing more please.
    """
    openaiAgent = Agent(openai_llm)
    eval_output = openaiAgent.run_sync(user_prompt=instructions)
    return eval_output.output

def generate_dataset():
    with open(prompt_dataset_path, "r", encoding="utf-8") as file:
        dict_reader = csv.DictReader(file)
        csv_rows = list(dict_reader)
        count = 0
        for row_dict in csv_rows:
            prompt = build_rag_prompt(row_dict["user_response"], row_dict["retrieved text"], row_dict["key_points_text"].replace("[", "").replace("]", "").split(","))
            response = evaluate(prompt, "Gemini")
            append_example(prompt, response, eval_dataset_path)
            count += 1
            if count >= 3:
                break

def generate_results():
    with open(eval_dataset_path, "r", encoding="utf-8") as file:
        dict_reader = csv.DictReader(file)
        csv_rows = list(dict_reader)
        for row_dict in csv_rows:
            output = run_eval(row_dict["prompt"], row_dict["response"]);
            append_result(row_dict["prompt"], row_dict["response"], output, results_path)

generate_dataset()
generate_results()
        

    

        
        
