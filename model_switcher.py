import os
import json
from typing import Dict

from pydantic import BaseModel
from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIModel
from pydantic_ai.providers.openai import OpenAIProvider
from pydantic_ai.models.gemini import GeminiModel
from pydantic_ai.providers.google_gla import GoogleGLAProvider

class OutputFormat(BaseModel):
    score: int
    correct_points: list[str]
    incorrect_points: list[str]
    missed_points: list[str]
    generated_feedback: str
    rewritten_answer: str


openai_llm = OpenAIModel(
    'gpt-4.1',
    provider=OpenAIProvider(api_key=os.getenv("OPENAI_API_KEY"))
)

openaiAgent = Agent(openai_llm, instructions="You are an expert Arabic language tutor.")

gemini_llm = GeminiModel(
    "gemini-2.5-pro",
    provider=GoogleGLAProvider(api_key=os.getenv("GEMINI_API_KEY"))
)

geminiAgent = Agent(gemini_llm, instructions="You are an expert Arabic language tutor.")


def evaluate(prompt: str, model_choice: str) -> dict:
    if model_choice == "Gemini":
        response = geminiAgent.run_sync(user_prompt=prompt)
    elif model_choice == "OpenAI":
        response = openaiAgent.run_sync(user_prompt=prompt)
    
    print(((response.output).strip("```")).strip("json"))
    return json.loads(((response.output).strip("```")).strip("json"))
