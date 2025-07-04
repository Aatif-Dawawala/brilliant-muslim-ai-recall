import os
import json
from typing import Dict

from pydantic import BaseModel
from pydantic_ai import PydanticAI
from pydantic_ai.openai import OpenAIChat
from pydantic_ai.genai import GeminiChat


class OutputFormat(BaseModel):
    score: int
    correct_points: list[str]
    incorrect_points: list[str]
    missed_points: list[str]
    generated_feedback: str
    rewritten_answer: str


openai_llm = OpenAIChat(
    model="gpt-4.1",
    api_key=os.getenv("OPENAI_API_KEY"),
)

gemini_llm = GeminiChat(
    model="gemini-2.5-pro-preview-06-05",
    api_key=os.getenv("GEMINI_API_KEY"),
)

MODEL_REGISTRY: Dict[str, PydanticAI] = {
    "OpenAI": PydanticAI(llm=openai_llm, output_model=OutputFormat),
    "Gemini": PydanticAI(llm=gemini_llm, output_model=OutputFormat),
}


def evaluate(prompt: str, model_choice: str) -> dict:
    ai = MODEL_REGISTRY.get(model_choice)
    if ai is None:
        raise ValueError("Unsupported model selected")
    response = ai(prompt)
    if isinstance(response, BaseModel):
        return json.loads(response.model_dump_json())
    return json.loads(response)
