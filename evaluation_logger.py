import pandas as pd
import os
import json
from pymongo.mongo_client import MongoClient
from pymongo.server_api import ServerApi

uri = os.getenv("uri")

client = MongoClient(uri, server_api=ServerApi("1"))

database = client["brilliant-muslim-ai-recall"]
collection = database["logged-data"]

try:
    client.admin.command('ping')
    print("Pinged!")
except Exception as e:
    print(e)


def append_example_mongo(prompt: str, response: dict, user_response, lesson):
    collection.insert_one({"prompt" : prompt,
                           "response" : json.dumps(response, ensure_ascii=False),
                            "user_response" : user_response,
                            "lesson": lesson
                            })
    


def append_example(prompt: str, response: dict, path: str = "eval_dataset.csv"):

    row = {
        "prompt": prompt,
        "response": json.dumps(response, ensure_ascii=False),
    }
    pd.DataFrame([row]).to_csv(
        path, mode="a", header=not os.path.exists(path), index=False
    )

def append_prompt_data(retrieved_text: str, key_points_text: str, user_response: str, path: str = "prompt_dataset.csv"):
    row = {
        "user_response": user_response,
        "key_points_text": key_points_text,
        "retrieved text": retrieved_text
    }
    pd.DataFrame([row]).to_csv(
        path, mode="a", header=not os.path.exists(path), index=False
    )
