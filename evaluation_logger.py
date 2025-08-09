import pandas as pd
import os
import json
from pymongo.mongo_client import MongoClient
from pymongo.server_api import ServerApi


def get_collection():
    """Return the MongoDB collection for logging.

    Raises:
        ValueError: If the MongoDB URI is not set.
        ConnectionError: If connection to MongoDB fails.
    """
    uri = os.getenv("uri")
    if not uri:
        raise ValueError("MongoDB URI is not set in environment variable 'uri'")
    try:
        client = MongoClient(uri, server_api=ServerApi("1"))
        client.admin.command('ping')
    except Exception as e:
        raise ConnectionError(f"Failed to connect to MongoDB: {e}")
    database = client["brilliant-muslim-ai-recall"]
    return database["logged-data"]


def append_example_mongo(prompt: str, response: dict, user_response, lesson):
    try:
        collection = get_collection()
    except Exception as e:
        print(f"Could not log to MongoDB: {e}")
        return
    collection.insert_one({"prompt": prompt,
                           "response": json.dumps(response, ensure_ascii=False),
                           "user_response": user_response,
                           "lesson": lesson})
    


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
