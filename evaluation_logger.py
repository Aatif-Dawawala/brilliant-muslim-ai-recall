import pandas as pd
import os
import json
from pymongo.mongo_client import MongoClient
from pymongo.server_api import ServerApi

uri = os.getenv("uri")

client = MongoClient(uri, server_api=ServerApi("1"))

try:
    client.admin.command('ping')
    print("Pinged!")
except Exception as e:
    print(e)

def append_example(prompt: str, response: dict, path: str = "eval_dataset.csv"):

    row = {
        "prompt": prompt,
        "response": json.dumps(response, ensure_ascii=False),
    }
    pd.DataFrame([row]).to_csv(
        path, mode="a", header=not os.path.exists(path), index=False
    )
