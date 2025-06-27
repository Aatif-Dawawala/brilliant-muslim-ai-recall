import pandas as pd
import os
import json


def append_example(prompt: str, response: dict, path: str = "eval_dataset.csv"):

    row = {
        "prompt": prompt,
        "response": json.dumps(response, ensure_ascii=False),
    }
    pd.DataFrame([row]).to_csv(
        path, mode="a", header=not os.path.exists(path), index=False
    )
