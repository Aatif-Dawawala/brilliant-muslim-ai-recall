import pandas as pd
import os

def append_example(prompt: str, response: dict, path: str = "eval_dataset.csv"):
    pd.DataFrame([]).to_csv(path, mode="a", header=not os.path.exists(path), index=False)