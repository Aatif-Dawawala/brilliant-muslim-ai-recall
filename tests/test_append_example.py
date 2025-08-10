import pytest 
import sys 
import os
import test_data
import csv
import json

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from evaluation_logger import append_example

def test_append_example():
    file_path = "temp_eval_dataset.csv"

    append_example(test_data.prompt, test_data.response, file_path)
    logged_prompt = ""
    logged_response = ""
    with open('temp_eval_dataset.csv', newline="") as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            logged_prompt = row[0]
            logged_response = row[1]

    os.remove(file_path)
    assert logged_prompt == test_data.prompt
    assert logged_response == json.dumps(test_data.response, ensure_ascii=False)



test_append_example()

        

