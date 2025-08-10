import pytest 
import sys 
import os
import test_data
import csv

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from evaluation_logger import append_example

def test_append_example():
    append_example(test_data.prompt, test_data.response, "temp_eval_dataset.csv")
    logged_prompt = ""
    logged_response = ""
    with open('temp_eval_dataset.csv', newline="") as csvfile:
        reader = csv.reader(csvfile)
        logged_prompt = reader[0][0]
        logged_response = reader[0][1]
        

