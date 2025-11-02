import csv
import os 
import sys
import json

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)


def get_average(file_name):
    scores = []
    sum = 0
    average = 0

    with open(file_name, 'r', encoding="utf-8") as file:
        dict_reader = csv.DictReader(file)
        csv_rows = list(dict_reader)
        for row_dict in csv_rows:
            if row_dict["grade"] != "":
                scores.append(float(row_dict["grade"]))
        
        for temp in scores:
            sum += temp
        
        average = sum / len(scores)

        return average

    
if __name__ == "__main__":
    print(get_average("data/new_eval_results.csv"));
