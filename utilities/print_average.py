import csv
import os 
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

def get_average(file_name = "data/eval_results.csv"):
    scores = []
    sum = 0

    with open(file_name, 'r') as file: 
        dict_reader = csv.DictReader(file)
        for row_dict in dict_reader:
            scores.append(float(row_dict["custom_text-quality/score"]))

    for temp in scores:
        sum += temp
        
    return sum/len(scores)

if __name__ == "__main__":
    print(get_average())

