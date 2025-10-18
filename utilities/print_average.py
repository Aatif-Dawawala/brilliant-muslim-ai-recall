import csv
import os 
import sys
import json

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

def get_average(file_name = "data/eval_results.csv"):
    metrics_path = "data/metrics.json"
    metrics = []
    averages = {}
    scores = []
    sum = 0

    with open(metrics_path, 'r') as f:
        metric_data = json.load(f)

    for i in range(len(metric_data)):
        metrics.append(metric_data[i][0] + "/score")

    with open(file_name, 'r', encoding="utf-8") as file: 
        dict_reader = csv.DictReader(file)
        csv_rows = list(dict_reader)
        for item in metrics:
            for row_dict in csv_rows:
                if row_dict[item] != "":
                    scores.append(float(row_dict[item]))
            for temp in scores:
                sum += temp

                averages[item] = sum/len(scores)

            scores = []
            sum = 0
        
    return averages

def get_new_average(file_name):
    sum =0;    

if __name__ == "__main__":
    print(get_average())
