import json
import sys
import os

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)


metrics_path = "eval/metrics.json"

with open(metrics_path, 'r') as f:
    metric_data = json.load(f)

for temp in metric_data:
    print(metric_data[temp]["criteria"])

