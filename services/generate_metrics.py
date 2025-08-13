import os
import sys
import json

from vertexai.evaluation import PointwiseMetric, PointwiseMetricPromptTemplate


current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

def generate_metrics():

    metrics_path = "eval/metrics.json"

    metrics = []

    with open(metrics_path, 'r') as f:
        metric_data = json.load(f)

    for i in range(len(metric_data)):
        
        metrics.append(PointwiseMetric(
                metric=metric_data[i][0],
                metric_prompt_template=PointwiseMetricPromptTemplate(
                    criteria=metric_data[i][1],
                    rating_rubric=metric_data[i][2]
                )
            )
        )

    return metrics