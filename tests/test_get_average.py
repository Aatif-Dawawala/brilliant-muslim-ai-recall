import os
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from utilities.print_average import get_average

eval_results = "tests/eval_results.csv"

def test_get_average():
    average = get_average(eval_results)

    assert average == 4.339285714285714

test_get_average()