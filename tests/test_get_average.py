import os
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from utilities.print_average import get_average

eval_results = "tests/eval_results.csv"

def test_get_average():
    average = get_average(eval_results)

    assert average == {'custom_text-quality/score': 4.379746835443038, 'instruction-following/score': 4.935897435897436}

test_get_average()