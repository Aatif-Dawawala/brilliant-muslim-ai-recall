import csv

file_name = "eval_results.csv"

def get_average():
    scores = []
    sum = 0

    with open(file_name, 'r') as file: 
        dict_reader = csv.DictReader(file)
        for row_dict in dict_reader:
            scores.append(float(row_dict["custom_text-quality/score"]))

    for temp in scores:
        sum += temp
        
    return sum/len(scores)

print(f"The average score is: {round(get_average(), 2)}")