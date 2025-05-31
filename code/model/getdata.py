import json
import sys

with open("./mlir_functions.json", 'r') as file:
    data = json.load(file)

def adddata(fpath):
    global data
    with open(fpath) as f:
        lines = f.readlines()
    lines = [line.strip() for line in lines]
    result_set = set(data).union(set(lines))
    result_list = list(result_set)
    data = result_list


index = int(sys.argv[1]) - 1

adddata(f"../exp_srcipt/correct_generated_{index}.txt")
adddata(f"../exp_srcipt/new_generated_{index}.txt")

with open("./mlir_functions.json", 'w') as file:
    file.write(json.dumps(data))

