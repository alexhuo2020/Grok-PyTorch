# Get parameters partition with the runing the test_run.py on CPU
# Output:
#   file 'params_partition.csv': contains rows of the form `module/submodule/...: shape of the parameter`
#   file 'params_partition.json': list of module: index in json format

# open the error log file
import json
with open("error.txt", "r") as f:
    content = f.read()

# Find the first occurrence of "param=" and get the params list
index = content.find("params=")
content = content[index+7:]
index = content.find("--------------------")
content = content[:index-3]



import re
content = re.sub(r"Traced<ShapedArray\(([^)]+)\)>with<DynamicJaxprTrace\(level=\d+/\d+\)>", r"'\1'", content)

try:
    data = eval(content)  # Cautious eval of cleaned content
    with open("params_partition.txt","w") as f:

        for key, subdict in data.items():
            for subkey, value in subdict.items():
                f.writelines(f"{key}/{subkey}: {value}\n")
except Exception as e:
    print("Error parsing file:", e)

with open("params_partition.txt", 'r') as f:
    data = f.readlines()

import pandas as pd
df = pd.DataFrame()
for data in data:
    shape_str = data.split(":")[1]
    match = re.match(r" float32\[(.*)\]", shape_str)
    if match:
        # Get the dimensions as a string and split by commas
        dimensions = match.group(1).split(',')
        dimensions = [int(dim) for dim in dimensions]
        new_row = {"name":data.split(":")[0], "dim": dimensions}
    df = df._append(new_row, ignore_index=True)
df.to_csv("params_partition.csv")


with open("params_partition.json", "w") as outfile: 
    json.dump(dict(zip(df['name'].values, range(len(df)))), outfile)
    