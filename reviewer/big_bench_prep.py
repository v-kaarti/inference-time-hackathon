import json
import pandas as pd

import glob

json_files = glob.glob("big_bench_data/*.jsonl")

formatted_data = []
for file_path in json_files:
    with open(file_path, "r") as file:
        for line in file:
            data = json.loads(line)

            input_equation = data["input"]
            steps = data["steps"]
            mistake_index = data["mistake_index"]

            # Construct incremental sequences
            for i in range(len(steps)):
                
                if mistake_index is None:
                    step_sequence = " ".join(steps[:len(steps)])
                    label = 0  # Correct reasoning
                    break
                
                step_sequence = " ".join(steps[: i + 1])
                # Determine label
                if mistake_index is None or i < mistake_index:
                    label = 0  # Correct reasoning
                elif i == mistake_index:
                    label = 1  # First mistake appears
                else:
                    break  # Stop including steps after the first mistake

                # Append to dataset
                formatted_data.append({
                    "input_equation": input_equation,
                    "step_sequence": step_sequence,
                    "label": label
                })


# Define the output file path
output_file_path = "incremental_reasoning_dataset.jsonl"

# Save the dataset to a JSONL file
with open(output_file_path, "w") as file:
    for record in formatted_data:
        file.write(json.dumps(record) + "\n")

# Provide the download link
output_file_path

