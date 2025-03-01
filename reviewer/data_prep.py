import pandas as pd
import json
import glob

# Get all CSV files in the directory
csv_files = glob.glob("data/*.csv")

# List to store data
classifier_data = []

# Load reference dataset for consistent label formatting
reference_df = pd.read_csv("data/edu_train.csv") if "data/edu_train.csv" in csv_files else None
if reference_df is not None and "updated_label" in reference_df.columns:
    reference_labels = set(reference_df["updated_label"].dropna().unique())
else:
    reference_labels = set()

# Iterate through all CSV files and append data
for file_path in csv_files:
    print(f"Proc. {file_path}")
    df = pd.read_csv(file_path)
    
    # Check if the file is edu_train.csv format
    if "source_article" in df.columns and "updated_label" in df.columns:
        df_filtered = df[["source_article", "updated_label"]]
    
    # Check if the file is LFUD.csv format
    elif "sentence" in df.columns and "fallacy_type" in df.columns:
        df_filtered = df[["sentence", "fallacy_type"]]
        df_filtered.rename(columns={"sentence": "source_article", "fallacy_type": "updated_label"}, inplace=True)
    
    # Check if the file is hf_fallacy_train.csv format
    elif "statement" in df.columns and "label" in df.columns:
        df_filtered = df[["statement", "label"]]
        df_filtered.rename(columns={"statement": "source_article", "label": "updated_label"}, inplace=True)
    
    else:
        continue  # Skip files that do not match any known format
    
    for _, row in df_filtered.iterrows():
        classifier_data.append({"prompt": row["source_article"], "label": row["updated_label"]})

import json
json_file_paths = glob.glob("data/*.json")

for json_file_path in json_file_paths:
    print(f"Proc. {json_file_path}")
    with open(json_file_path, "r", encoding="utf-8") as f:
        json_data = json.load(f)

    # Extract the relevant keys for (prompt, label) pairs
    for entry in json_data:
        if "comments" in entry:
            for comment in entry["comments"]:
                if "comment" in comment and "fallacy" in comment:
                    classifier_data.append({
                        "prompt": comment["comment"],
                        "label": comment["fallacy"]
                    })


# Save as JSONL format
classifier_jsonl_file_path = "classifier_dataset.jsonl"
with open(classifier_jsonl_file_path, "w", encoding="utf-8") as f:
    for entry in classifier_data:
        f.write(json.dumps(entry) + "\n")


file_path = "classifier_dataset.jsonl"  
df = pd.read_json(file_path, lines=True)

# Define a mapping of redundant labels to their chosen replacements
label_mapping = {
    'intentional': 'intentional fallacy',
    'faulty generalization': 'hasty generalization',
    'appeal to majority': 'appeal to authority',
    'fallacy of logic': 'deductive fallacy',
    'fallacy of credibility': 'ad hominem',
    'fallacy of relevance': 'fallacy of extension'
}

df['label'] = df['label'].replace(label_mapping)

label_mapping_update = {
    'hasty generalization': 'deductive fallacy',  # Both relate to flawed logic
    'ad populum': 'appeal to authority',  # Both rely on external validation
    'appeal to nature': 'appeal to tradition',  # Similar reasoning
    'appeal to worse problems': 'appeal to emotion',  # Often an emotional appeal
    'miscellaneous': None  # Remove miscellaneous labels
}

df['label'] = df['label'].replace(label_mapping_update)
df = df.dropna(subset=['label'])

final_file_path = "classifier_dataset_cleaned.jsonl"
df.to_json(final_file_path, orient='records', lines=True)
