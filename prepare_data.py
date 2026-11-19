
from datasets import load_dataset
import pandas as pd

# Load the dataset (it should use the cached download)
dataset = load_dataset(
    "microsoft/ms_marco",
    "v1.1",
    split="train[:5000]"
)

prepared_data = []

# Go through every query
for row in dataset:

    query = row["query"]

    passages = row["passages"]["passage_text"]
    labels = row["passages"]["is_selected"]

    # Extract passages labelled as relevant
    for passage, label in zip(passages, labels):

        if label == 1 and passage.strip():

            prepared_data.append({
                "query_id": row["query_id"],
                "query": query,
                "positive_passage": passage
            })

# Convert the list into a DataFrame
df = pd.DataFrame(prepared_data)

print("Prepared Dataset Shape:", df.shape)
print("\nFirst 5 Query-Passage Pairs:")
print(df.head())

# Save the prepared data
df.to_csv("positive_pairs.csv", index=False)

print("\nData saved successfully!")
