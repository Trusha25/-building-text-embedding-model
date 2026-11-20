import pandas as pd
import pickle

# Load training and testing data
train_df = pd.read_csv("train_data.csv")
test_df = pd.read_csv("test_data.csv")

# Load vocabulary
with open("vocab.pkl", "rb") as f:
    vocab = pickle.load(f)

# Function to convert text into IDs
def text_to_ids(text):
    tokens = text.lower().split()

    ids = [
        vocab.get(token, vocab["<UNK>"])
        for token in tokens
    ]

    return ids

# Convert training queries and passages
train_df["query_ids"] = train_df["query"].apply(text_to_ids)
train_df["passage_ids"] = train_df["positive_passage"].apply(text_to_ids)

# Convert testing queries and passages
test_df["query_ids"] = test_df["query"].apply(text_to_ids)
test_df["passage_ids"] = test_df["positive_passage"].apply(text_to_ids)

# Display an example
print("Original query:")
print(train_df["query"].iloc[0])

print("\nQuery IDs:")
print(train_df["query_ids"].iloc[0])

print("\nOriginal passage:")
print(train_df["positive_passage"].iloc[0][:200])

print("\nPassage IDs:")
print(train_df["passage_ids"].iloc[0][:20])