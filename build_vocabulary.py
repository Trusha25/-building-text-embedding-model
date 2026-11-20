import pandas as pd
import pickle

# Load only training data
df = pd.read_csv("train_data.csv")

# Create an empty vocabulary
vocab = {}

# Combine queries and positive passages
all_text = (
    df["query"].fillna("").tolist()
    + df["positive_passage"].fillna("").tolist()
)

# Build vocabulary
for sentence in all_text:
    tokens = sentence.lower().split()

    for token in tokens:
        if token not in vocab:
            vocab[token] = len(vocab)

# Add an unknown token
vocab["<UNK>"] = len(vocab)

# Display vocabulary information
print("Training vocabulary size:", len(vocab))
print("\nFirst 20 entries:")
print(list(vocab.items())[:20])
with open("vocab.pkl", "wb") as f:
    pickle.dump(vocab, f)

print("\nVocabulary saved successfully!")