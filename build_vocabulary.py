
import pandas as pd

# 1. Load our prepared dataset
df = pd.read_csv("positive_pairs.csv")

# 2. Create an empty vocabulary
vocab = {}

# 3. Combine queries and positive passages
all_text = (
    df["query"].fillna("").tolist()
    + df["positive_passage"].fillna("").tolist()
)

# 4. Build vocabulary
for sentence in all_text:

    # Convert text to lowercase and split into tokens
    tokens = sentence.lower().split()

    # Assign IDs to unique tokens
    for token in tokens:
        if token not in vocab:
            vocab[token] = len(vocab)

# 5. Display vocabulary information
print("Vocabulary size:", len(vocab))

print("\nFirst 20 vocabulary entries:")
print(list(vocab.items())[:20])
