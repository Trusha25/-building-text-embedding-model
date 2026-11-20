
import pandas as pd
import pickle
import torch
import torch.nn as nn

# 1. Load our vocabulary
with open("vocab.pkl", "rb") as f:
    vocab = pickle.load(f)

# 2. Define embedding dimensions
embedding_dim = 128

# 3. Create the embedding layer
embedding = nn.Embedding(
    num_embeddings=len(vocab),
    embedding_dim=embedding_dim
)

# 4. Function to convert text into IDs
def text_to_ids(text):
    tokens = text.lower().split()

    ids = [
        vocab.get(token, vocab["<UNK>"])
        for token in tokens
    ]

    return ids

# 5. Function to generate a sentence embedding
def get_sentence_embedding(text):

    # Convert sentence into IDs
    ids = text_to_ids(text)

    # Convert IDs into a PyTorch tensor
    input_ids = torch.tensor(ids, dtype=torch.long)

    # Look up the word embeddings
    word_vectors = embedding(input_ids)

    # Average the word vectors
    sentence_vector = word_vectors.mean(dim=0)

    return sentence_vector


# 6. Load an example query
df = pd.read_csv("train_data.csv")

query = df["query"].iloc[0]

# 7. Generate its embedding
vector = get_sentence_embedding(query)

# Display results
print("Original query:", query)
print("Query IDs:", text_to_ids(query))
print("Embedding shape:", vector.shape)
print("First 10 embedding values:", vector[:10])
