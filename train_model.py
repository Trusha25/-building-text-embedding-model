
import pandas as pd
import pickle
import torch
import torch.nn.functional as F

from model import TextEmbeddingModel


# ---------------------------------
# 1. Load vocabulary
# ---------------------------------

with open("vocab.pkl", "rb") as f:
    vocab = pickle.load(f)

print("Vocabulary loaded:", len(vocab))


# ---------------------------------
# 2. Load triplet dataset
# ---------------------------------

df = pd.read_csv("triplets_train.csv")

df = df[
    ["query", "positive_passage", "negative_passage"]
].fillna("")

print("Training dataset:", df.shape)


# ---------------------------------
# 3. Model configuration
# ---------------------------------

embedding_dim = 128
learning_rate = 0.001
margin = 0.2
epochs = 3

model = TextEmbeddingModel(
    vocab_size=len(vocab),
    embedding_dim=embedding_dim
)


# ---------------------------------
# 4. Optimizer
# ---------------------------------

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=learning_rate
)


# ---------------------------------
# 5. Convert text into token IDs
# ---------------------------------

def text_to_ids(text):

    tokens = str(text).lower().split()

    ids = [
        vocab.get(token, vocab["<UNK>"])
        for token in tokens
    ]

    # Handle empty text
    if len(ids) == 0:
        ids = [vocab["<UNK>"]]

    return torch.tensor(ids, dtype=torch.long)


# ---------------------------------
# 6. Training loop
# ---------------------------------

for epoch in range(epochs):

    total_loss = 0

    # Shuffle triplets for this epoch
    shuffled_df = df.sample(frac=1)

    for row in shuffled_df.itertuples(index=False):

        query = row.query
        positive_passage = row.positive_passage
        negative_passage = row.negative_passage

        # Convert all three texts into IDs
        query_ids = text_to_ids(query)
        positive_ids = text_to_ids(positive_passage)
        negative_ids = text_to_ids(negative_passage)

        # Generate sentence embeddings
        query_vector = model(query_ids)

        positive_vector = model(positive_ids)

        negative_vector = model(negative_ids)

        # Calculate cosine similarities
        positive_similarity = F.cosine_similarity(
            query_vector,
            positive_vector,
            dim=0
        )

        negative_similarity = F.cosine_similarity(
            query_vector,
            negative_vector,
            dim=0
        )

        # Calculate triplet loss
        loss = F.relu(
            negative_similarity
            - positive_similarity
            + margin
        )

        # Reset previous gradients
        optimizer.zero_grad()

        # Calculate gradients
        loss.backward()

        # Update embedding weights
        optimizer.step()

        total_loss += loss.item()

    average_loss = total_loss / len(df)

    print(
        f"Epoch {epoch + 1}/{epochs} "
        f"| Average Loss: {average_loss:.4f}"
    )


# ---------------------------------
# 7. Save trained model
# ---------------------------------

torch.save(
    {
        "model_state_dict": model.state_dict(),
        "vocab_size": len(vocab),
        "embedding_dim": embedding_dim
    },
    "trained_embedding_model.pth"
)

print("\nTraining completed!")
print("Model saved successfully!")
