
import pandas as pd

# Load our training dataset
df = pd.read_csv("train_data.csv")

# Shuffle positive passages to create negative examples
negative_passages = (
    df["positive_passage"]
    .shift(-1)
)

# Move the first passage to the last position
negative_passages.iloc[-1] = df["positive_passage"].iloc[0]

# Create triplets
triplets = pd.DataFrame({
    "query": df["query"],
    "positive_passage": df["positive_passage"],
    "negative_passage": negative_passages
})

# Save the triplets
triplets.to_csv("triplets_train.csv", index=False)

print("Triplets created successfully!")
print("Dataset shape:", triplets.shape)

print("\nFirst triplet:")
print("Query:", triplets["query"].iloc[0])
print("\nPositive passage:")
print(triplets["positive_passage"].iloc[0])
print("\nNegative passage:")
print(triplets["negative_passage"].iloc[0])
