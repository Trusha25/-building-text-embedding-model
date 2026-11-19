import pandas as pd
from sklearn.model_selection import train_test_split

# Load prepared data
df = pd.read_csv("positive_pairs.csv")

# Split into training and testing data
train_df, test_df = train_test_split(
    df,
    test_size=0.2,
    random_state=42
)

# Save both datasets
train_df.to_csv("train_data.csv", index=False)
test_df.to_csv("test_data.csv", index=False)

print("Training data shape:", train_df.shape)
print("Testing data shape:", test_df.shape)