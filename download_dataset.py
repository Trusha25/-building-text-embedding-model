
from datasets import load_dataset

print("Downloading MS MARCO dataset...")

dataset = load_dataset(
    "microsoft/ms_marco",
    "v1.1",
    split="train[:5000]"
)

print("\nDataset downloaded successfully!")
print(dataset)

# --- Dataset Exploration (EDA) ---
print("\nDataset Shape:")
print(dataset.shape)
print("\nNumber of Rows:")
print(dataset.num_rows)
print("\nNumber of Columns:")
print(dataset.num_columns)

print("\nDataset Features:")
print(dataset.features)

print("\nColumn Names:")
print(dataset.column_names)

print("\nFirst Record:")
print(dataset[0])

# Count relevant and non-relevant labelled passages
selected_count = 0
not_selected_count = 0
for row in dataset:
    labels = row["passages"]["is_selected"]
    selected_count += labels.count(1)
    not_selected_count += labels.count(0)
print("\nSelected passages:", selected_count)
print("Not selected passages:", not_selected_count)

# Check missing values in queries
missing_queries = sum(
    1 for row in dataset
    if not row["query"] or not row["query"].strip()
)
print("\nMissing queries:", missing_queries)
