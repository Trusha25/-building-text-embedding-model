
import torch
import torch.nn.functional as F

# Example similarity scores
positive_similarity = torch.tensor(0.9859)
negative_similarity = torch.tensor(-0.1517)

# Minimum desired gap
margin = 0.2

# Calculate triplet loss
loss = F.relu(
    negative_similarity - positive_similarity + margin
)

print("Positive similarity:", positive_similarity.item())
print("Negative similarity:", negative_similarity.item())
print("Triplet loss:", loss.item())
