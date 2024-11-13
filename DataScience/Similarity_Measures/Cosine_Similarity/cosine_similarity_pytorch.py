import torch
import torch.nn.functional as F

# Define two vectors (as PyTorch tensors)
vector_a = torch.tensor([0.0, 0.0, 1.0])
vector_b = torch.tensor([1.0, 1.0, 1.0])

# Cosine similarity computation
cos_sim = F.cosine_similarity(vector_a.unsqueeze(0), vector_b.unsqueeze(0))

print(f"Cosine Similarity between vector_a and vector_b: {cos_sim.item():.4f}")
