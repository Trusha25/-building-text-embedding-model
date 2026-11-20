
import torch
import torch.nn as nn


class TextEmbeddingModel(nn.Module):

    def __init__(self, vocab_size, embedding_dim):
        super().__init__()

        self.embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=embedding_dim
        )

    def forward(self, input_ids):

        # Convert token IDs into word vectors
        word_vectors = self.embedding(input_ids)

        # Average word vectors to get sentence vector
        sentence_vector = word_vectors.mean(dim=0)

        return sentence_vector
