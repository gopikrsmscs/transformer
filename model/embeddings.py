from torch import nn


class Embeddings(nn.Module):
    def __init__(self, vocab_size, embedding_dim):
        super().__init__()
        self.embeddings = nn.Embedding(vocab_size, embedding_dim)

    def forward(self, input):
        embeddings = self.embeddings(input)
        return embeddings
