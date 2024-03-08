import torch
from src.transformer.embeddings import Embeddings

def test_embeddings_output_shape():
    """Test that the output of the Embeddings layer has the correct shape."""
    vocab_size = 10
    embedding_dim = 5
    batch_size = 2
    sequence_length = 3

    model = Embeddings(vocab_size, embedding_dim)
    input = torch.randint(0, vocab_size, (batch_size, sequence_length))

    output = model(input)
    expected_shape = (batch_size, sequence_length, embedding_dim)

    assert output.shape == expected_shape, f"Expected output shape {expected_shape}, but got {output.shape}"
