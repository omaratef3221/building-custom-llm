import torch
import torch.nn as nn

def create_embedding_layer(vocab_size: int, d_model: int):
    return nn.Embedding(vocab_size, d_model)
3
def embed_tokens(embedding_layer: nn.Embedding, tokens: torch.tensor, d_model: int):
    return embedding_layer(tokens) * torch.sqrt(torch.tensor(d_model))



