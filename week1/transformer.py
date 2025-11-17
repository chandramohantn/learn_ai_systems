import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class InputEmbedding(nn.Module):
    """
    Converts input token indices into dense vector representations

    Why its important: Models process number and not text. Embeddings map tokens to a continuous vector space where semantic similarity is reflected by distance and direction.
    How we build it: We use PyTorch's nn.Embedding layer and scale the outputs by the square root of the embedding dimension. This is mainly for training stability.
    """
    def __init__(self, embedding_size, vocab_size):
        super().__init__()
        self.embedding_size = embedding_size
        self.vocab_size = vocab_size
        self.embedding = nn.Embedding(vocab_size, embedding_size)

    def forward(self, x):
        return self.embedding(x) * math.sqrt(self.embedding_size)

class PositionalEncoding(nn.Module):
    """
    Adds positional information to token embeddings using sinusoidal functions.

    Why its important: Transformers lack inherent sequence order awareness. Positional encodings provide a way to inject order information into the model.
    How we build it: We create a fixed positional encoding matrix using sine and cosine functions of different frequencies, then add this to the input embeddings.
    """
    def __init__(self, embedding_size, droput: float, sequence_len=5000):
        super().__init__()
        self.dropout = nn.Dropout(droput)
        self.embedding_size = embedding_size

        # Create a positional encoding matrix of shape (sequence_len, embedding_size)
        positional_encoding = torch.zeros(sequence_len, embedding_size)
        position = torch.arange(0, sequence_len, dtype=torch.float).unsqueeze(1) # (sequence_len, 1)
        div_term = torch.exp(torch.arange(0, embedding_size, 2).float() * (-math.log(10000.0) / embedding_size))

        # Apply sine to even indices and cosine to odd indices
        positional_encoding[:, 0::2] = torch.sin(position * div_term)
        positional_encoding[:, 1::2] = torch.cos(position * div_term)
        positional_encoding = positional_encoding.unsqueeze(0) # (1, sequence_len, embedding_size) for batch broadcasting

        self.register_buffer("positional_encoding", positional_encoding) # Not a model parameter but a part of the state

    def forward(self, x):
        """
        Args:
            x: Input tensor of shape (batch_size, sequence_len, embedding_size)
        """
        x = x + self.positional_encoding[:, :x.size(1)] # Add positional encoding
        return self.dropout(x)

class LayerNormalization(nn.Module):
    """
    Normalizes the inputs across the features for each data point.

    Why its important: Layer normalization stabilizes and accelerates training by normalizing the inputs to each layer, reducing internal covariate shift.
    How we build it: We compute the mean and standard deviation across the feature dimension and normalize the inputs accordingly, followed by learnable scaling and shifting.
    """
    def __init__(self, embedding_size: int, eps: float=1e-6):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(embedding_size))  # Scale parameter
        self.beta = nn.Parameter(torch.zeros(embedding_size))  # Shift parameter
        self.eps = eps

    def forward(self, x):
        mean = x.mean(dim=-1, keepdims=True)
        std = x.std(dim=-1, keepdims=True, unbiased=False)
        normalised = (x - mean) / (std + self.eps)
        return self.gamma * normalised + self.beta

class MultiHeadAttention(nn.Module):
    """
    The multi-head attention mechanism allows the model to focus on different parts of the input sequence simultaneously.
    ANALOGY: Researching a topic (query) when you have multiple books (keys) with different content (values). Attention is like deciding which books are relevant and how much to read from each.
    """
    def __init__(self, embedding_size: int, n_heads: int, dropout: float):
        super().__init__()
        assert embedding_size % n_heads == 0, "Embedding size must be divisible by number of heads"
        self.embedding_size = embedding_size
        self.n_heads = n_heads
        self.dimensions_per_head = embedding_size // n_heads

        self.w_q = nn.Linear(embedding_size, embedding_size)
        self.w_k = nn.Linear(embedding_size, embedding_size)
        self.w_v = nn.Linear(embedding_size, embedding_size)
        self.w_o = nn.Linear(embedding_size, embedding_size)
        self.dropout = nn.Dropout(dropout)

    # Attention mechanism: Core calculation
    @staticmethod
    def attention(query, key, value, mask=None, dropout=None):
        """
        Computes the scaled dot product attention
        """
        head_dimension = query.size(-1)
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(head_dimension)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))

        attention_weights = F.softmax(scores, dim=-1)
        if dropout is not None:
            attention_weights = dropout(attention_weights)

        return torch.matmul(attention_weights, value), attention_weights
    
    def forward(self, query, key, value, mask=None):
        """
        Query, Key, Value: Tensors of shape (batch_size, seq_len, embedding_size)
        mask: To prevent attention of certain positions
        """
        batch_size = query.size(0)

        # Linear projects and split into multiple heads
        query = self.w_q(query).view(batch_size, -1, self.n_heads, self.dimensions_per_head).transpose(1, 2)
        key = self.w_k(key).view(batch_size, -1, self.n_heads, self.dimensions_per_head).transpose(1, 2)
        value = self.w_v(value).view(batch_size, -1, self.n_heads, self.dimensions_per_head).transpose(1, 2)

        # Apply attention on all the projected vectors in batch
        x, attention_weights = self.attention(query, key, value, mask, self.dropout)

        # Concatenate heads and put through final linear layer
        x = x.transpose(1, 2).contiguous().view(batch_size, -1, self.embedding_size)
        return self.w_o(x)

class PositionWiseFFN(nn.Module):
    """
    A fully connected feed-forward network applied to each position separately and identically.

    Why its important: It introduces non-linearity and allows the model to learn complex transformations of the input features at each position.
    How we build it: We use two linear layers with a ReLU activation in between, applied independently to each position in the sequence.
    """
    def __init__(self, embedding_size: int, hidden_size: int, dropout: float):
        super().__init__()
        self.linear1 = nn.Linear(embedding_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, embedding_size)
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.ReLU()

    def forward(self, x):
        x = self.linear1(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.linear2(x)
        return x

class EncoderBlock(nn.Module):
    """
    Combines multi-head attention and the feed-forward network with residual connections and layer normalization.

    Why its important: Transforms input sequences into contextualized representations. Stacking blocks allows the model to build up increasingly abstract and complex representations of the input.
    How we build it: Wrap the MultiHeadAttention and PositionWiseFFN with residual connections and layer norm for stable training.
    """
    def __init__(self, embedding_size: int, n_heads: int, hidden_size: int, dropout: float):
        super().__init__()
        self.multi_head_attention = MultiHeadAttention(embedding_size, n_heads, dropout)
        self.layer_norm1 = LayerNormalization(embedding_size)
        self.position_wise_ffn = PositionWiseFFN(embedding_size, hidden_size, dropout)
        self.layer_norm2 = LayerNormalization(embedding_size)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, mask=None):
        # Multi-head attention sub-layer
        attention_output = self.multi_head_attention(x, x, x, mask)
        x = self.layer_norm1(x + self.dropout(attention_output))  # Add & Norm

        # Position-wise feed-forward sub-layer
        feed_forward_output = self.position_wise_ffn(x)
        x = self.layer_norm2(x + self.dropout(feed_forward_output))  # Add & Norm
        return x
    
class DecoderBlock(nn.Module):
