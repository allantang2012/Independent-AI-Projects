
import torch

# Select Apple's GPU if available
device = "mps" if torch.backends.mps.is_available() else "cpu"

# Load our dataset
with open("input.txt", "r", encoding="utf-8") as f:
    text = f.read()

# Build our character vocabulary
chars = sorted(list(set(text)))
vocab_size = len(chars)

# Map characters to integers
stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for i, ch in enumerate(chars)}

# Encode and decode text
def encode(s):
    return [stoi[c] for c in s]

def decode(tokens):
    return "".join(itos[i] for i in tokens)

# Convert text into a tensor of token IDs
data = torch.tensor(encode(text), dtype=torch.long)

# Split into training and validation sets
n = int(0.9 * len(data))
train_data = data[:n]
val_data = data[n:]

print("Device:", device)
print("Vocabulary size:", vocab_size)
print("Training tokens:", len(train_data))
print("Validation tokens:", len(val_data))
print("Example encoded text:", encode("hello"))
print("Decoded text:", decode(encode("hello")))


import torch.nn as nn
import torch.nn.functional as F

# Hyperparameters
torch.manual_seed(42)

batch_size = 16
block_size = 64
n_embd = 128


# Generate a batch of training examples
def get_batch(split):
    source = train_data if split == "train" else val_data

    ix = torch.randint(
        0, len(source) - block_size,
        (batch_size,)
    )

    x = torch.stack([
        source[i:i + block_size] for i in ix
    ])

    y = torch.stack([
        source[i + 1:i + block_size + 1] for i in ix
    ])

    return x.to(device), y.to(device)


# Single-head causal self-attention
class CausalSelfAttention(nn.Module):
    def __init__(self, n_embd):
        super().__init__()

        self.query = nn.Linear(n_embd, n_embd, bias=False)
        self.key = nn.Linear(n_embd, n_embd, bias=False)
        self.value = nn.Linear(n_embd, n_embd, bias=False)

        # Mask future token positions
        self.register_buffer(
            "mask",
            torch.tril(torch.ones(block_size, block_size))
        )

    def forward(self, x):
        B, T, C = x.shape

        q = self.query(x)
        k = self.key(x)
        v = self.value(x)

        # Compute scaled attention scores
        scores = q @ k.transpose(-2, -1) * (C ** -0.5)

        # Hide future characters
        scores = scores.masked_fill(
            self.mask[:T, :T] == 0,
            float("-inf")
        )

        # Convert scores into attention weights
        weights = F.softmax(scores, dim=-1)

        # Combine token information
        out = weights @ v

        return out


class MultiHeadAttention(nn.Module):
    def __init__(self, n_embd, num_heads):
        super().__init__()

        assert n_embd % num_heads == 0

        self.num_heads = num_heads
        self.head_dim = n_embd // num_heads

        self.heads = nn.ModuleList([
            CausalAttentionHead(n_embd, self.head_dim)
            for _ in range(num_heads)
        ])

        self.proj = nn.Linear(n_embd, n_embd)

    def forward(self, x):
        # Run attention heads in parallel conceptually
        outputs = [head(x) for head in self.heads]

        # Combine outputs
        out = torch.cat(outputs, dim=-1)

        # Project back to embedding dimension
        return self.proj(out)


class CausalAttentionHead(nn.Module):
    def __init__(self, n_embd, head_dim):
        super().__init__()

        self.query = nn.Linear(n_embd, head_dim, bias=False)
        self.key = nn.Linear(n_embd, head_dim, bias=False)
        self.value = nn.Linear(n_embd, head_dim, bias=False)

        self.register_buffer(
            "mask",
            torch.tril(torch.ones(block_size, block_size))
        )

    def forward(self, x):
        B, T, C = x.shape

        q = self.query(x)
        k = self.key(x)
        v = self.value(x)

        scores = q @ k.transpose(-2, -1)
        scores = scores * (q.shape[-1] ** -0.5)

        scores = scores.masked_fill(
            self.mask[:T, :T] == 0,
            float("-inf")
        )

        weights = F.softmax(scores, dim=-1)

        return weights @ v


# Test the attention layer
embedding = nn.Embedding(vocab_size, n_embd).to(device)
attention = CausalSelfAttention(n_embd).to(device)

x, y = get_batch("train")
embedded = embedding(x)
output = attention(embedded)

print("\n--- Causal Self-Attention Test ---")
print("Input shape:", x.shape)
print("Target shape:", y.shape)
print("Embedded shape:", embedded.shape)
print("Attention output shape:", output.shape)


# Test multi-head attention
multi_head = MultiHeadAttention(
    n_embd=128,
    num_heads=4
).to(device)

multi_output = multi_head(embedded)

print("\n--- Multi-Head Attention Test ---")
print("Number of heads:", multi_head.num_heads)
print("Dimension per head:", multi_head.head_dim)
print("Input shape:", embedded.shape)
print("Output shape:", multi_output.shape)

assert multi_output.shape == embedded.shape
print("Multi-head attention test passed!")

