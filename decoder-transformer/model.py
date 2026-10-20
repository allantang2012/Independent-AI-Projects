
import torch
import torch.nn as nn
import torch.nn.functional as F

# Model configuration
block_size = 64
n_embd = 128
vocab_size = 65

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




class FeedForward(nn.Module):
    def __init__(self, n_embd):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd)
        )

    def forward(self, x):
        return self.net(x)


class TransformerBlock(nn.Module):
    def __init__(self, n_embd, num_heads):
        super().__init__()

        self.attention = MultiHeadAttention(n_embd, num_heads)
        self.feedforward = FeedForward(n_embd)

        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        # Attention + residual connection
        x = x + self.attention(self.ln1(x))

        # Feedforward + residual connection
        x = x + self.feedforward(self.ln2(x))

        return x


# Full GPT-style decoder-only language model
class GPTLanguageModel(nn.Module):
    def __init__(self):
        super().__init__()

        self.token_embedding = nn.Embedding(vocab_size, n_embd)
        self.position_embedding = nn.Embedding(block_size, n_embd)

        self.blocks = nn.Sequential(*[
            TransformerBlock(n_embd, num_heads=4)
            for _ in range(4)
        ])

        self.ln_f = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size)

    def forward(self, idx, targets=None):
        B, T = idx.shape

        token_emb = self.token_embedding(idx)

        positions = torch.arange(T, device=idx.device)
        pos_emb = self.position_embedding(positions)

        x = token_emb + pos_emb

        x = self.blocks(x)
        x = self.ln_f(x)

        logits = self.lm_head(x)

        loss = None

        if targets is not None:
            B, T, C = logits.shape

            loss = F.cross_entropy(
                logits.reshape(B * T, C),
                targets.reshape(B * T)
            )

        return logits, loss


