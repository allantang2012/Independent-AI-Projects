
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


# Test transformer decoder block
block = TransformerBlock(
    n_embd=128,
    num_heads=4
).to(device)

block_output = block(embedded)

print("\n--- Transformer Block Test ---")
print("Input shape:", embedded.shape)
print("Output shape:", block_output.shape)

assert block_output.shape == embedded.shape
assert torch.isfinite(block_output).all().item()

print("Transformer decoder block test passed!")


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


# Initialize the GPT model on Apple's GPU
model = GPTLanguageModel().to(device)

# Count trainable parameters
num_params = sum(
    p.numel() for p in model.parameters()
    if p.requires_grad
)

# Get a training batch
x, y = get_batch("train")

# Run a forward pass
logits, loss = model(x, y)

print("\n--- GPT Language Model Test ---")
print(f"Trainable parameters: {num_params:,}")
print("Input shape:", x.shape)
print("Logits shape:", logits.shape)
print("Initial loss:", loss.item())

assert logits.shape == (batch_size, block_size, vocab_size)
assert torch.isfinite(loss).item()

print("GPT language model forward pass successful!")


import time

# Training configuration
learning_rate = 3e-4
max_iters = 3000
eval_interval = 250
eval_batches = 10

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=learning_rate
)

# Evaluate the model without updating parameters
@torch.no_grad()
def estimate_loss():
    results = {}

    model.eval()

    for split in ["train", "val"]:
        losses = []

        for _ in range(eval_batches):
            xb, yb = get_batch(split)
            _, loss = model(xb, yb)
            losses.append(loss.item())

        results[split] = sum(losses) / len(losses)

    model.train()
    return results


# Training loop
print("\n--- Starting GPT Training ---")

start_time = time.time()
history = []

model.train()

for step in range(max_iters + 1):

    if step % eval_interval == 0:
        results = estimate_loss()

        elapsed = time.time() - start_time

        print(
            f"Step {step:4d} | "
            f"Train loss: {results['train']:.4f} | "
            f"Val loss: {results['val']:.4f} | "
            f"Time: {elapsed:.1f}s",
            flush=True
        )

        history.append({
            "step": step,
            "train_loss": results["train"],
            "val_loss": results["val"],
            "elapsed_seconds": elapsed
        })

    if step == max_iters:
        break

    # Get a batch of training data
    xb, yb = get_batch("train")

    # Forward pass
    logits, loss = model(xb, yb)

    # Reset gradients
    optimizer.zero_grad(set_to_none=True)

    # Backpropagation
    loss.backward()

    # Update model parameters
    optimizer.step()


# Save model weights and training history
import json

torch.save(model.state_dict(), "gpt_checkpoint.pt")

with open("training_history.json", "w") as f:
    json.dump(history, f, indent=2)

print("\nTraining complete!")
print(f"Total time: {time.time() - start_time:.1f} seconds")
print("Model saved to gpt_checkpoint.pt")
print("Metrics saved to training_history.json")


# Generate new text using the trained model
@torch.no_grad()
def generate(model, idx, max_new_tokens=300, temperature=1.0):
    model.eval()

    for _ in range(max_new_tokens):
        # Only use the most recent block_size tokens
        idx_cond = idx[:, -block_size:]

        # Predict the next character
        logits, _ = model(idx_cond)

        # Take predictions from the final position
        logits = logits[:, -1, :] / temperature

        # Convert logits to probabilities
        probs = F.softmax(logits, dim=-1)

        # Sample the next character
        next_token = torch.multinomial(probs, num_samples=1)

        # Append the sampled token
        idx = torch.cat((idx, next_token), dim=1)

    return idx


print("\n--- Generated Shakespeare Text ---")

prompt = "ROMEO:"
context = torch.tensor(
    [encode(prompt)],
    dtype=torch.long,
    device=device
)

generated_tokens = generate(
    model,
    context,
    max_new_tokens=500,
    temperature=0.8
)

generated_text = decode(generated_tokens[0].tolist())

print(generated_text)

with open("generated_sample.txt", "w", encoding="utf-8") as f:
    f.write(generated_text)


