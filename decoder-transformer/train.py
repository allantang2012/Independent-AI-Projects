
import json
import time

import torch
from model import GPTLanguageModel, block_size, n_embd, vocab_size

# Select Apple's GPU if available
device = "mps" if torch.backends.mps.is_available() else "cpu"

# Load our dataset
with open("input.txt", "r", encoding="utf-8") as f:
    text = f.read()

# Build our character vocabulary
chars = sorted(list(set(text)))
assert len(chars) == vocab_size, "Dataset vocabulary does not match model"

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
max_iters = 10000
eval_interval = 500
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

torch.save(model.state_dict(), "gpt_smoke_test.pt")

with open("training_history_smoke_test.json", "w") as f:
    json.dump(history, f, indent=2)

print("\nTraining complete!")
print(f"Total time: {time.time() - start_time:.1f} seconds")
print("Model saved to gpt_checkpoint.pt")
print("Metrics saved to training_history.json")