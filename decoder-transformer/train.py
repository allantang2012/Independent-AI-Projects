
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
