
import torch

from model import GPTLanguageModel, block_size

# Select hardware
device = "mps" if torch.backends.mps.is_available() else "cpu"

# Load the same vocabulary used during training
with open("input.txt", "r", encoding="utf-8") as f:
    text = f.read()

chars = sorted(set(text))
stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for i, ch in enumerate(chars)}

def encode(s):
    return [stoi[c] for c in s]

def decode(tokens):
    return "".join(itos[i] for i in tokens)


# Create the same GPT architecture
model = GPTLanguageModel().to(device)

# Load trained weights
checkpoint = torch.load(
    "gpt_checkpoint_10000.pt",
    map_location=device,
    weights_only=True
)

model.load_state_dict(checkpoint)
model.eval()

print("Trained GPT checkpoint loaded successfully!")


@torch.no_grad()
def generate(model, idx, max_new_tokens=500, temperature=0.8):
    if temperature <= 0:
        raise ValueError("Temperature must be positive")

    model.eval()

    for _ in range(max_new_tokens):
        # Keep only the latest context tokens
        idx_cond = idx[:, -block_size:]

        # Predict next-character probabilities
        logits, _ = model(idx_cond)
        logits = logits[:, -1, :] / temperature

        probs = torch.softmax(logits, dim=-1)

        # Sample a character
        next_token = torch.multinomial(probs, num_samples=1)

        # Append it to the sequence
        idx = torch.cat((idx, next_token), dim=1)

    return idx


# Generate Shakespeare-style text
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

print("\n--- Generated Shakespeare Text ---")
print(generated_text)
