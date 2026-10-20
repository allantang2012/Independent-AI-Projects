import math
import torch
from transformers import AutoModelForCausalLM

device = "mps" if torch.backends.mps.is_available() else "cpu"

# Load validation tokens
val_data = torch.load(
    "val_tokens.pt",
    weights_only=True
)

# Use fixed, non-overlapping validation sequences
block_size = 128
num_batches = min(100, (len(val_data) - 1) // block_size)

def evaluate(model):
    model.eval()
    total_loss = 0.0

    with torch.no_grad():
        for i in range(num_batches):
            start = i * block_size

            # 129 tokens produce 128 predictions
            x = val_data[start:start + block_size + 1]
            x = x.unsqueeze(0).to(device)

            output = model(input_ids=x, labels=x)
            total_loss += output.loss.item()

    return total_loss / num_batches


models = {
    "Pretrained": "distilbert/distilgpt2",
    "Fine-tuned": "./finetuned_model"
}

print("\n--- Fixed Validation Evaluation ---")
print("Validation sequences:", num_batches)

for name, path in models.items():
    print(f"\nEvaluating {name}...")

    model = AutoModelForCausalLM.from_pretrained(path).to(device)

    loss = evaluate(model)
    perplexity = math.exp(loss)

    print(f"{name} loss: {loss:.4f}")
    print(f"{name} perplexity: {perplexity:.2f}")

    del model

    if device == "mps":
        torch.mps.empty_cache()
