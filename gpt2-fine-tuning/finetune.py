
import json
import time

import torch
from transformers import AutoModelForCausalLM

torch.manual_seed(42)

device = "mps" if torch.backends.mps.is_available() else "cpu"
model_name = "distilbert/distilgpt2"

# Training configuration
batch_size = 2
block_size = 128
learning_rate = 2e-5
max_steps = 50
eval_interval = 10
eval_batches = 5

# Load prepared data
train_data = torch.load(
    "train_tokens.pt", weights_only=True
)
val_data = torch.load(
    "val_tokens.pt", weights_only=True
)

# Load pretrained model
print("Loading pretrained DistilGPT2...")
model = AutoModelForCausalLM.from_pretrained(model_name)
model = model.to(device)

# Create training batches
def get_batch(split):
    data = train_data if split == "train" else val_data

    ix = torch.randint(
        0,
        len(data) - block_size,
        (batch_size,)
    )

    x = torch.stack([
        data[i:i + block_size] for i in ix
    ]).to(device)

    return x


optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=learning_rate
)

@torch.no_grad()
def estimate_loss():
    model.eval()
    results = {}

    for split in ["train", "val"]:
        losses = []

        for _ in range(eval_batches):
            x = get_batch(split)
            output = model(
                input_ids=x,
                labels=x
            )
            losses.append(output.loss.item())

        results[split] = sum(losses) / len(losses)

    model.train()
    return results


print("\n--- Starting Fine-Tuning ---")

start_time = time.time()
history = []

model.train()

for step in range(max_steps + 1):

    if step % eval_interval == 0:
        results = estimate_loss()
        elapsed = time.time() - start_time

        print(
            f"Step {step:3d} | "
            f"Train: {results['train']:.4f} | "
            f"Val: {results['val']:.4f} | "
            f"Time: {elapsed:.1f}s",
            flush=True
        )

        history.append({
            "step": step,
            "train_loss": results["train"],
            "val_loss": results["val"]
        })

    if step == max_steps:
        break

    x = get_batch("train")

    output = model(
        input_ids=x,
        labels=x
    )

    optimizer.zero_grad(set_to_none=True)
    output.loss.backward()
    optimizer.step()


print("\nFine-tuning complete!")
print(f"Total time: {time.time() - start_time:.1f}s")

# Save experiment metrics
with open("finetuning_history.json", "w") as f:
    json.dump(history, f, indent=2)

# Save model weights and configuration
model.save_pretrained("finetuned_model")

print("Model saved to finetuned_model/")
