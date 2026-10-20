
import torch
from transformers import AutoTokenizer

model_name = "distilbert/distilgpt2"

# Load pretrained GPT-2 tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name)

with open("input.txt", "r", encoding="utf-8") as f:
    text = f.read()

# Tokenize the entire Shakespeare corpus
tokens = tokenizer(
    text,
    add_special_tokens=False,
    return_tensors=None
)["input_ids"]

data = torch.tensor(tokens, dtype=torch.long)

# Split into 90% training and 10% validation
n = int(0.9 * len(data))
train_data = data[:n]
val_data = data[n:]

# Save tokenized data
torch.save(train_data, "train_tokens.pt")
torch.save(val_data, "val_tokens.pt")

print("\n--- Shakespeare Dataset ---")
print("Characters:", len(text))
print("GPT-2 tokens:", len(data))
print("Training tokens:", len(train_data))
print("Validation tokens:", len(val_data))
print("Tokenizer vocabulary:", tokenizer.vocab_size)
print("Example token IDs:", tokens[:20])
print("Decoded example:", tokenizer.decode(tokens[:20]))
