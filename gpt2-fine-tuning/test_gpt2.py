
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# Use Apple's GPU if available
device = "mps" if torch.backends.mps.is_available() else "cpu"

model_name = "distilbert/distilgpt2"

print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(model_name)

print("Loading pretrained model...")
model = AutoModelForCausalLM.from_pretrained(model_name)
model = model.to(device)
model.eval()

# Count model parameters
num_params = sum(p.numel() for p in model.parameters())

print("\n--- Pretrained GPT-2 Model ---")
print("Device:", device)
print(f"Parameters: {num_params:,}")

# Prepare an example input
prompt = "ROMEO: O, my love"

inputs = tokenizer(prompt, return_tensors="pt").to(device)

# Generate text before fine-tuning
with torch.no_grad():
    outputs = model.generate(
    **inputs,
    max_new_tokens=100,
    do_sample=True,
    temperature=0.9,
    top_k=50,
    top_p=0.92,
    repetition_penalty=1.2,
    no_repeat_ngram_size=3,
    pad_token_id=tokenizer.eos_token_id
)

print("\n--- Before Fine-Tuning ---")
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
