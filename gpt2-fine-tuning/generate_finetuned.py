
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

device = "mps" if torch.backends.mps.is_available() else "cpu"

# Load the tokenizer used during fine-tuning
tokenizer = AutoTokenizer.from_pretrained(
    "distilbert/distilgpt2"
)

# Load your locally fine-tuned weights
model = AutoModelForCausalLM.from_pretrained(
    "./finetuned_model"
).to(device)

model.eval()

prompt = "ROMEO:\n"

inputs = tokenizer(
    prompt,
    return_tensors="pt"
).to(device)

with torch.no_grad():
    outputs = model.generate(
        **inputs,
        max_new_tokens=150,
        do_sample=True,
        temperature=0.9,
        top_k=50,
        top_p=0.92,
        repetition_penalty=1.2,
        no_repeat_ngram_size=3,
        pad_token_id=tokenizer.eos_token_id
    )

print("\n--- Fine-Tuned DistilGPT2 ---")
print(tokenizer.decode(
    outputs[0],
    skip_special_tokens=True
))
