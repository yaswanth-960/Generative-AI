!pip install -q transformers torch

from transformers import pipeline, set_seed
import torch

# -----------------------------
# Device
# -----------------------------
device = 0 if torch.cuda.is_available() else -1
print("Using Device:", "GPU" if device == 0 else "CPU")

# -----------------------------
# Load GPT-2 Model
# -----------------------------
print("\nLoading GPT-2 Model...\n")

generator = pipeline(
    "text-generation",
    model="gpt2",
    device=device
)

set_seed(42)

print("Model Loaded Successfully!\n")

# -----------------------------
# User Prompt
# -----------------------------
prompt = input("Enter Prompt: ")

# -----------------------------
# Generate Text
# -----------------------------
result = generator(
    prompt,
    max_new_tokens=120,
    do_sample=True,
    temperature=0.8,
    top_k=50,
    top_p=0.95,
    repetition_penalty=1.2,
    pad_token_id=50256
)

# -----------------------------
# Output
# -----------------------------
print("\n====================================")
print("      AI GENERATED TEXT OUTPUT")
print("====================================\n")

print(result[0]["generated_text"])
