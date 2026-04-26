!pip install -q transformers datasets accelerate torch

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
    pipeline
)
from datasets import Dataset
import torch

# -----------------------------
# Device
# -----------------------------
device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using Device:", device)

# -----------------------------
# Small Custom Dataset
# -----------------------------
texts = [
    "Python is a powerful programming language for AI.",
    "Machine learning helps computers learn from data.",
    "Deep learning uses neural networks for prediction.",
    "Artificial intelligence is transforming industries.",
    "Data science combines statistics and computing.",
    "Natural language processing works with text data.",
    "Generative AI can create text and images.",
    "Python libraries include NumPy, Pandas and PyTorch."
] * 20   # repeat for training

dataset = Dataset.from_dict({"text": texts})

# -----------------------------
# Load Model
# -----------------------------
model_name = "distilgpt2"

tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(model_name)

# -----------------------------
# Tokenization
# -----------------------------
def tokenize_function(example):
    return tokenizer(
        example["text"],
        truncation=True,
        padding="max_length",
        max_length=64
    )

tokenized_dataset = dataset.map(tokenize_function, batched=True)

# -----------------------------
# Training Arguments
# -----------------------------
training_args = TrainingArguments(
    output_dir="./results",
    overwrite_output_dir=True,
    num_train_epochs=1,
    per_device_train_batch_size=2,
    save_steps=50,
    save_total_limit=1,
    logging_steps=10,
    report_to="none"
)

# -----------------------------
# Data Collator
# -----------------------------
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False
)

# -----------------------------
# Trainer
# -----------------------------
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    data_collator=data_collator
)

# -----------------------------
# Start Training
# -----------------------------
print("\nTraining Started...\n")

trainer.train()

print("\nFine-Tuning Completed!")

# -----------------------------
# Generate Output
# -----------------------------
generator = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    device=0 if torch.cuda.is_available() else -1
)

prompt = "Artificial intelligence"

result = generator(
    prompt,
    max_new_tokens=60,
    do_sample=True,
    temperature=0.8,
    top_p=0.95
)

# -----------------------------
# Final Output
# -----------------------------
print("\n===================================")
print(" Fine-Tuned GPT-2 Generated Output")
print("===================================\n")

print(result[0]["generated_text"])
