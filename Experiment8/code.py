!pip install -q diffusers transformers accelerate torch safetensors matplotlib

import torch
import matplotlib.pyplot as plt
from diffusers import StableDiffusionPipeline

# -----------------------------
# Device
# -----------------------------
device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using Device:", device)

# -----------------------------
# Load Model
# -----------------------------
print("\nLoading Stable Diffusion Model...\n")

pipe = StableDiffusionPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5",
    torch_dtype=torch.float16 if device == "cuda" else torch.float32
)

pipe = pipe.to(device)

print("Model Loaded Successfully!\n")

# -----------------------------
# User Prompt
# -----------------------------
prompt = input("Enter Image Prompt: ")

# Example:
# futuristic city at night with flying cars

# -----------------------------
# Generate Image
# -----------------------------
image = pipe(
    prompt,
    num_inference_steps=25,
    guidance_scale=7.5
).images[0]

# -----------------------------
# Display Output
# -----------------------------
plt.figure(figsize=(8,8))
plt.imshow(image)
plt.axis("off")
plt.title("AI Generated Image", fontsize=16)
plt.show()
