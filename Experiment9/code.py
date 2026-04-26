!pip install -q transformers pillow matplotlib torch

from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image
import matplotlib.pyplot as plt
from google.colab import files
import torch

# -----------------------------
# Device
# -----------------------------
device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using Device:", device)

# -----------------------------
# Load BLIP Model
# -----------------------------
print("\nLoading BLIP Model...\n")

processor = BlipProcessor.from_pretrained(
    "Salesforce/blip-image-captioning-base"
)

model = BlipForConditionalGeneration.from_pretrained(
    "Salesforce/blip-image-captioning-base"
).to(device)

print("Model Loaded Successfully!\n")

# -----------------------------
# Upload Image
# -----------------------------
print("Upload an Image File...\n")

uploaded = files.upload()

image_path = list(uploaded.keys())[0]

image = Image.open(image_path).convert("RGB")

# -----------------------------
# Generate Caption
# -----------------------------
inputs = processor(images=image, return_tensors="pt").to(device)

output = model.generate(**inputs, max_new_tokens=30)

caption = processor.decode(output[0], skip_special_tokens=True)

# -----------------------------
# Display Output
# -----------------------------
plt.figure(figsize=(8,8))
plt.imshow(image)
plt.axis("off")
plt.title("Uploaded Image", fontsize=16)
plt.show()

print("\n===================================")
print("      AI GENERATED CAPTION")
print("===================================\n")

print(caption)
