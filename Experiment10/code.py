!pip install -q detoxify torch pandas

from detoxify import Detoxify
import pandas as pd
import matplotlib.pyplot as plt

# -----------------------------
# Load Model
# -----------------------------
print("Loading Ethical AI Model...\n")

model = Detoxify('original')

print("Model Loaded Successfully!\n")

# -----------------------------
# User Input
# -----------------------------
text = input("Enter Text to Analyze: ")

# Example:
# You are stupid

# -----------------------------
# Predict Scores
# -----------------------------
result = model.predict(text)

# -----------------------------
# Display Scores
# -----------------------------
df = pd.DataFrame(result.items(), columns=["Category", "Score"])

print("\n===================================")
print("        TOXICITY ANALYSIS")
print("===================================\n")

print(df)

# -----------------------------
# Graph Output
# -----------------------------
plt.figure(figsize=(10,5))
plt.bar(df["Category"], df["Score"])
plt.xticks(rotation=45)
plt.title("Toxicity Scores")
plt.ylabel("Probability")
plt.show()

# -----------------------------
# Final Verdict
# -----------------------------
if result["toxicity"] > 0.5:
    print("\n⚠️ Final Result: Toxic Content Detected")
else:
    print("\n✅ Final Result: Safe Text")
