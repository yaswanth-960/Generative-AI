!pip install -q torch torchvision matplotlib

import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# -----------------------------
# Device
# -----------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using Device:", device)

# -----------------------------
# Hyperparameters
# -----------------------------
batch_size = 128
epochs = 5
lr = 0.001
latent_dim = 32

# -----------------------------
# Load MNIST Dataset
# -----------------------------
transform = transforms.ToTensor()

train_dataset = datasets.MNIST(
    root="./data",
    train=True,
    download=True,
    transform=transform
)

train_loader = DataLoader(
    train_dataset,
    batch_size=batch_size,
    shuffle=True
)

# -----------------------------
# Autoencoder Model
# -----------------------------
class Autoencoder(nn.Module):
    def __init__(self):
        super().__init__()

        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(784, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, latent_dim)
        )

        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, 784),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = x.view(-1, 784)
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

model = Autoencoder().to(device)

criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=lr)

# -----------------------------
# Training
# -----------------------------
print("\nTraining Started...\n")

for epoch in range(epochs):
    total_loss = 0

    for images, _ in train_loader:
        images = images.to(device)

        outputs = model(images)
        loss = criterion(outputs, images.view(-1, 784))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(f"Epoch [{epoch+1}/{epochs}] Loss: {total_loss/len(train_loader):.6f}")

print("\nTraining Completed!")

# -----------------------------
# Test Output
# -----------------------------
model.eval()

with torch.no_grad():
    sample_images, _ = next(iter(train_loader))
    sample_images = sample_images[:10].to(device)

    reconstructed = model(sample_images)

# -----------------------------
# Display Results
# -----------------------------
fig, axes = plt.subplots(2, 10, figsize=(15,4))

for i in range(10):

    # Original Images
    axes[0, i].imshow(sample_images[i].cpu().view(28,28), cmap="gray")
    axes[0, i].axis("off")
    axes[0, i].set_title("Original")

    # Reconstructed Images
    axes[1, i].imshow(reconstructed[i].cpu().view(28,28), cmap="gray")
    axes[1, i].axis("off")
    axes[1, i].set_title("Output")

plt.tight_layout()
plt.show()
