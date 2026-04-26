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
lr = 0.0002
latent_dim = 100

# -----------------------------
# Load MNIST Dataset
# -----------------------------
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

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
# Generator
# -----------------------------
class Generator(nn.Module):
    def __init__(self):
        super().__init__()

        self.model = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, 784),
            nn.Tanh()
        )

    def forward(self, z):
        img = self.model(z)
        return img.view(-1, 1, 28, 28)

# -----------------------------
# Discriminator
# -----------------------------
class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()

        self.model = nn.Sequential(
            nn.Linear(784, 512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )

    def forward(self, img):
        img_flat = img.view(-1, 784)
        return self.model(img_flat)

# Initialize Models
G = Generator().to(device)
D = Discriminator().to(device)

# Loss
criterion = nn.BCELoss()

# Optimizers
optimizer_G = optim.Adam(G.parameters(), lr=lr)
optimizer_D = optim.Adam(D.parameters(), lr=lr)

# -----------------------------
# Training
# -----------------------------
print("\nTraining Started...\n")

for epoch in range(epochs):

    for real_imgs, _ in train_loader:

        batch_size_now = real_imgs.size(0)

        real_imgs = real_imgs.to(device)

        real_labels = torch.ones(batch_size_now, 1).to(device)
        fake_labels = torch.zeros(batch_size_now, 1).to(device)

        # -----------------
        # Train Generator
        # -----------------
        z = torch.randn(batch_size_now, latent_dim).to(device)
        fake_imgs = G(z)

        g_loss = criterion(D(fake_imgs), real_labels)

        optimizer_G.zero_grad()
        g_loss.backward()
        optimizer_G.step()

        # -----------------
        # Train Discriminator
        # -----------------
        real_loss = criterion(D(real_imgs), real_labels)
        fake_loss = criterion(D(fake_imgs.detach()), fake_labels)

        d_loss = (real_loss + fake_loss) / 2

        optimizer_D.zero_grad()
        d_loss.backward()
        optimizer_D.step()

    print(f"Epoch [{epoch+1}/{epochs}] D Loss: {d_loss.item():.4f} | G Loss: {g_loss.item():.4f}")

print("\nTraining Completed!")

# -----------------------------
# Generate Final Images
# -----------------------------
with torch.no_grad():
    z = torch.randn(16, latent_dim).to(device)
    generated_imgs = G(z).cpu()

# -----------------------------
# Display Output
# -----------------------------
fig, axes = plt.subplots(4, 4, figsize=(8,8))

count = 0
for i in range(4):
    for j in range(4):
        axes[i, j].imshow(generated_imgs[count].squeeze(), cmap="gray")
        axes[i, j].axis("off")
        count += 1

plt.suptitle("Generated Digit Images", fontsize=16)
plt.tight_layout()
plt.show()
