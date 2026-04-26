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
num_classes = 10

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

        self.label_emb = nn.Embedding(num_classes, num_classes)

        self.model = nn.Sequential(
            nn.Linear(latent_dim + num_classes, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, 784),
            nn.Tanh()
        )

    def forward(self, noise, labels):
        label_input = self.label_emb(labels)
        x = torch.cat([noise, label_input], dim=1)
        img = self.model(x)
        return img.view(-1, 1, 28, 28)

# -----------------------------
# Discriminator
# -----------------------------
class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()

        self.label_emb = nn.Embedding(num_classes, num_classes)

        self.model = nn.Sequential(
            nn.Linear(784 + num_classes, 512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )

    def forward(self, img, labels):
        img_flat = img.view(-1, 784)
        label_input = self.label_emb(labels)
        x = torch.cat([img_flat, label_input], dim=1)
        return self.model(x)

# -----------------------------
# Initialize Models
# -----------------------------
G = Generator().to(device)
D = Discriminator().to(device)

criterion = nn.BCELoss()

optimizer_G = optim.Adam(G.parameters(), lr=lr)
optimizer_D = optim.Adam(D.parameters(), lr=lr)

# -----------------------------
# Training
# -----------------------------
print("\nTraining Started...\n")

for epoch in range(epochs):

    for real_imgs, real_labels in train_loader:

        bs = real_imgs.size(0)

        real_imgs = real_imgs.to(device)
        real_labels = real_labels.to(device)

        valid = torch.ones(bs, 1).to(device)
        fake = torch.zeros(bs, 1).to(device)

        # -----------------
        # Train Generator
        # -----------------
        noise = torch.randn(bs, latent_dim).to(device)
        gen_labels = torch.randint(0, num_classes, (bs,)).to(device)

        gen_imgs = G(noise, gen_labels)

        g_loss = criterion(D(gen_imgs, gen_labels), valid)

        optimizer_G.zero_grad()
        g_loss.backward()
        optimizer_G.step()

        # -----------------
        # Train Discriminator
        # -----------------
        real_loss = criterion(D(real_imgs, real_labels), valid)
        fake_loss = criterion(D(gen_imgs.detach(), gen_labels), fake)

        d_loss = (real_loss + fake_loss) / 2

        optimizer_D.zero_grad()
        d_loss.backward()
        optimizer_D.step()

    print(f"Epoch [{epoch+1}/{epochs}] D Loss: {d_loss.item():.4f} | G Loss: {g_loss.item():.4f}")

print("\nTraining Completed!")

# -----------------------------
# Generate Digits 0 to 9
# -----------------------------
with torch.no_grad():
    noise = torch.randn(10, latent_dim).to(device)
    labels = torch.arange(0,10).to(device)

    generated_imgs = G(noise, labels).cpu()

# -----------------------------
# Display Output
# -----------------------------
fig, axes = plt.subplots(1, 10, figsize=(15,3))

for i in range(10):
    axes[i].imshow(generated_imgs[i].squeeze(), cmap="gray")
    axes[i].axis("off")
    axes[i].set_title(str(i))

plt.suptitle("Generated Digits by Labels", fontsize=16)
plt.tight_layout()
plt.show()
