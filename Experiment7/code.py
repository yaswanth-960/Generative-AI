!pip install -q torch torchvision matplotlib pillow

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torchvision import datasets
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

# -----------------------------
# Device
# -----------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using Device:", device)

# -----------------------------
# Hyperparameters
# -----------------------------
batch_size = 64
epochs = 5
lr = 0.0002

# -----------------------------
# Dataset (MNIST used as demo)
# Real Image = MNIST
# Sketch Image = Edge-like inverse
# -----------------------------
transform = transforms.ToTensor()

dataset = datasets.MNIST(
    root="./data",
    train=True,
    download=True,
    transform=transform
)

loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# -----------------------------
# Generator (Simple CNN)
# -----------------------------
class Generator(nn.Module):
    def __init__(self):
        super().__init__()

        self.model = nn.Sequential(
            nn.Conv2d(1, 64, 3, padding=1),
            nn.ReLU(),

            nn.Conv2d(64, 64, 3, padding=1),
            nn.ReLU(),

            nn.Conv2d(64, 1, 3, padding=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.model(x)

# -----------------------------
# Discriminator
# -----------------------------
class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()

        self.model = nn.Sequential(
            nn.Flatten(),
            nn.Linear(28*28, 256),
            nn.LeakyReLU(0.2),

            nn.Linear(256, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.model(x)

G = Generator().to(device)
D = Discriminator().to(device)

criterion = nn.BCELoss()
l1_loss = nn.L1Loss()

opt_G = optim.Adam(G.parameters(), lr=lr)
opt_D = optim.Adam(D.parameters(), lr=lr)

# -----------------------------
# Training
# -----------------------------
print("\nTraining Started...\n")

for epoch in range(epochs):

    for real_imgs, _ in loader:

        real_imgs = real_imgs.to(device)

        # Create sketch images
        sketch_imgs = 1 - real_imgs

        valid = torch.ones(real_imgs.size(0),1).to(device)
        fake = torch.zeros(real_imgs.size(0),1).to(device)

        # -----------------
        # Train Generator
        # -----------------
        fake_imgs = G(sketch_imgs)

        adv_loss = criterion(D(fake_imgs), valid)
        pixel_loss = l1_loss(fake_imgs, real_imgs)

        g_loss = adv_loss + 100 * pixel_loss

        opt_G.zero_grad()
        g_loss.backward()
        opt_G.step()

        # -----------------
        # Train Discriminator
        # -----------------
        real_loss = criterion(D(real_imgs), valid)
        fake_loss = criterion(D(fake_imgs.detach()), fake)

        d_loss = (real_loss + fake_loss) / 2

        opt_D.zero_grad()
        d_loss.backward()
        opt_D.step()

    print(f"Epoch [{epoch+1}/{epochs}] D Loss: {d_loss.item():.4f} | G Loss: {g_loss.item():.4f}")

print("\nTraining Completed!")

# -----------------------------
# Output Visualization
# -----------------------------
G.eval()

with torch.no_grad():

    sample_imgs, _ = next(iter(loader))
    sample_imgs = sample_imgs[:5].to(device)

    sketch_imgs = 1 - sample_imgs
    generated_imgs = G(sketch_imgs)

# -----------------------------
# Display Results
# -----------------------------
fig, axes = plt.subplots(3,5, figsize=(10,6))

for i in range(5):

    # Sketch
    axes[0,i].imshow(sketch_imgs[i].cpu().squeeze(), cmap="gray")
    axes[0,i].axis("off")
    axes[0,i].set_title("Sketch")

    # Generated
    axes[1,i].imshow(generated_imgs[i].cpu().squeeze(), cmap="gray")
    axes[1,i].axis("off")
    axes[1,i].set_title("Output")

    # Real
    axes[2,i].imshow(sample_imgs[i].cpu().squeeze(), cmap="gray")
    axes[2,i].axis("off")
    axes[2,i].set_title("Real")

plt.tight_layout()
plt.show())
