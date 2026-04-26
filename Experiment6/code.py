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
latent_dim = 20

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
# VAE Model
# -----------------------------
class VAE(nn.Module):
    def __init__(self):
        super().__init__()

        self.fc1 = nn.Linear(784, 400)
        self.fc_mu = nn.Linear(400, latent_dim)
        self.fc_logvar = nn.Linear(400, latent_dim)

        self.fc2 = nn.Linear(latent_dim, 400)
        self.fc3 = nn.Linear(400, 784)

    def encode(self, x):
        h = torch.relu(self.fc1(x))
        return self.fc_mu(h), self.fc_logvar(h)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        h = torch.relu(self.fc2(z))
        return torch.sigmoid(self.fc3(h))

    def forward(self, x):
        x = x.view(-1, 784)
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z)
        return recon, mu, logvar

model = VAE().to(device)

# -----------------------------
# Loss Function
# -----------------------------
def loss_function(recon_x, x, mu, logvar):
    BCE = nn.functional.binary_cross_entropy(
        recon_x, x.view(-1, 784), reduction='sum'
    )

    KLD = -0.5 * torch.sum(
        1 + logvar - mu.pow(2) - logvar.exp()
    )

    return BCE + KLD

optimizer = optim.Adam(model.parameters(), lr=lr)

# -----------------------------
# Training
# -----------------------------
print("\nTraining Started...\n")

for epoch in range(epochs):

    total_loss = 0

    for images, _ in train_loader:

        images = images.to(device)

        recon, mu, logvar = model(images)

        loss = loss_function(recon, images, mu, logvar)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(f"Epoch [{epoch+1}/{epochs}] Loss: {total_loss/len(train_loader.dataset):.4f}")

print("\nTraining Completed!")

# -----------------------------
# Reconstruction Output
# -----------------------------
model.eval()

with torch.no_grad():

    sample_images, _ = next(iter(train_loader))
    sample_images = sample_images[:10].to(device)

    recon, _, _ = model(sample_images)

# -----------------------------
# Display Results
# -----------------------------
fig, axes = plt.subplots(2, 10, figsize=(15,4))

for i in range(10):

    # Original
    axes[0, i].imshow(sample_images[i].cpu().view(28,28), cmap="gray")
    axes[0, i].axis("off")
    axes[0, i].set_title("Original")

    # Reconstructed
    axes[1, i].imshow(recon[i].cpu().view(28,28), cmap="gray")
    axes[1, i].axis("off")
    axes[1, i].set_title("Output")

plt.tight_layout()
plt.show()

# -----------------------------
# Generate New Digits
# -----------------------------
with torch.no_grad():

    z = torch.randn(10, latent_dim).to(device)
    generated = model.decode(z).cpu()

fig, axes = plt.subplots(1,10, figsize=(15,2))

for i in range(10):
    axes[i].imshow(generated[i].view(28,28), cmap="gray")
    axes[i].axis("off")

plt.suptitle("Generated New Digits", fontsize=16)
plt.tight_layout()
plt.show()
