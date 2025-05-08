# Autoencoders: Undercomplete autoencoders, regularized autoencoders, sparse autoencoders, denoising autoencoders, representational power, layer, size, and depth of autoencoders, stochastic encoders and decoders

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
import matplotlib.pyplot as plt

# Set random seed for reproducibility
torch.manual_seed(42)

# --------------------------------------
# MNIST Dataset (For Autoencoder)
# --------------------------------------

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

train_data = datasets.MNIST(root='./data', train=True, transform=transform, download=True)
test_data = datasets.MNIST(root='./data', train=False, transform=transform, download=True)

train_loader = torch.utils.data.DataLoader(train_data, batch_size=128, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_data, batch_size=128, shuffle=False)

# --------------------------------------
# Base Autoencoder Architecture (Undercomplete Autoencoder)
# --------------------------------------

class Autoencoder(nn.Module):
    def __init__(self, encoding_dim=32):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(28*28, 128),
            nn.ReLU(),
            nn.Linear(128, encoding_dim)
        )
        self.decoder = nn.Sequential(
            nn.Linear(encoding_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 28*28),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = x.view(-1, 28*28)
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded.view(-1, 1, 28, 28)

# --------------------------------------
# Regularized Autoencoder
# --------------------------------------

class RegularizedAutoencoder(Autoencoder):
    def __init__(self, encoding_dim=32, l2_lambda=0.001):
        super(RegularizedAutoencoder, self).__init__(encoding_dim)
        self.l2_lambda = l2_lambda

    def loss_function(self, output, target):
        loss = nn.MSELoss()(output, target)
        # L2 regularization
        l2_reg = sum(p.pow(2.0).sum() for p in self.parameters())
        return loss + self.l2_lambda * l2_reg

# --------------------------------------
# Sparse Autoencoder
# --------------------------------------

class SparseAutoencoder(Autoencoder):
    def __init__(self, encoding_dim=32, sparsity_lambda=0.1):
        super(SparseAutoencoder, self).__init__(encoding_dim)
        self.sparsity_lambda = sparsity_lambda

    def loss_function(self, output, target):
        loss = nn.MSELoss()(output, target)
        # Sparse regularization
        sparsity_penalty = self.sparsity_lambda * torch.sum(torch.abs(self.encoder[0].weight))
        return loss + sparsity_penalty

# --------------------------------------
# Denoising Autoencoder
# --------------------------------------

class DenoisingAutoencoder(Autoencoder):
    def __init__(self, encoding_dim=32, noise_factor=0.5):
        super(DenoisingAutoencoder, self).__init__(encoding_dim)
        self.noise_factor = noise_factor

    def add_noise(self, x):
        return x + self.noise_factor * torch.randn_like(x)

    def forward(self, x):
        x_noisy = self.add_noise(x)
        return super().forward(x_noisy)

# --------------------------------------
# Stochastic Encoder/Decoder Autoencoder
# --------------------------------------

class StochasticAutoencoder(Autoencoder):
    def __init__(self, encoding_dim=32, noise_std=0.1):
        super(StochasticAutoencoder, self).__init__(encoding_dim)
        self.noise_std = noise_std

    def forward(self, x):
        x = x.view(-1, 28*28)
        encoded = self.encoder(x)
        encoded = encoded + self.noise_std * torch.randn_like(encoded)
        decoded = self.decoder(encoded)
        return decoded.view(-1, 1, 28, 28)

# --------------------------------------
# Training Function
# --------------------------------------

def train_autoencoder(model, train_loader, epochs=10, lr=0.001, device='cpu'):
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for data, _ in train_loader:
            data = data.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = model.loss_function(output, data)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch [{epoch+1}/{epochs}] - Loss: {total_loss / len(train_loader):.4f}")

# Evaluate Autoencoder
def evaluate_autoencoder(model, test_loader, device='cpu'):
    model.to(device)
    model.eval()
    with torch.no_grad():
        total_loss = 0
        for data, _ in test_loader:
            data = data.to(device)
            output = model(data)
            loss = nn.MSELoss()(output, data)
            total_loss += loss.item()
        print(f"Test Loss: {total_loss / len(test_loader):.4f}")

# Visualize Reconstructed Images
def visualize_reconstruction(model, test_loader, device='cpu'):
    model.to(device)
    model.eval()
    with torch.no_grad():
        data, _ = next(iter(test_loader))
        data = data.to(device)
        reconstructed = model(data)
        data = data.cpu().numpy()
        reconstructed = reconstructed.cpu().numpy()

    fig, axes = plt.subplots(2, 8, figsize=(15, 4))
    for i in range(8):
        axes[0, i].imshow(data[i][0], cmap='gray')
        axes[1, i].imshow(reconstructed[i][0], cmap='gray')
        axes[0, i].axis('off')
        axes[1, i].axis('off')
    plt.show()

# --------------------------------------
# Example: Train and Visualize Different Autoencoders
# --------------------------------------

# Base Autoencoder
print("Training Base Autoencoder...")
base_autoencoder = Autoencoder(encoding_dim=64)
train_autoencoder(base_autoencoder, train_loader, epochs=5, lr=0.001)
evaluate_autoencoder(base_autoencoder, test_loader)
visualize_reconstruction(base_autoencoder, test_loader)

# Regularized Autoencoder
print("Training Regularized Autoencoder...")
regularized_autoencoder = RegularizedAutoencoder(encoding_dim=64, l2_lambda=0.001)
train_autoencoder(regularized_autoencoder, train_loader, epochs=5, lr=0.001)
evaluate_autoencoder(regularized_autoencoder, test_loader)
visualize_reconstruction(regularized_autoencoder, test_loader)

# Sparse Autoencoder
print("Training Sparse Autoencoder...")
sparse_autoencoder = SparseAutoencoder(encoding_dim=64, sparsity_lambda=0.1)
train_autoencoder(sparse_autoencoder, train_loader, epochs=5, lr=0.001)
evaluate_autoencoder(sparse_autoencoder, test_loader)
visualize_reconstruction(sparse_autoencoder, test_loader)

# Denoising Autoencoder
print("Training Denoising Autoencoder...")
denoising_autoencoder = DenoisingAutoencoder(encoding_dim=64, noise_factor=0.5)
train_autoencoder(denoising_autoencoder, train_loader, epochs=5, lr=0.001)
evaluate_autoencoder(denoising_autoencoder, test_loader)
visualize_reconstruction(denoising_autoencoder, test_loader)

# Stochastic Autoencoder
print("Training Stochastic Autoencoder...")
stochastic_autoencoder = StochasticAutoencoder(encoding_dim=64, noise_std=0.1)
train_autoencoder(stochastic_autoencoder, train_loader, epochs=5, lr=0.001)
evaluate_autoencoder(stochastic_autoencoder, test_loader)
visualize_reconstruction(stochastic_autoencoder, test_loader)
