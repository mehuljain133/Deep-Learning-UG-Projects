# Neural Networks: Feedforward neural networks, deep networks, regularizing a deep network, model exploration, and hyper parameter tuning

pip install torch torchvision matplotlib scikit-learn

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from sklearn.model_selection import ParameterGrid
import matplotlib.pyplot as plt
import time

# Set random seeds for reproducibility
torch.manual_seed(0)

# -------------------------
# Load MNIST Data
# -------------------------
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
test_dataset  = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader  = torch.utils.data.DataLoader(test_dataset, batch_size=1000, shuffle=False)

# -------------------------
# Define Flexible Deep Network
# -------------------------
class DeepNet(nn.Module):
    def __init__(self, input_size, hidden_layers, output_size, dropout=0.0):
        super(DeepNet, self).__init__()
        layers = []
        all_layers = [input_size] + hidden_layers + [output_size]
        for i in range(len(all_layers)-2):
            layers.append(nn.Linear(all_layers[i], all_layers[i+1]))
            layers.append(nn.ReLU())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
        layers.append(nn.Linear(all_layers[-2], all_layers[-1]))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)

# -------------------------
# Training & Evaluation
# -------------------------
def train(model, optimizer, criterion, epochs=5):
    model.train()
    for epoch in range(epochs):
        for data, target in train_loader:
            data = data.view(data.size(0), -1)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

def evaluate(model):
    model.eval()
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data = data.view(data.size(0), -1)
            output = model(data)
            pred = output.argmax(dim=1)
            correct += pred.eq(target).sum().item()
    accuracy = 100. * correct / len(test_loader.dataset)
    return accuracy

# -------------------------
# Hyperparameter Grid Search
# -------------------------
param_grid = {
    'hidden_layers': [[128, 64], [256, 128, 64], [512, 256, 128]],
    'dropout': [0.0, 0.2, 0.5],
    'lr': [0.01, 0.001],
    'weight_decay': [0, 1e-4]
}

best_acc = 0
best_model = None
results = []

print("Starting grid search...\n")

for params in ParameterGrid(param_grid):
    print(f"Testing config: {params}")
    model = DeepNet(input_size=28*28, hidden_layers=params['hidden_layers'],
                    output_size=10, dropout=params['dropout'])
    optimizer = optim.Adam(model.parameters(), lr=params['lr'], weight_decay=params['weight_decay'])
    criterion = nn.CrossEntropyLoss()

    start_time = time.time()
    train(model, optimizer, criterion, epochs=5)
    acc = evaluate(model)
    elapsed = time.time() - start_time

    print(f" → Accuracy: {acc:.2f}%, Time: {elapsed:.2f}s\n")
    results.append((params, acc))

    if acc > best_acc:
        best_acc = acc
        best_model = model

# -------------------------
# Results Summary
# -------------------------
results.sort(key=lambda x: x[1], reverse=True)
print("Top 3 models:")
for i in range(3):
    print(f"Rank {i+1}: Acc={results[i][1]:.2f}% | Config={results[i][0]}")

# -------------------------
# Final Evaluation
# -------------------------
final_acc = evaluate(best_model)
print(f"\nBest model final test accuracy: {final_acc:.2f}%")
