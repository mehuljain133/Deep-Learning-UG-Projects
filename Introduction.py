# Introduction: Historical context and motivation for deep learning; basic supervised classification task, optimizing logistic classifier using gradient descent, stochastic gradient descent, momentum, and adaptive sub-gradient method.

import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# --------------------------------------------
# Historical context (in comments)
# --------------------------------------------
# In the early days of AI, symbolic methods ruled.
# Later, shallow neural networks appeared, but limited data and compute held them back.
# In the 2000s, with more data, GPU power, and techniques like backpropagation, deep learning emerged.
# Today, we revisit a simple logistic regression example (a building block of deep learning).
# --------------------------------------------

# Generate synthetic binary classification data
X, y = make_classification(n_samples=1000, n_features=2, n_classes=2, n_redundant=0, n_informative=2, random_state=42)
X = StandardScaler().fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Convert to PyTorch tensors
X_train = torch.tensor(X_train, dtype=torch.float32)
y_train = torch.tensor(y_train.reshape(-1, 1), dtype=torch.float32)
X_test = torch.tensor(X_test, dtype=torch.float32)
y_test = torch.tensor(y_test.reshape(-1, 1), dtype=torch.float32)

# Define a logistic regression model
class LogisticRegressionModel(nn.Module):
    def __init__(self, input_dim):
        super(LogisticRegressionModel, self).__init__()
        self.linear = nn.Linear(input_dim, 1)

    def forward(self, x):
        return torch.sigmoid(self.linear(x))

# Training loop
def train_model(optimizer_name='sgd', lr=0.01, momentum=0.9, epochs=100):
    model = LogisticRegressionModel(input_dim=2)
    criterion = nn.BCELoss()

    # Choose optimizer
    if optimizer_name == 'gd':
        optimizer = optim.SGD(model.parameters(), lr=lr)
    elif optimizer_name == 'sgd':
        optimizer = optim.SGD(model.parameters(), lr=lr)
    elif optimizer_name == 'momentum':
        optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum)
    elif optimizer_name == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=lr)
    else:
        raise ValueError("Unsupported optimizer")

    losses = []
    for epoch in range(epochs):
        model.train()
        # Full-batch GD vs mini-batch/SGD
        if optimizer_name == 'gd':
            optimizer.zero_grad()
            outputs = model(X_train)
            loss = criterion(outputs, y_train)
            loss.backward()
            optimizer.step()
        else:  # SGD-style
            for i in range(X_train.shape[0]):
                optimizer.zero_grad()
                output = model(X_train[i].unsqueeze(0))
                loss = criterion(output, y_train[i].unsqueeze(0))
                loss.backward()
                optimizer.step()

        with torch.no_grad():
            outputs = model(X_train)
            loss = criterion(outputs, y_train)
            losses.append(loss.item())

        if epoch % 10 == 0:
            print(f'Epoch {epoch}, Loss: {loss.item():.4f}')

    return model, losses

# Train and plot for each optimizer
optimizers = ['gd', 'sgd', 'momentum', 'adam']
all_losses = {}

for opt in optimizers:
    print(f"\nTraining with {opt.upper()} optimizer")
    model, losses = train_model(optimizer_name=opt)
    all_losses[opt] = losses

# Plotting losses
plt.figure(figsize=(10, 6))
for opt, losses in all_losses.items():
    plt.plot(losses, label=opt.upper())
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Training Loss per Optimizer')
plt.legend()
plt.grid(True)
plt.show()
