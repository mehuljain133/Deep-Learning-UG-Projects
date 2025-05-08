# Deep Learning PhD-Level Comprehensive Implementation
# Author: OpenAI
# Note: This is a modular, illustrative framework combining all major topics

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.datasets import make_classification, make_regression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Utility function: Sigmoid and cost

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def compute_cost(X, y, weights):
    m = len(y)
    h = sigmoid(np.dot(X, weights))
    cost = -(1/m) * (np.dot(y, np.log(h)) + np.dot((1 - y), np.log(1 - h)))
    return cost

# 1. Logistic Regression (GD and SGD)
def logistic_regression(X, y, method="gd", alpha=0.01, epochs=1000):
    m, n = X.shape
    weights = np.zeros(n)
    cost_history = []

    for epoch in range(epochs):
        if method == "gd":
            h = sigmoid(np.dot(X, weights))
            gradient = np.dot(X.T, (h - y)) / m
            weights -= alpha * gradient
        elif method == "sgd":
            for i in range(m):
                xi = X[i, :].reshape(1, -1)
                yi = y[i]
                hi = sigmoid(np.dot(xi, weights))
                gradient = np.dot(xi.T, (hi - yi))
                weights -= alpha * gradient.flatten()
        cost = compute_cost(X, y, weights)
        cost_history.append(cost)

    return weights, cost_history

# 2. Momentum & Adagrad on Logistic Regression

def logistic_with_optimizer(X, y, method="momentum", alpha=0.01, beta=0.9, epochs=1000):
    m, n = X.shape
    weights = np.zeros(n)
    v = np.zeros(n)
    G = np.zeros(n)
    eps = 1e-8
    cost_history = []

    for epoch in range(epochs):
        h = sigmoid(np.dot(X, weights))
        gradient = np.dot(X.T, (h - y)) / m

        if method == "momentum":
            v = beta * v + (1 - beta) * gradient
            weights -= alpha * v
        elif method == "adagrad":
            G += gradient**2
            adjusted_gradient = alpha * gradient / (np.sqrt(G) + eps)
            weights -= adjusted_gradient

        cost = compute_cost(X, y, weights)
        cost_history.append(cost)

    return weights, cost_history

# 3. Perceptron Learning for Boolean Functions

def perceptron(X, y, epochs=10):
    X = np.insert(X, 0, 1, axis=1)  # bias
    weights = np.zeros(X.shape[1])
    for _ in range(epochs):
        for xi, target in zip(X, y):
            output = 1 if np.dot(xi, weights) > 0 else 0
            weights += (target - output) * xi
    return weights

# Boolean function examples
bool_inputs = np.array([[0,0],[0,1],[1,0],[1,1]])
labels = {
    "NOT": np.array([1, 1, 0, 0]),  # dummy 2-input NOT
    "OR":  np.array([0, 1, 1, 1]),
    "AND": np.array([0, 0, 0, 1]),
    "NOR": np.array([1, 0, 0, 0]),
    "NAND":np.array([1, 1, 1, 0]),
}

perceptron_weights = {name: perceptron(bool_inputs, y) for name, y in labels.items()}

# 4. Feedforward Neural Net for Regression and Classification

class FFNN(nn.Module):
    def __init__(self, input_size, output_size, hidden=[64, 32]):
        super().__init__()
        layers = []
        in_size = input_size
        for h in hidden:
            layers.append(nn.Linear(in_size, h))
            layers.append(nn.ReLU())
            in_size = h
        layers.append(nn.Linear(in_size, output_size))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)

# 5. Softmax for Multi-class Classification
class MultiClassNet(nn.Module):
    def __init__(self, input_size, num_classes):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, 100),
            nn.ReLU(),
            nn.Linear(100, num_classes),
            nn.Softmax(dim=1)
        )
    def forward(self, x):
        return self.net(x)

# 6. CNN Architectures

class CNN2D(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 16, 3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.fc = nn.Linear(32 * 7 * 7, 10)

    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)

# 7. RNN, GRU, LSTM

class RNNNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.rnn = nn.RNN(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out, _ = self.rnn(x)
        return self.fc(out[:, -1, :])

class GRUNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.gru = nn.GRU(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out, _ = self.gru(x)
        return self.fc(out[:, -1, :])

class LSTMNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])

# 8. Autoencoders (vanilla, denoising, sparse)

class Autoencoder(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim, input_dim),
            nn.Sigmoid()
        )
    def forward(self, x):
        z = self.encoder(x)
        return self.decoder(z)

# 9. Stochastic Encoder-Decoder
class StochasticEncoderDecoder(nn.Module):
    def __init__(self, input_dim, latent_dim):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, latent_dim),
            nn.Softplus()
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, input_dim),
            nn.Sigmoid()
        )

    def forward(self, x):
        z = self.encoder(x)
        z = z + torch.randn_like(z) * 0.1  # inject noise
        return self.decoder(z)

# Note: Each module should be separately trained with appropriate loss functions and datasets.
# You can expand this code by writing training loops for each model based on your specific datasets.
