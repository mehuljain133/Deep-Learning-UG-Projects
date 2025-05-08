# Structuring Machine Learning Projects: Orthogonalization, evaluation metrics, train/dev/test distributions, size of the dev and test sets, cleaning up incorrectly labeled data, bias and variance with mismatched data distributions, transfer learning, multi-task learning.

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, Dataset

# Set random seed for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# --------------------------------------
# 1. Orthogonalization (Removing Multicollinearity)
# --------------------------------------

# For this task, orthogonalization often refers to removing highly correlated features in your data.
# Here, we'll assume we have some feature matrix X that we want to orthogonalize.

# Example function for orthogonalizing features in a dataset (using PCA for illustration)
from sklearn.decomposition import PCA

def orthogonalize_features(X, n_components=2):
    pca = PCA(n_components=n_components)
    X_orthogonalized = pca.fit_transform(X)
    return X_orthogonalized

# --------------------------------------
# 2. Dataset Preparation (Train/Dev/Test Split)
# --------------------------------------

# We will use CIFAR-10 dataset and manually split it into train, dev, and test sets.
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # Normalization for CIFAR-10
])

# Load the CIFAR-10 dataset
train_data = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
test_data = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

# Split the training data into train and dev sets
train_data, dev_data = torch.utils.data.random_split(train_data, [40000, 10000])

train_loader = DataLoader(train_data, batch_size=64, shuffle=True)
dev_loader = DataLoader(dev_data, batch_size=64, shuffle=False)
test_loader = DataLoader(test_data, batch_size=64, shuffle=False)

# --------------------------------------
# 3. Evaluation Metrics
# --------------------------------------

def evaluate_model(model, loader, device='cpu'):
    model.eval()
    y_true = []
    y_pred = []
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            y_true.extend(labels.cpu().numpy())
            y_pred.extend(predicted.cpu().numpy())

    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average='weighted')
    recall = recall_score(y_true, y_pred, average='weighted')
    f1 = f1_score(y_true, y_pred, average='weighted')

    print(f'Accuracy: {accuracy:.4f}')
    print(f'Precision: {precision:.4f}')
    print(f'Recall: {recall:.4f}')
    print(f'F1 Score: {f1:.4f}')

# --------------------------------------
# 4. Cleaning Incorrectly Labeled Data
# --------------------------------------

# In real scenarios, mislabeled data can be detected and removed by various methods,
# such as manual inspection, using confidence scores, or using models to predict labels.

# Example function to clean mislabeled data by identifying suspicious predictions.
def clean_mislabeled_data(model, data_loader, threshold=0.7, device='cpu'):
    model.eval()
    clean_data = []
    with torch.no_grad():
        for images, labels in data_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            softmax_probs = torch.nn.functional.softmax(outputs, dim=1)
            max_probs, preds = torch.max(softmax_probs, 1)
            
            # Only keep data points with high confidence predictions
            mask = max_probs > threshold
            clean_data.extend([(images[i], labels[i]) for i in range(len(mask)) if mask[i]])

    return clean_data

# --------------------------------------
# 5. Bias and Variance with Mismatched Data Distributions
# --------------------------------------

# To handle mismatched data distributions, we can use techniques like:
# - Resampling the data to balance the classes
# - Domain adaptation methods to handle domain shifts

# Simple example of a resampling technique to balance classes in a dataset (random oversampling)
def balance_classes(data_loader, target_class=1):
    # Collect all data points and labels
    all_data = [(image, label) for image, label in data_loader.dataset]
    class_data = [item for item in all_data if item[1] == target_class]
    balanced_data = all_data + class_data  # Oversampling target_class
    balanced_loader = DataLoader(balanced_data, batch_size=64, shuffle=True)
    return balanced_loader

# --------------------------------------
# 6. Transfer Learning (Using Pretrained Models)
# --------------------------------------

# We'll use a pretrained ResNet model for transfer learning.
from torchvision import models

class TransferLearningModel(nn.Module):
    def __init__(self, num_classes=10):
        super(TransferLearningModel, self).__init__()
        # Use a pretrained ResNet18 model
        self.resnet = models.resnet18(pretrained=True)
        
        # Freeze layers of ResNet except for the final fully connected layer
        for param in self.resnet.parameters():
            param.requires_grad = False
        
        # Modify the final fully connected layer
        self.resnet.fc = nn.Linear(self.resnet.fc.in_features, num_classes)

    def forward(self, x):
        return self.resnet(x)

# Example of training with transfer learning
transfer_model = TransferLearningModel(num_classes=10)
optimizer = optim.Adam(transfer_model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()

def train_transfer_model(model, train_loader, optimizer, criterion, epochs=5, device='cpu'):
    model.to(device)
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        correct = 0
        total = 0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)
        print(f"Epoch [{epoch+1}/{epochs}] - Loss: {total_loss/len(train_loader):.4f}, Accuracy: {100 * correct / total:.2f}%")

# --------------------------------------
# 7. Multi-task Learning
# --------------------------------------

class MultiTaskModel(nn.Module):
    def __init__(self, num_classes1=10, num_classes2=5):
        super(MultiTaskModel, self).__init__()
        # Shared feature extraction
        self.shared_fc = nn.Linear(28*28, 512)
        
        # Task-specific output layers
        self.task1_fc = nn.Linear(512, num_classes1)
        self.task2_fc = nn.Linear(512, num_classes2)
        
    def forward(self, x):
        x = x.view(-1, 28*28)
        shared_rep = torch.relu(self.shared_fc(x))
        
        task1_out = self.task1_fc(shared_rep)
        task2_out = self.task2_fc(shared_rep)
        
        return task1_out, task2_out

# Example of multi-task model training
multi_task_model = MultiTaskModel(num_classes1=10, num_classes2=5)
optimizer = optim.Adam(multi_task_model.parameters(), lr=0.001)

def train_multi_task_model(model, train_loader, optimizer, epochs=5, device='cpu'):
    model.to(device)
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            
            task1_out, task2_out = model(images)
            task1_loss = nn.CrossEntropyLoss()(task1_out, labels)
            task2_loss = nn.CrossEntropyLoss()(task2_out, labels)
            
            loss = task1_loss + task2_loss
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        print(f"Epoch [{epoch+1}/{epochs}] - Loss: {total_loss/len(train_loader):.4f}")

# --------------------------------------
# Example Usage
# --------------------------------------

# Train transfer learning model
train_transfer_model(transfer_model, train_loader, optimizer, criterion)

# Train multi-task model
train_multi_task_model(multi_task_model, train_loader, optimizer)

# Evaluate models
print("\nEvaluating Transfer Learning Model:")
evaluate_model(transfer_model, dev_loader)

print("\nEvaluating Multi-Task Model:")
evaluate_model(multi_task_model, dev_loader)
