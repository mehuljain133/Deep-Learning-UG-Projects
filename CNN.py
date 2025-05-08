# Convolution Neural Networks: Introduction to convolution neural networks: stacking, striding and pooling, applications like image, and text classification.

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torchtext.datasets import IMDB
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

# Set seed
torch.manual_seed(42)

# --------------------------------------
# IMAGE CLASSIFICATION WITH CNN
# --------------------------------------

# Data transforms
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# CIFAR-10 dataset
train_imgset = datasets.CIFAR10(root='./data', train=True, transform=transform, download=True)
test_imgset = datasets.CIFAR10(root='./data', train=False, transform=transform, download=True)

train_img_loader = torch.utils.data.DataLoader(train_imgset, batch_size=64, shuffle=True)
test_img_loader = torch.utils.data.DataLoader(test_imgset, batch_size=1000, shuffle=False)

# CNN for image classification
class CNNImage(nn.Module):
    def __init__(self):
        super(CNNImage, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)  # stack 1
        self.pool = nn.MaxPool2d(2, 2)  # stride 2 pooling
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)  # stack 2
        self.fc1 = nn.Linear(64 * 8 * 8, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))  # 32x32 -> 16x16
        x = self.pool(F.relu(self.conv2(x)))  # 16x16 -> 8x8
        x = x.view(-1, 64 * 8 * 8)
        x = F.relu(self.fc1(x))
        return self.fc2(x)

# Train function
def train_model(model, loader, epochs=5, lr=0.001):
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    for epoch in range(epochs):
        model.train()
        running_loss = 0
        for data, target in loader:
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        print(f"[Image] Epoch {epoch+1}/{epochs} - Loss: {running_loss/len(loader):.4f}")

# Evaluate function
def evaluate_model(model, loader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for data, target in loader:
            output = model(data)
            preds = output.argmax(1)
            correct += (preds == target).sum().item()
            total += target.size(0)
    return 100 * correct / total

# Run image classification
cnn_img = CNNImage()
train_model(cnn_img, train_img_loader, epochs=5)
img_acc = evaluate_model(cnn_img, test_img_loader)
print(f"[Image] Test Accuracy: {img_acc:.2f}%")

# --------------------------------------
# TEXT CLASSIFICATION WITH CNN
# --------------------------------------

# Prepare IMDB text data
tokenizer = get_tokenizer('basic_english')

def yield_tokens(data_iter):
    for label, line in data_iter:
        yield tokenizer(line)

train_iter = IMDB(split='train')
vocab = build_vocab_from_iterator(yield_tokens(train_iter), specials=["<unk>"])
vocab.set_default_index(vocab["<unk>"])

def text_pipeline(x):
    return vocab(tokenizer(x))

def label_pipeline(label):
    return 1 if label == "pos" else 0

# Process data
def collate_batch(batch):
    labels, texts = [], []
    for label, text in batch:
        labels.append(label_pipeline(label))
        processed = torch.tensor(text_pipeline(text), dtype=torch.int64)
        texts.append(processed)
    lengths = [len(t) for t in texts]
    padded = nn.utils.rnn.pad_sequence(texts, batch_first=True)
    return padded, torch.tensor(labels), torch.tensor(lengths)

train_iter = IMDB(split='train')
test_iter = IMDB(split='test')
train_text_loader = DataLoader(list(train_iter), batch_size=32, shuffle=True, collate_fn=collate_batch)
test_text_loader = DataLoader(list(test_iter), batch_size=64, shuffle=False, collate_fn=collate_batch)

# CNN for text
class CNNText(nn.Module):
    def __init__(self, vocab_size, embed_dim=64, num_classes=2):
        super(CNNText, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.conv1 = nn.Conv1d(embed_dim, 128, kernel_size=5, stride=1)
        self.pool = nn.AdaptiveMaxPool1d(1)
        self.fc = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.embedding(x)  # (batch, seq_len, embed)
        x = x.permute(0, 2, 1)  # (batch, embed, seq_len)
        x = F.relu(self.conv1(x))
        x = self.pool(x).squeeze(2)
        return self.fc(x)

# Train & eval for text
def train_text_model(model, loader, epochs=5):
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for x, y, _ in loader:
            optimizer.zero_grad()
            output = model(x)
            loss = criterion(output, y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"[Text] Epoch {epoch+1} - Loss: {total_loss/len(loader):.4f}")

def evaluate_text_model(model, loader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for x, y, _ in loader:
            output = model(x)
            preds = output.argmax(1)
            correct += (preds == y).sum().item()
            total += y.size(0)
    return 100 * correct / total

# Run text classification
cnn_text = CNNText(vocab_size=len(vocab))
train_text_model(cnn_text, train_text_loader, epochs=3)
text_acc = evaluate_text_model(cnn_text, test_text_loader)
print(f"[Text] Test Accuracy: {text_acc:.2f}%")
