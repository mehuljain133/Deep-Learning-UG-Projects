# Sequence Modeling: Recurrent Nets: Unfolding computational graphs, recurrent neural networks (RNNs), bidirectional RNNs, encoder-decoder sequence to sequence architectures, deep recurrent networks, LSTM networks.

import torch
import torch.nn as nn
import torch.optim as optim
from torchtext.datasets import IMDB
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
import matplotlib.pyplot as plt

# Set seed for reproducibility
torch.manual_seed(42)

# --------------------------------------
# Preprocess Text Data (IMDB Dataset)
# --------------------------------------

tokenizer = get_tokenizer('basic_english')

# Tokenize and build vocabulary
def yield_tokens(data_iter):
    for label, line in data_iter:
        yield tokenizer(line)

train_iter = IMDB(split='train')
vocab = build_vocab_from_iterator(yield_tokens(train_iter), specials=["<unk>"])
vocab.set_default_index(vocab["<unk>"])

# Define the text and label pipeline
def text_pipeline(x):
    return vocab(tokenizer(x))

def label_pipeline(label):
    return 1 if label == "pos" else 0

# Collate function to pad sequences
def collate_batch(batch):
    labels, texts = [], []
    for label, text in batch:
        labels.append(label_pipeline(label))
        processed = torch.tensor(text_pipeline(text), dtype=torch.int64)
        texts.append(processed)
    lengths = [len(t) for t in texts]
    padded = pad_sequence(texts, batch_first=True)
    return padded, torch.tensor(labels), torch.tensor(lengths)

train_iter = IMDB(split='train')
test_iter = IMDB(split='test')
train_loader = DataLoader(list(train_iter), batch_size=32, shuffle=True, collate_fn=collate_batch)
test_loader = DataLoader(list(test_iter), batch_size=64, shuffle=False, collate_fn=collate_batch)

# --------------------------------------
# Recurrent Neural Networks (RNN)
# --------------------------------------

class SimpleRNN(nn.Module):
    def __init__(self, vocab_size, embed_dim=64, hidden_dim=128, num_classes=2, num_layers=1):
        super(SimpleRNN, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.rnn = nn.RNN(embed_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, num_classes)

    def forward(self, x, lengths):
        x = self.embedding(x)
        packed_input = nn.utils.rnn.pack_padded_sequence(x, lengths, batch_first=True, enforce_sorted=False)
        packed_output, _ = self.rnn(packed_input)
        output, _ = nn.utils.rnn.pad_packed_sequence(packed_output, batch_first=True)
        final_output = output[torch.arange(output.size(0)), lengths - 1]
        return self.fc(final_output)

# --------------------------------------
# Bidirectional RNN (BiRNN)
# --------------------------------------

class BiRNN(nn.Module):
    def __init__(self, vocab_size, embed_dim=64, hidden_dim=128, num_classes=2, num_layers=1):
        super(BiRNN, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.rnn = nn.RNN(embed_dim, hidden_dim, num_layers, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(2 * hidden_dim, num_classes)  # *2 for bidirectional

    def forward(self, x, lengths):
        x = self.embedding(x)
        packed_input = nn.utils.rnn.pack_padded_sequence(x, lengths, batch_first=True, enforce_sorted=False)
        packed_output, _ = self.rnn(packed_input)
        output, _ = nn.utils.rnn.pad_packed_sequence(packed_output, batch_first=True)
        final_output = output[torch.arange(output.size(0)), lengths - 1]
        return self.fc(final_output)

# --------------------------------------
# LSTM Networks
# --------------------------------------

class LSTMNetwork(nn.Module):
    def __init__(self, vocab_size, embed_dim=64, hidden_dim=128, num_classes=2, num_layers=1):
        super(LSTMNetwork, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, num_classes)

    def forward(self, x, lengths):
        x = self.embedding(x)
        packed_input = nn.utils.rnn.pack_padded_sequence(x, lengths, batch_first=True, enforce_sorted=False)
        packed_output, (hn, cn) = self.lstm(packed_input)
        output, _ = nn.utils.rnn.pad_packed_sequence(packed_output, batch_first=True)
        final_output = output[torch.arange(output.size(0)), lengths - 1]
        return self.fc(final_output)

# --------------------------------------
# Encoder-Decoder Sequence-to-Sequence
# --------------------------------------

class Seq2SeqEncoder(nn.Module):
    def __init__(self, vocab_size, embed_dim=64, hidden_dim=128, num_layers=1):
        super(Seq2SeqEncoder, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.rnn = nn.LSTM(embed_dim, hidden_dim, num_layers, batch_first=True)

    def forward(self, x, lengths):
        x = self.embedding(x)
        packed_input = nn.utils.rnn.pack_padded_sequence(x, lengths, batch_first=True, enforce_sorted=False)
        packed_output, (hn, cn) = self.rnn(packed_input)
        return hn, cn

class Seq2SeqDecoder(nn.Module):
    def __init__(self, hidden_dim=128, output_dim=2, num_layers=1):
        super(Seq2SeqDecoder, self).__init__()
        self.rnn = nn.LSTM(hidden_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, encoder_hn, encoder_cn):
        output, _ = self.rnn(encoder_hn)
        return self.fc(output[:, -1, :])

class Seq2Seq(nn.Module):
    def __init__(self, vocab_size, embed_dim=64, hidden_dim=128, num_layers=1):
        super(Seq2Seq, self).__init__()
        self.encoder = Seq2SeqEncoder(vocab_size, embed_dim, hidden_dim, num_layers)
        self.decoder = Seq2SeqDecoder(hidden_dim)

    def forward(self, x, lengths):
        encoder_hn, encoder_cn = self.encoder(x, lengths)
        return self.decoder(encoder_hn, encoder_cn)

# --------------------------------------
# Training Function
# --------------------------------------

def train_model(model, loader, epochs=5):
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for x, y, lengths in loader:
            optimizer.zero_grad()
            output = model(x, lengths)
            loss = criterion(output, y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1}/{epochs} - Loss: {total_loss/len(loader):.4f}")

def evaluate_model(model, loader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for x, y, lengths in loader:
            output = model(x, lengths)
            preds = output.argmax(1)
            correct += (preds == y).sum().item()
            total += y.size(0)
    return 100 * correct / total

# Example Usage: Train and evaluate RNN models
rnn_model = SimpleRNN(vocab_size=len(vocab))
train_model(rnn_model, train_loader, epochs=5)
rnn_acc = evaluate_model(rnn_model, test_loader)
print(f"RNN Test Accuracy: {rnn_acc:.2f}%")

birnn_model = BiRNN(vocab_size=len(vocab))
train_model(birnn_model, train_loader, epochs=5)
birnn_acc = evaluate_model(birnn_model, test_loader)
print(f"BiRNN Test Accuracy: {birnn_acc:.2f}%")

lstm_model = LSTMNetwork(vocab_size=len(vocab))
train_model(lstm_model, train_loader, epochs=5)
lstm_acc = evaluate_model(lstm_model, test_loader)
print(f"LSTM Test Accuracy: {lstm_acc:.2f}%")

seq2seq_model = Seq2Seq(vocab_size=len(vocab))
train_model(seq2seq_model, train_loader, epochs=5)
seq2seq_acc = evaluate_model(seq2seq_model, test_loader)
print(f"Seq2Seq Test Accuracy: {seq2seq_acc:.2f}%")
