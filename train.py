# ==========================================
# SELF-PRUNING CNN 
# ==========================================

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import numpy as np
import random

# ==========================================
# REPRODUCIBILITY
# ==========================================
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

set_seed()

# ==========================================
# PATH SETUP
# ==========================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
RESULTS_DIR = os.path.join(BASE_DIR, "results")

os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ==========================================
# PRUNABLE LINEAR LAYER
# ==========================================
class PrunableLinear(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(out_features, in_features) * 0.01)
        self.bias = nn.Parameter(torch.zeros(out_features))
        self.gate_scores = nn.Parameter(torch.randn(out_features, in_features))

    def forward(self, x):
        gates = torch.sigmoid(self.gate_scores)
        pruned_weight = self.weight * gates
        return F.linear(x, pruned_weight, self.bias)

    def get_gates(self):
        return torch.sigmoid(self.gate_scores)

# ==========================================
# CNN MODEL 
# ==========================================
class PrunableCNN(nn.Module):
    def __init__(self):
        super().__init__()

        # Convolutional backbone
        self.conv = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )

        self.dropout = nn.Dropout(0.3)

        # Prunable fully connected layers
        self.fc1 = PrunableLinear(64 * 8 * 8, 256)
        self.fc2 = PrunableLinear(256, 10)

    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        x = self.dropout(F.relu(self.fc1(x)))
        x = self.fc2(x)
        return x

    def get_all_gates(self):
        gates = []
        for m in self.modules():
            if isinstance(m, PrunableLinear):
                gates.append(m.get_gates().view(-1))
        return torch.cat(gates)

# ==========================================
# DATA NORMALIZATION
# ==========================================
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

train_dataset = datasets.CIFAR10(DATA_DIR, train=True, download=True, transform=transform)
test_dataset = datasets.CIFAR10(DATA_DIR, train=False, download=True, transform=transform)

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=128, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=128, shuffle=False)

# ==========================================
# TRAINING
# ==========================================
def train_model(lambda_val, epochs=10):
    model = PrunableCNN().to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(epochs):
        model.train()
        running_loss = 0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()

            outputs = model(images)
            ce_loss = criterion(outputs, labels)

            gates = model.get_all_gates()
            sparsity_loss = torch.sum(gates)

            loss = ce_loss + lambda_val * sparsity_loss

            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        print(f"[λ={lambda_val}] Epoch {epoch+1}/{epochs} | Loss: {running_loss:.2f}")

    return model

# ==========================================
# EVALUATION
# ==========================================
def evaluate_model(model):
    model.eval()
    correct, total = 0, 0

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, pred = torch.max(outputs, 1)

            correct += (pred == labels).sum().item()
            total += labels.size(0)

    acc = 100 * correct / total

    gates = model.get_all_gates().detach().cpu().numpy()
    sparsity = np.mean(gates < 1e-2) * 100

    return acc, sparsity, gates

# ==========================================
# EXPERIMENTS
# ==========================================
lambda_values = [1e-5, 5e-5, 1e-4]
results = []

for lam in lambda_values:
    print(f"\nTraining λ={lam}")
    model = train_model(lam)

    acc, sparsity, gates = evaluate_model(model)
    results.append((lam, acc, sparsity))

    print(f"λ={lam} | Acc={acc:.2f}% | Sparsity={sparsity:.2f}%")

    # Plot histogram
    plt.figure()
    plt.hist(gates, bins=50)
    plt.title(f"Gate Distribution (λ={lam})")

    path = os.path.join(RESULTS_DIR, f"gate_lambda_{lam}.png")
    plt.savefig(path)
    plt.close()

# ==========================================
# SAVE RESULTS
# ==========================================
results_path = os.path.join(RESULTS_DIR, "results.txt")

with open(results_path, "w") as f:
    f.write("Lambda\tAccuracy\tSparsity\n")
    for r in results:
        f.write(f"{r[0]}\t{r[1]:.2f}%\t{r[2]:.2f}%\n")

print("\nFinal Results:")
for r in results:
    print(r)