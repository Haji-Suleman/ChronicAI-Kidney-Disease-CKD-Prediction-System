# model.py
# Date: 2026-02-04
# Author: Haji Suleman
# Project: 🩺 Chronic Kidney Disease (CKD) Prediction System

from cleaning import clean_ckd_dataset
import torch
from torch import nn
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

RANDOM_SEED = 42

# Load dataset
df = clean_ckd_dataset()


# Features & target
X = df.drop("classification", axis=1).values
y = df["classification"].values
scaler = StandardScaler()
X = scaler.fit_transform(X)
# Convert to tensors
X = torch.tensor(X, dtype=torch.float32)
y = torch.tensor(y, dtype=torch.float32).unsqueeze(1)

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=RANDOM_SEED
)


# Define model
class Chronic_Kidney_Disease(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 28),
            nn.ReLU(),
            nn.Linear(28, 10),
            nn.ReLU(),
            nn.Linear(10, 1),
            nn.Sigmoid(),  # binary classification
        )

    def forward(self, X):
        return self.model(X)


model_8 = Chronic_Kidney_Disease(X_train.shape[1])

# Loss & Optimizer
loss_fn = nn.BCELoss()
optimizer = torch.optim.Adam(model_8.parameters(), lr=0.001)  # lower LR

# Training loop
epochs = 400
train_loss_list = []
test_loss_list = []

for epoch in range(epochs):
    model_8.train()
    y_logits = model_8(X_train)  # sigmoid already in model
    loss = loss_fn(y_logits, y_train)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    model_8.eval()
    with torch.no_grad():
        test_logits = model_8(X_test)
        test_loss = loss_fn(test_logits, y_test)

    train_loss_list.append(loss.item())
    test_loss_list.append(test_loss.item())

    if epoch % 40 == 0:
        print(
            f"Epoch {epoch}: Train Loss={loss.item():.4f}, Test Loss={test_loss.item():.4f}"
        )

# Plot losses
plt.plot(range(epochs), train_loss_list, label="Train Loss")
plt.plot(range(epochs), test_loss_list, label="Test Loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.title("Chronic Kidney Disease (CKD) Prediction System")
plt.legend()
plt.show()
