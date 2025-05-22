import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler, StandardScaler
from sklearn.metrics import r2_score

# Paths to data (relative to project root)
TRAIN_INPUT_PATHS = [
    "dataset/Data_10k_sets/Data_10k_rng1/LHS_parameters_m.txt",
    "dataset/Data_10k_sets/Data_10k_rng2/LHS_parameters_m.txt",
    "dataset/Data_10k_sets/Data_10k_rng3/LHS_parameters_m.txt"
]
TRAIN_OUTPUT_PATHS = [
    "dataset/Data_10k_sets/Data_10k_rng1/iV_m.txt",
    "dataset/Data_10k_sets/Data_10k_rng2/iV_m.txt",
    "dataset/Data_10k_sets/Data_10k_rng3/iV_m.txt"
]
TEST_INPUT_PATHS = [
    "dataset/Data_1k_sets/Data_1k_rng1/LHS_parameters_m.txt",
    "dataset/Data_1k_sets/Data_1k_rng2/LHS_parameters_m.txt",
    "dataset/Data_1k_sets/Data_1k_rng3/LHS_parameters_m.txt"
]
TEST_OUTPUT_PATHS = [
    "dataset/Data_1k_sets/Data_1k_rng1/iV_m.txt",
    "dataset/Data_1k_sets/Data_1k_rng2/iV_m.txt",
    "dataset/Data_1k_sets/Data_1k_rng3/iV_m.txt"
]

# Load and concatenate data
X_train = np.vstack([np.loadtxt(p, delimiter=",") for p in TRAIN_INPUT_PATHS])
y_train = np.vstack([np.loadtxt(p, delimiter=",") for p in TRAIN_OUTPUT_PATHS])
X_test = np.vstack([np.loadtxt(p, delimiter=",") for p in TEST_INPUT_PATHS])
y_test = np.vstack([np.loadtxt(p, delimiter=",") for p in TEST_OUTPUT_PATHS])

epsilon = 1e-40
X_train = np.log10(X_train + epsilon)
X_test = np.log10(X_test + epsilon)

scaler = RobustScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

scaler_output = RobustScaler()
y_train_scaled = scaler_output.fit_transform(y_train)
y_test_scaled = scaler_output.transform(y_test)

# Print statistics for input and output
# print("Input features (train) mean/std/min/max:", np.mean(X_train, axis=0), np.std(X_train, axis=0), np.min(X_train, axis=0), np.max(X_train, axis=0))
# print("Input features (test) mean/std/min/max:", np.mean(X_test, axis=0), np.std(X_test, axis=0), np.min(X_test, axis=0), np.max(X_test, axis=0))
# print("Output (train) mean/std/min/max:", np.mean(y_train), np.std(y_train), np.min(y_train), np.max(y_train))
# print("Output (test) mean/std/min/max:", np.mean(y_test), np.std(y_test), np.min(y_test), np.max(y_test))

# Convert to PyTorch tensors
X_train_tensor = torch.tensor(X_train_scaled, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train_scaled, dtype=torch.float32)
X_test_tensor = torch.tensor(X_test_scaled, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test_scaled, dtype=torch.float32)

# Dataset and DataLoader
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32)

# MLP Model
class MLP(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(MLP, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, output_dim)
        )

    def forward(self, x):
        return self.net(x)

input_dim = X_train.shape[1] # shape[0] is number of samples
output_dim = y_train.shape[1]
model = MLP(input_dim, output_dim)

# Loss and optimizer
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=2e-2)

# Training loop
EPOCHS = 300
for epoch in range(EPOCHS):
    model.train()
    running_loss = 0.0
    for xb, yb in train_loader:
        optimizer.zero_grad()
        preds = model(xb)
        loss = criterion(preds, yb)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * xb.size(0)
    epoch_loss = running_loss / len(train_loader.dataset)
    if (epoch + 1) % 10 == 0 or epoch == 0:
        print(f"Epoch {epoch+1}/{EPOCHS}, Loss: {epoch_loss:.4f}")

# Evaluation
model.eval()
with torch.no_grad():
    preds = model(X_test_tensor)
    test_loss = criterion(preds, y_test_tensor).item()
print(f"Test MSE: {test_loss:.4f}")

# R^2 score
r2 = r2_score(y_test_tensor.numpy(), preds.numpy())
print(f"Test R^2: {r2:.4f}")

# Save model
torch.save(model.state_dict(), "model/mlp_ivcurve.pth")