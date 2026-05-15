# =========================
# 1) Imports
# =========================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, classification_report

import torch
import torch.nn as nn
import torch.optim as optim


# =========================
# 2) Load Dataset
# =========================
df = pd.read_csv("heart.csv")

print("Dataset Shape:", df.shape)
print("\nMissing Values:\n", df.isnull().sum())

# Handle missing values (if any)
df = df.dropna()

# (Optional) simulate larger dataset
df = pd.concat([df] * 100, ignore_index=True)
df = df.sample(frac=1, random_state=42).reset_index(drop=True)


# =========================
# 3) Encode Categorical Variables
# =========================
# heart.csv features are already numeric, but we confirm here
# If any object columns exist, encode them
cat_cols = df.select_dtypes(include=["object"]).columns.tolist()
if cat_cols:
    df = pd.get_dummies(df, columns=cat_cols, drop_first=True)
    print(f"\nEncoded categorical columns: {cat_cols}")
else:
    print("\nNo categorical columns found — all features are numeric.")


# =========================
# 4) Split Features / Target
# =========================
X = df.drop("target", axis=1)
y = df["target"]


# =========================
# 5) Train / Test Split
# =========================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# further split train -> train / validation
X_train, X_val, y_train, y_val = train_test_split(
    X_train, y_train, test_size=0.2, random_state=42
)


# =========================
# 6) Scaling
# =========================
scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_val   = scaler.transform(X_val)
X_test  = scaler.transform(X_test)


# =========================
# 7) Convert to Tensors
# =========================
X_train_t = torch.tensor(X_train, dtype=torch.float32)
X_val_t   = torch.tensor(X_val,   dtype=torch.float32)
X_test_t  = torch.tensor(X_test,  dtype=torch.float32)

y_train_t = torch.tensor(y_train.values, dtype=torch.float32).view(-1, 1)
y_val_t   = torch.tensor(y_val.values,   dtype=torch.float32).view(-1, 1)
y_test_t  = torch.tensor(y_test.values,  dtype=torch.float32).view(-1, 1)

input_size = X_train_t.shape[1]


# =========================
# 8) Model Definition (MLP)
# =========================
class MLP(nn.Module):
    def __init__(self, hidden_size=32, activation=nn.ReLU()):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            activation,
            nn.Dropout(0.3),

            nn.Linear(hidden_size, hidden_size // 2),
            activation,
            nn.Dropout(0.3),

            nn.Linear(hidden_size // 2, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.net(x)


# =========================
# 9) Training Function
# =========================
def train_model(model, optimizer, epochs=50):

    criterion = nn.BCELoss()

    train_losses, val_losses = [], []
    train_accs,   val_accs   = [], []

    for epoch in range(epochs):

        # -------- TRAIN --------
        model.train()

        outputs = model(X_train_t)
        loss    = criterion(outputs, y_train_t)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_pred = (outputs > 0.5).float()
        train_acc  = (train_pred == y_train_t).sum().item() / y_train_t.size(0)

        # -------- VALIDATION --------
        model.eval()
        with torch.no_grad():

            val_outputs = model(X_val_t)
            val_loss    = criterion(val_outputs, y_val_t)

            val_pred = (val_outputs > 0.5).float()
            val_acc  = (val_pred == y_val_t).sum().item() / y_val_t.size(0)

        # save
        train_losses.append(loss.item())
        val_losses.append(val_loss.item())
        train_accs.append(train_acc)
        val_accs.append(val_acc)

        if (epoch + 1) % 10 == 0:
            print(f"  Epoch {epoch+1:>3}: "
                  f"Train Loss={loss.item():.4f}, "
                  f"Val Loss={val_loss.item():.4f}, "
                  f"Train Acc={train_acc:.4f}, "
                  f"Val Acc={val_acc:.4f}")

    return train_losses, val_losses, train_accs, val_accs


# =========================
# 10) Evaluation Function
# =========================
def evaluate_model(model, experiment_name):

    criterion = nn.BCELoss()

    model.eval()
    with torch.no_grad():

        test_outputs = model(X_test_t)
        test_loss    = criterion(test_outputs, y_test_t)
        test_pred    = (test_outputs > 0.5).float()
        test_acc     = (test_pred == y_test_t).sum().item() / y_test_t.size(0)

        # MSE
        test_mse = nn.MSELoss()(test_outputs, y_test_t).item()

    print(f"\n{'='*50}")
    print(f"  {experiment_name} — Test Results")
    print(f"{'='*50}")
    print(f"  Final Test Loss : {test_loss.item():.4f}")
    print(f"  Test Accuracy   : {test_acc:.4f}")
    print(f"  Test MSE        : {test_mse:.4f}")

    cm = confusion_matrix(y_test_t, test_pred)
    print(f"\n  Confusion Matrix:\n{cm}")
    print(f"\n  Classification Report:\n")
    print(classification_report(y_test_t, test_pred))

    return test_loss.item(), test_acc, test_mse


# =========================
# 11) Experiments
# =========================

# ---- Experiment 1: ReLU | 32 neurons | lr=0.0005 (Baseline) ----
print("\n" + "="*60)
print("EXPERIMENT 1: ReLU | hidden=32 | lr=0.0005")
print("="*60)

model_exp1    = MLP(hidden_size=32, activation=nn.ReLU())
optimizer_exp1 = optim.Adam(model_exp1.parameters(), lr=0.0005)

losses1 = train_model(model_exp1, optimizer_exp1, epochs=50)
result1 = evaluate_model(model_exp1, "Experiment 1: ReLU | 32 | lr=0.0005")


# ---- Experiment 2: Tanh | 32 neurons | lr=0.0005 ----
print("\n" + "="*60)
print("EXPERIMENT 2: Tanh | hidden=32 | lr=0.0005")
print("="*60)

model_exp2     = MLP(hidden_size=32, activation=nn.Tanh())
optimizer_exp2 = optim.Adam(model_exp2.parameters(), lr=0.0005)

losses2 = train_model(model_exp2, optimizer_exp2, epochs=50)
result2 = evaluate_model(model_exp2, "Experiment 2: Tanh | 32 | lr=0.0005")


# ---- Experiment 3: ReLU | 64 neurons | lr=0.001 ----
print("\n" + "="*60)
print("EXPERIMENT 3: ReLU | hidden=64 | lr=0.001")
print("="*60)

model_exp3     = MLP(hidden_size=64, activation=nn.ReLU())
optimizer_exp3 = optim.Adam(model_exp3.parameters(), lr=0.001)

losses3 = train_model(model_exp3, optimizer_exp3, epochs=50)
result3 = evaluate_model(model_exp3, "Experiment 3: ReLU | 64 | lr=0.001")


# =========================
# 12) Comparison Table
# =========================
print("\n" + "="*60)
print("COMPARISON TABLE")
print("="*60)

results = {
    "Experiment": [
        "Exp 1: ReLU | 32 | lr=0.0005",
        "Exp 2: Tanh | 32 | lr=0.0005",
        "Exp 3: ReLU | 64 | lr=0.001",
    ],
    "Test Loss": [result1[0], result2[0], result3[0]],
    "Test Accuracy": [result1[1], result2[1], result3[1]],
    "Test MSE": [result1[2], result2[2], result3[2]],
}

results_df = pd.DataFrame(results)
print(results_df.to_string(index=False))


# =========================
# 13) Visualization
# =========================
train_losses1, val_losses1, train_accs1, val_accs1 = losses1
train_losses2, val_losses2, train_accs2, val_accs2 = losses2
train_losses3, val_losses3, train_accs3, val_accs3 = losses3

# ---- Loss Curves ----
fig, axes = plt.subplots(1, 3, figsize=(15, 4))

for ax, tl, vl, title in zip(
    axes,
    [train_losses1, train_losses2, train_losses3],
    [val_losses1,   val_losses2,   val_losses3],
    ["Exp 1: ReLU | 32 | lr=0.0005",
     "Exp 2: Tanh | 32 | lr=0.0005",
     "Exp 3: ReLU | 64 | lr=0.001"]
):
    ax.plot(tl, label="Train Loss")
    ax.plot(vl, label="Val Loss")
    ax.set_title(title)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.legend()

plt.suptitle("Loss Curves — All Experiments", fontsize=14)
plt.tight_layout()
plt.show()

# ---- Accuracy Curves ----
fig, axes = plt.subplots(1, 3, figsize=(15, 4))

for ax, ta, va, title in zip(
    axes,
    [train_accs1, train_accs2, train_accs3],
    [val_accs1,   val_accs2,   val_accs3],
    ["Exp 1: ReLU | 32 | lr=0.0005",
     "Exp 2: Tanh | 32 | lr=0.0005",
     "Exp 3: ReLU | 64 | lr=0.001"]
):
    ax.plot(ta, label="Train Accuracy")
    ax.plot(va, label="Val Accuracy")
    ax.set_title(title)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Accuracy")
    ax.legend()

plt.suptitle("Accuracy Curves — All Experiments", fontsize=14)
plt.tight_layout()
plt.show()

# ---- Bar Chart: Test Accuracy Comparison ----
exp_labels = ["Exp 1\nReLU|32|0.0005", "Exp 2\nTanh|32|0.0005", "Exp 3\nReLU|64|0.001"]
accuracies  = [result1[1], result2[1], result3[1]]

plt.figure(figsize=(7, 4))
plt.bar(exp_labels, accuracies, color=["steelblue", "tomato", "seagreen"])
plt.ylim(0, 1)
plt.title("Test Accuracy Comparison")
plt.ylabel("Accuracy")
for i, v in enumerate(accuracies):
    plt.text(i, v + 0.01, f"{v:.4f}", ha="center", fontsize=10)
plt.tight_layout()
plt.show()
