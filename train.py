import os
os.environ["MLFLOW_ALLOW_FILE_STORE"] = "true"

import numpy as np
import torch
import mlflow
import mlflow.pytorch
from torch.utils.data import DataLoader
from pathlib import Path

from dataset import BearingDataset
from model import BearingCNN

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
CHECKPOINTS_DIR = ROOT / "checkpoints"
CHECKPOINTS_DIR.mkdir(exist_ok=True)

EPOCHS = 20
BATCH_SIZE = 16
LR = 1e-3
SEED = 42

CLASSES = ["ball", "inner_race", "normal", "outer_race"]

torch.manual_seed(SEED)
np.random.seed(SEED)

# --- 1. Charger les données déjà split par build_dataset.py ---
X_train = np.load(DATA_DIR / "X_train.npy")
y_train = np.load(DATA_DIR / "y_train.npy")
X_val = np.load(DATA_DIR / "X_val.npy")
y_val = np.load(DATA_DIR / "y_val.npy")

print(f"Train : {X_train.shape[0]} exemples | Val : {X_val.shape[0]} exemples")

train_loader = DataLoader(BearingDataset(X_train, y_train), batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(BearingDataset(X_val, y_val), batch_size=BATCH_SIZE, shuffle=False)

# --- 2. Modèle, optimiseur, fonction de perte ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device utilisé : {device}")

model = BearingCNN(num_classes=len(CLASSES)).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=LR)
criterion = torch.nn.CrossEntropyLoss()


def run_epoch(loader, training):
    model.train() if training else model.eval()
    total_loss, correct, total = 0.0, 0, 0
    with torch.set_grad_enabled(training):
        for X_batch, y_batch in loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)

            if training:
                optimizer.zero_grad()
            out = model(X_batch)
            loss = criterion(out, y_batch)
            if training:
                loss.backward()
                optimizer.step()

            total_loss += loss.item() * X_batch.size(0)
            correct += (out.argmax(1) == y_batch).sum().item()
            total += X_batch.size(0)
    return total_loss / total, correct / total


# --- 3. Entraînement avec tracking MLflow ---
tracking_dir = ROOT / "mlruns"
tracking_dir.mkdir(exist_ok=True)
mlflow.set_tracking_uri(tracking_dir.as_uri())

mlflow.set_experiment("bearing-fault-diagnosis")

with mlflow.start_run():
    mlflow.log_params({
        "epochs": EPOCHS,
        "batch_size": BATCH_SIZE,
        "lr": LR,
        "seed": SEED,
        "model": "BearingCNN",
    })

    best_val_acc = 0.0
    for epoch in range(1, EPOCHS + 1):
        train_loss, train_acc = run_epoch(train_loader, training=True)
        val_loss, val_acc = run_epoch(val_loader, training=False)

        print(f"epoch {epoch:02d} | train_loss {train_loss:.4f} acc {train_acc:.3f} "
              f"| val_loss {val_loss:.4f} acc {val_acc:.3f}")

        mlflow.log_metrics({
            "train_loss": train_loss,
            "train_acc": train_acc,
            "val_loss": val_loss,
            "val_acc": val_acc,
        }, step=epoch)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), CHECKPOINTS_DIR / "best_model.pt")

    mlflow.log_metric("best_val_acc", best_val_acc)

    # exemple d'entrée nécessaire pour que MLflow trace correctement le modèle
    input_example = np.random.randn(1, 1, 65, 31).astype(np.float32)
    mlflow.pytorch.log_model(model, "model", input_example=input_example)

    print(f"\nMeilleure accuracy validation : {best_val_acc:.3f}")
    print(f"Modèle sauvegardé dans checkpoints/best_model.pt")