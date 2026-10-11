import numpy as np
import torch
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
from pathlib import Path

from dataset import BearingDataset
from model import BearingCNN
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
CLASSES = ["ball", "inner_race", "normal", "outer_race"]

X_val = np.load(DATA_DIR / "X_val.npy")
y_val = np.load(DATA_DIR / "y_val.npy")

model = BearingCNN(num_classes=len(CLASSES))
model.load_state_dict(torch.load(ROOT / "checkpoints" / "best_model.pt"))
model.eval()

loader = DataLoader(BearingDataset(X_val, y_val), batch_size=32)
all_preds, all_labels = [], []
with torch.no_grad():
    for X_batch, y_batch in loader:
        preds = model(X_batch).argmax(1)
        all_preds.extend(preds.tolist())
        all_labels.extend(y_batch.tolist())

print(classification_report(all_labels, all_preds, target_names=CLASSES, digits=3))

cm = confusion_matrix(all_labels, all_preds)
fig, ax = plt.subplots(figsize=(5, 5))
ax.imshow(cm, cmap="Blues")
ax.set_xticks(range(len(CLASSES)))
ax.set_yticks(range(len(CLASSES)))
ax.set_xticklabels(CLASSES, rotation=45, ha="right")
ax.set_yticklabels(CLASSES)
ax.set_xlabel("Prédit")
ax.set_ylabel("Réel")
for i in range(len(CLASSES)):
    for j in range(len(CLASSES)):
        ax.text(j, i, cm[i, j], ha="center", va="center",
                 color="white" if cm[i, j] > cm.max()/2 else "black")
plt.tight_layout()
(ROOT / "results").mkdir(exist_ok=True)
plt.savefig(ROOT / "results" / "confusion_matrix.png", dpi=150)
plt.show()