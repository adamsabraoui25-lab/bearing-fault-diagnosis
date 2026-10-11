import scipy.io as sio
from scipy.signal import spectrogram
import numpy as np
from pathlib import Path
import json

FS = 12000
WINDOW_LENGTH = 2048
STRIDE = 1024
VAL_FRACTION = 0.2  # 20% de la fin de chaque signal réservé à la validation

DATA_DIR = Path(__file__).resolve().parent / "data"

LABEL_MAP = {
    "Normal_0.mat": "normal",
    "Ball.mat": "ball",
    "Inner_race.mat": "inner_race",
    "Outer_race.mat": "outer_race",
}
CLASSES = sorted(set(LABEL_MAP.values()))
LABEL_TO_IDX = {label: i for i, label in enumerate(CLASSES)}
print("Classes :", LABEL_TO_IDX)


def load_de_signal(path):
    mat = sio.loadmat(path)
    de_key = [k for k in mat.keys() if k.endswith("_DE_time")][0]
    return mat[de_key].squeeze()


def segment_signal(signal, window_length=WINDOW_LENGTH, stride=STRIDE):
    windows = []
    for start in range(0, len(signal) - window_length + 1, stride):
        windows.append(signal[start:start + window_length])
    return windows


def to_spectrogram(window):
    _, _, Sxx = spectrogram(window, fs=FS, nperseg=128, noverlap=64)
    return np.log1p(Sxx).astype(np.float32)


X_train, y_train, X_val, y_val = [], [], [], []

for filename, label in LABEL_MAP.items():
    signal = load_de_signal(DATA_DIR / filename)

    # 1. Découpe le signal en un bloc train et un bloc validation AVANT le fenêtrage
    split_point = int(len(signal) * (1 - VAL_FRACTION))
    signal_train = signal[:split_point]
    signal_val = signal[split_point:]

    windows_train = segment_signal(signal_train)
    windows_val = segment_signal(signal_val)
    print(f"{filename} -> {len(windows_train)} fenêtres train, {len(windows_val)} fenêtres val")

    for w in windows_train:
        X_train.append(to_spectrogram(w))
        y_train.append(LABEL_TO_IDX[label])
    for w in windows_val:
        X_val.append(to_spectrogram(w))
        y_val.append(LABEL_TO_IDX[label])

X_train, y_train = np.stack(X_train), np.array(y_train)
X_val, y_val = np.stack(X_val), np.array(y_val)

print(f"\nTrain : {X_train.shape}, répartition {np.bincount(y_train)}")
print(f"Val   : {X_val.shape}, répartition {np.bincount(y_val)}")

# Normalisation : calculée UNIQUEMENT sur le train, appliquée aux deux
# (sinon on "triche" en laissant les stats de validation influencer la normalisation)
mean, std = X_train.mean(), X_train.std()
X_train = (X_train - mean) / (std + 1e-8)
X_val = (X_val - mean) / (std + 1e-8)


with open(DATA_DIR / "norm_stats.json", "w") as f:
    json.dump({"mean": float(mean), "std": float(std)}, f)










np.save(DATA_DIR / "X_train.npy", X_train)
np.save(DATA_DIR / "y_train.npy", y_train)
np.save(DATA_DIR / "X_val.npy", X_val)
np.save(DATA_DIR / "y_val.npy", y_val)
print("\nSauvegardé : X_train.npy, y_train.npy, X_val.npy, y_val.npy")