import json
from pathlib import Path

import numpy as np
import torch
from fastapi import FastAPI
from pydantic import BaseModel, field_validator
from scipy.signal import spectrogram

from model import BearingCNN

ROOT = Path(__file__).resolve().parent
FS = 12000
WINDOW_LENGTH = 2048
CLASSES = ["ball", "inner_race", "normal", "outer_race"]

# --- Chargé UNE SEULE FOIS au démarrage de l'API ---
with open(ROOT / "data" / "norm_stats.json") as f:
    stats = json.load(f)
MEAN, STD = stats["mean"], stats["std"]

model = BearingCNN(num_classes=len(CLASSES))
model.load_state_dict(torch.load(ROOT / "checkpoints" / "best_model.pt", map_location="cpu"))
model.eval()

app = FastAPI(title="Bearing Fault Diagnosis API", version="1.0.0")


class SignalRequest(BaseModel):
    signal: list[float]

    @field_validator("signal")
    @classmethod
    def check_length(cls, v):
        if len(v) != WINDOW_LENGTH:
            raise ValueError(f"Le signal doit contenir exactement {WINDOW_LENGTH} valeurs (reçu : {len(v)})")
        return v


def preprocess(signal):
    """Même pipeline qu'à l'entraînement : signal -> spectrogramme log -> normalisation."""
    _, _, Sxx = spectrogram(np.array(signal, dtype=np.float32), fs=FS, nperseg=128, noverlap=64)
    Sxx = np.log1p(Sxx).astype(np.float32)
    Sxx = (Sxx - MEAN) / (STD + 1e-8)
    return torch.from_numpy(Sxx).unsqueeze(0).unsqueeze(0)  # shape (1, 1, 65, 31)


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": True, "classes": CLASSES}


@app.post("/predict")
def predict(request: SignalRequest):
    x = preprocess(request.signal)
    with torch.no_grad():
        probs = torch.softmax(model(x), dim=1)[0]
    idx = int(probs.argmax())
    return {
        "prediction": CLASSES[idx],
        "confidence": round(float(probs[idx]), 4),
        "probabilities": {c: round(float(p), 4) for c, p in zip(CLASSES, probs)},
    }