import requests
import scipy.io as sio
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent / "data"

for filename in ["Normal_0.mat", "Ball.mat", "Inner_race.mat", "Outer_race.mat"]:
    mat = sio.loadmat(DATA_DIR / filename)
    key = [k for k in mat.keys() if k.endswith("_DE_time")][0]
    signal = mat[key].squeeze()
    window = signal[-2048:]  
    # fin du signal = zone de validation, jamais vue à l'entraînement

    r = requests.post("http://127.0.0.1:8000/predict", json={"signal": window.tolist()})
    print(filename, "->", r.json())