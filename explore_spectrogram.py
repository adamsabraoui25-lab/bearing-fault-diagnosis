import scipy.io as sio
from scipy.signal import spectrogram
import matplotlib.pyplot as plt
import numpy as np

FILENAME = "Ball.mat"
FS = 12000  # fréquence d'échantillonnage CWRU (12 kHz pour nos fichiers)

# 1. Charger le signal (comme avant)
mat = sio.loadmat(f"data/{FILENAME}")
de_key = [k for k in mat.keys() if k.endswith("_DE_time")][0]
signal = mat[de_key].squeeze()

# 2. Prendre une fenêtre de 2048 échantillons (~0.17 secondes)
window = signal[:2048]

# 3. Calculer le spectrogramme (STFT)
frequencies, times, Sxx = spectrogram(window, fs=FS, nperseg=128, noverlap=64)

# 4. Compression logarithmique (sinon les basses fréquences dominent tout visuellement)
Sxx_log = np.log1p(Sxx)

print("Forme du spectrogramme (freq_bins, time_bins) :", Sxx_log.shape)

# 5. Affichage côte à côte : signal brut vs spectrogramme
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

axes[0].plot(window)
axes[0].set_title(f"Signal brut - {FILENAME}")
axes[0].set_xlabel("Échantillon")
axes[0].set_ylabel("Amplitude")

axes[1].pcolormesh(times, frequencies, Sxx_log, shading="gouraud")
axes[1].set_title("Spectrogramme (log)")
axes[1].set_xlabel("Temps (s)")
axes[1].set_ylabel("Fréquence (Hz)")

plt.tight_layout()
plt.show()