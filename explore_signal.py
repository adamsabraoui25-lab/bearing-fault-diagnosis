import scipy.io as sio
import matplotlib.pyplot as plt

FILENAME = "Normal_0.mat"

# 1. Charger le fichier .mat
mat = sio.loadmat(f"data/{FILENAME}")

print("Clés disponibles dans ce fichier :")
for key in mat.keys():
    print(" -", key)

# 2. Chercher automatiquement la clé du signal Drive End (DE)
#    On ne connaît pas le numéro exact à l'avance, donc on cherche
#    la clé qui SE TERMINE par "_DE_time"
de_key = [k for k in mat.keys() if k.endswith("_DE_time")][0]
print(f"\nClé DE trouvée : {de_key}")

# 3. Extraire le signal (c'est un tableau 2D de forme (N, 1), on l'aplatit en 1D)
signal = mat[de_key].squeeze()
print(f"Nombre d'échantillons : {len(signal)}")

# 4. Tracer les 2000 premiers points (tout tracer serait illisible)
plt.figure(figsize=(10, 4))
plt.plot(signal[:2000])
plt.title(f"Signal vibratoire (Drive End) - {FILENAME}")
plt.xlabel("Échantillon")
plt.ylabel("Amplitude")
plt.grid(True)
plt.show()