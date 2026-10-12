# ⚙️ Bearing Fault Diagnosis & MLOps Pipeline

An end-to-end industrial predictive maintenance system that leverages Deep Learning, Signal Processing, and MLOps principles to automatically detect and classify mechanical bearing faults from vibration signals.

---

## 🛠️ Tech Stack & Architecture

- **Signal Processing (DSP):** Windowing (2048 points) of raw vibration data converted into log-scaled spectrograms using Short-Time Fourier Transform (STFT).
- **Deep Learning:** Custom Convolutional Neural Network (`BearingCNN`) built with **PyTorch** for multi-class classification.
- **API & Backend:** Asynchronous web service powered by **FastAPI** and **Uvicorn**.
- **Containerization:** Fully containerized environment using **Docker** for guaranteed reproducibility across any machine.

---

## 📂 Project Structure

```text
bearing-fault-diagnosis/
│
├── api.py                  # FastAPI application exposing the /predict endpoint
├── model.py                # PyTorch model definition (BearingCNN)
├── best_model.pt           # Pre-trained model weights
├── Dockerfile              # Docker configuration for containerization
├── requirements.txt        # Python dependencies
├── .dockerignore           # Optimized build context
└── README.md