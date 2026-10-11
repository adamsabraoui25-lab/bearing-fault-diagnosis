FROM python:3.12-slim

WORKDIR /app

# PyTorch version CPU uniquement (beaucoup plus légère que la version GPU)
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

COPY requirements-api.txt .
RUN pip install --no-cache-dir -r requirements-api.txt

# Seulement ce dont l'API a besoin
COPY api.py model.py ./
COPY checkpoints/best_model.pt checkpoints/best_model.pt
COPY data/norm_stats.json data/norm_stats.json

EXPOSE 8000

CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]