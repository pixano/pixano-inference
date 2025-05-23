## Utiliser une image Cuda officielle
FROM nvidia/cuda:12.3.0-base-ubuntu22.04

RUN apt-get update && apt-get install -y \
    git python3 python3-pip \
    && rm -rf /var/lib/apt/lists/*

# Définir le répertoire de travail
WORKDIR /app

# Installer pixano-inference et ses dependances
RUN pip install --upgrade pip \
    && pip install pixano-inference \
    && pip install pixano-inference[vllm]

# Exposer le port utilisé par FastAPI (défini dans main.py -> port 8000 par défaut)
EXPOSE 8000

ENV UVICORN_HOST=0.0.0.0
ENV UVICORN_PORT=8000

# Commande de démarrage du service (via uvicorn)
CMD ["uvicorn", "pixano_inference.main:fast_api_app"]


