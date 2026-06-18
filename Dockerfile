# ============================================================================
# Fuser — imagen Docker con CUDA, FaceFusion y modelos incluidos.
#
# Deja TODO dentro de la imagen: dependencias, ambos motores (InsightFace +
# FaceFusion) y los modelos. En la otra PC solo necesitas Docker + NVIDIA
# Container Toolkit y:   docker compose up   (o el docker run de abajo).
#
# Base con CUDA 12 + cuDNN (compatible con onnxruntime-gpu).
# ============================================================================
FROM nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_NO_CACHE_DIR=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

WORKDIR /app

# --- Dependencias del sistema -------------------------------------------------
# python, git (para FaceFusion), ffmpeg, toolchain (insightface) y libs de OpenCV.
RUN apt-get update && apt-get install -y --no-install-recommends \
        python3.10 python3.10-venv python3-pip \
        git ffmpeg build-essential cmake \
        libgl1 libglib2.0-0 \
    && ln -sf /usr/bin/python3.10 /usr/bin/python \
    && rm -rf /var/lib/apt/lists/*

# --- Dependencias de Python (capa cacheable) ---------------------------------
COPY requirements.txt .
RUN python -m pip install --upgrade pip && pip install -r requirements.txt

# --- Código de la app ---------------------------------------------------------
COPY . .

# --- Motor FaceFusion (alta calidad), dentro de la imagen --------------------
RUN python scripts/install_facefusion.py || echo "FaceFusion opcional: continúo sin él"

# --- Pre-descarga de modelos (no fatal si no hay red en el build) -------------
RUN python scripts/download_models.py || true \
 && python -c "from insightface.app import FaceAnalysis; FaceAnalysis(name='buffalo_l', root='models').prepare(ctx_id=-1)" || true

# --- Arranque -----------------------------------------------------------------
EXPOSE 7860
ENV FUSER_HOST=0.0.0.0 FUSER_PORT=7860
CMD ["python", "app.py", "--listen"]
