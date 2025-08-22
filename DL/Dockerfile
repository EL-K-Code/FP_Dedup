FROM tensorflow/tensorflow:2.10.1

ENV DEBIAN_FRONTEND=noninteractive PIP_NO_CACHE_DIR=1 \
    TF_CPP_MIN_LOG_LEVEL=2 TF_ENABLE_ONEDNN_OPTS=0

RUN apt-get update && apt-get install -y --no-install-recommends \
    git build-essential pkg-config \
    libglib2.0-0 libsm6 libxext6 libxrender1 libgl1 \
    ffmpeg ca-certificates \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# 🔒 versions stables compatibles FingerFlow/TF 2.10
RUN python -m pip install --upgrade pip && \
    pip install \
      numpy==1.23.5 \
      scipy==1.10.1 \
      scikit-image==0.19.3 \
      opencv-python-headless==4.9.0.80 \
      pandas==2.0.3 \
      matplotlib==3.7.3 \
      fingerflow==3.0.1

COPY . /app

ENV PYTHONPATH=/app/src
CMD ["python", "scripts/run_verifier.py"]
