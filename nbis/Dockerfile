# ---------- STAGE 1 : build NBIS ----------
FROM ubuntu:22.04 AS nbis-builder

ARG DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential make g++ gcc unzip ca-certificates wget \
    zlib1g-dev libpng-dev libjpeg-turbo8-dev libtiff-dev \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /opt

# Option A: si tu places nbis.zip à la racine du repo, on la copie
# (si tu n'as pas nbis.zip localement, commente ces 2 lignes et utilise l'option B)
ADD nbis.zip /opt/nbis.zip

# Option B: sinon, passe une URL au build : --build-arg NBIS_ZIP_URL=...
ARG NBIS_ZIP_URL=""
RUN if [ ! -f /opt/nbis.zip ]; then \
      if [ -n "$NBIS_ZIP_URL" ]; then wget -O /opt/nbis.zip "$NBIS_ZIP_URL"; \
      else echo "Fournis nbis.zip (Option A) ou NBIS_ZIP_URL (Option B)" && exit 1; fi; \
    fi

RUN unzip -q /opt/nbis.zip && \
    cd /opt/Rel_5.0.0 && \
    mkdir -p /usr/local/NBIS/Main && \
    ./setup.sh /usr/local/NBIS/Main --64 && \
    make config && make it && make install LIBNBIS=yes

# ---------- STAGE 2 : runtime Python + NBIS ----------
FROM python:3.11-slim AS runtime

# Copie des binaires NBIS
COPY --from=nbis-builder /usr/local/NBIS/Main /usr/local/NBIS/Main
ENV PATH="/usr/local/NBIS/Main/bin:${PATH}"

# Deps système minimes (si besoin de libjpeg/libtiff au runtime)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libjpeg-turbo-progs libtiff5 && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copie du code
COPY nbis_mindtct_extractor.py /app/nbis_mindtct_extractor.py
COPY scripts /app/scripts
COPY data /app/data  # facultatif, juste pour avoir un exemple

# Script par défaut (peut être changé à l'exécution)
ENTRYPOINT ["python", "/app/scripts/run_extract.py"]
