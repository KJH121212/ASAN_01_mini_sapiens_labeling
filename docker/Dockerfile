# ================================================================
# Dockerfile.sapiens
# PyTorch 2.1.0 + CUDA 11.8 + mmpose/sapiens 환경
# ================================================================
FROM pytorch/pytorch:2.1.0-cuda11.8-cudnn8-runtime

ARG DEBIAN_FRONTEND=noninteractive
ENV TZ=Asia/Seoul
ENV PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
WORKDIR /workspace

# ------------------------------------------------------------
# 1️⃣  System dependencies
# ------------------------------------------------------------
RUN sed -i 's|http://[a-z]\+.ubuntu.com|https://mirror.kakao.com|g' /etc/apt/sources.list && \
    apt-get update && apt-get install -y --no-install-recommends \
        git curl wget tzdata ca-certificates \
        build-essential pkg-config \
        libgl1-mesa-glx libglib2.0-0 libgtk2.0-dev \
        ffmpeg python3-pip && \
    rm -rf /var/lib/apt/lists/*

# ------------------------------------------------------------
# 2️⃣  Locale settings (UTF-8)
# ------------------------------------------------------------
RUN apt-get update && apt-get install -y locales && \
    echo "ko_KR.UTF-8 UTF-8" >> /etc/locale.gen && \
    echo "en_US.UTF-8 UTF-8" >> /etc/locale.gen && \
    locale-gen && \
    update-locale LANG=ko_KR.UTF-8 LC_ALL=ko_KR.UTF-8

ENV LANG=ko_KR.UTF-8
ENV LANGUAGE=ko_KR:ko
ENV LC_ALL=ko_KR.UTF-8

# ------------------------------------------------------------
# 3️⃣  Pre-install core Python tools
# ------------------------------------------------------------
RUN python3 -m pip install --upgrade pip setuptools wheel --no-cache-dir

# ------------------------------------------------------------
# 4️⃣  Install chumpy manually (to prevent pip build error)
# ------------------------------------------------------------
RUN git clone https://github.com/mattloper/chumpy.git /tmp/chumpy && \
    cd /tmp/chumpy && python3 setup.py install && \
    rm -rf /tmp/chumpy

# ------------------------------------------------------------
# 5️⃣  Install main Python dependencies
# ------------------------------------------------------------
COPY docker/requirements.txt /tmp/requirements.txt

RUN python -m pip install --no-cache-dir -r /tmp/requirements.txt \
    --find-links https://download.openmmlab.com/mmcv/dist/cu118/torch2.1/index.html \
    --trusted-host pypi.org \
    --trusted-host files.pythonhosted.org \
    --trusted-host download.openmmlab.com

# ------------------------------------------------------------
# 6️⃣  Entry point
# ------------------------------------------------------------
CMD ["/bin/bash"]
