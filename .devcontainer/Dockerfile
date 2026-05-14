FROM python:3.7-slim-buster

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        git \
        build-essential \
    && rm -rf /var/lib/apt/lists/*

RUN python -m pip install --upgrade "pip<24" setuptools wheel

# Install legacy runtime dependencies (CPU TensorFlow 1.x).
# If tensorflow==1.15.5 is unavailable on a given mirror/platform,
# fallback to tensorflow==1.15.0.
RUN pip install \
    tensorflow==1.15.5 \
    numpy==1.19.5 \
    jieba==0.42.1 \
    nltk==3.6.7

WORKDIR /workspace
