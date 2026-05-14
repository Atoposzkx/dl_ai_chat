# Legacy TensorFlow 1.x Run Guide

This repository is a **legacy TensorFlow 1.x** Chinese Seq2Seq chatbot project.

## Runtime Compatibility

- Recommended Python version: **3.7**
- Recommended TensorFlow version: **1.15.5**
- If `tensorflow==1.15.5` cannot be installed in your local environment, try fallback:
  - `tensorflow==1.15.0`

## Entry Points

- `train.py` is the **training entry point**.
- `pred.py` is the **inference entry point**.

## Data and Model Directories

- `model/` contains TensorFlow checkpoint files used for restore/training continuation and prediction.
- `data/` contains generated vocabulary and NumPy data files (such as `.pkl` and `.npy`).

## Create Environment (Conda)

```bash
conda env create -f environment.yml
conda activate dl_ai_chat_tf1
```

## Install Dependencies (pip alternative)

If you prefer pip + virtualenv instead of conda:

```bash
python3.7 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

## Preprocessing

Run data preprocessing to generate vocabulary/data artifacts:

```bash
python data_processing_ch.py
```

## Train

```bash
python train.py
```

## Predict

```bash
python pred.py
```

## Run in GitHub Codespaces / Dev Container

This project can be run fully in a cloud dev environment (for example, GitHub Codespaces)
using the included `.devcontainer/` configuration.

After the container is created, verify runtime and TensorFlow:

```bash
python --version
python -c "import tensorflow as tf; print(tf.__version__)"
```

Run prediction with existing `data/` and `model/` artifacts:

```bash
python pred.py
```
