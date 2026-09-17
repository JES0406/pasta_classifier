# Pasta Classifier

## Toy example to learn to deploy a real model using MLFlow

Synthetic 4-class image dataset (spaghetti / tagliatelle / fusilli / penne),
generated procedurally, trained with a small CNN in PyTorch. MLflow tracking
is left out for the future.

## Install

Using venv

```
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Using conda

```
conda create -n pasta-classifier python=3.11
conda activate pasta-classifier
pip install -r requirements.txt
```

Using uv

```
uv venv
source .venv/bin/activate
uv pip install -r requirements.txt
```

## Run the flow

```
# 1. Generate the synthetic dataset -> data/pasta/<class>/*.png
python generate_pasta.py

# 2. Train the CNN
python train.py --epochs 10 --lr 1e-3 --batch-size 16
```

`train.py` currently just prints train loss / val accuracy per epoch.

## Dependencies

```
numpy==1.26.4
torch
pillow
dvc==3.32.0
mlflow
```
