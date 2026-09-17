# Pasta Classifier

## Toy example to learn to deploy a real model using MLFlow

Synthetic 4-class image dataset (spaghetti / tagliatelle / fusilli / penne),
generated procedurally, trained with a small CNN in PyTorch. MLflow tracking
is left out for the future.

DagsHub project (data + MLflow tracking): https://dagshub.com/JES0406/pasta_classifier

## Clone / fork

```
git clone https://dagshub.com/JES0406/pasta_classifier.git
cd pasta_classifier
dvc pull
```

Or fork on DagsHub first (top-right "Fork" on the project page), then clone
your fork the same way.

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
# 1. Generate the synthetic dataset -> data/<class>/*.png
python generate_pasta.py

# 2. Train the CNN
python train.py --epochs 10 --lr 1e-3 --batch-size 16
```

`train.py` logs params/metrics/model to MLflow. Exact run command (tracking
URI, DagsHub credentials, etc.) still being adapted — TODO once settled.

## Dependencies

```
numpy==1.26.4
torch
pillow
dvc==3.32.0
mlflow
```
