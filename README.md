# Pasta Classifier

MLOps toy project: a small PyTorch CNN classifying 4 synthetic pasta shapes
(spaghetti / tagliatelle / fusilli / penne), with the full pipeline —
data versioning (DVC), remote storage (DagsHub S3), and experiment tracking
(MLflow on DagsHub) — wired around it.

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
python train.py
```

`train.py` calls `dagshub.init(repo_owner="JES0406", repo_name="pasta_classifier", mlflow=True)`,
so params/metrics/model get logged to this project's MLflow tab on DagsHub
(browser-based GitHub OAuth on first run, no manual token/`.env` needed).

Training args (all optional):

| Flag | Default | Meaning |
|---|---|---|
| `--epochs` | `10` | Training epochs |
| `--lr` | `1e-3` | Adam learning rate |
| `--batch-size` | `16` | Batch size for train/val loaders |

Example: `python train.py --epochs 20 --lr 1e-4 --batch-size 32`

## Project structure

```
.
├── generate_pasta.py   # builds the synthetic dataset -> data/<class>/*.png
├── train.py             # CNN + training loop, MLflow logging via DagsHub
├── data/                # dataset (DVC-tracked, pulled from DagsHub S3)
├── data.dvc             # DVC pointer file for data/
├── requirements.txt
└── docs/                # assignment write-up + process screenshots
```

## Dependencies

```
numpy==1.26.4
torch
pillow
dvc[s3]>=3.60
mlflow>=3.0
dagshub==0.7.2
```

## License

[MIT](LICENSE)

## Author

Javier Escobar Serrano — [@JES0406](https://github.com/JES0406)
