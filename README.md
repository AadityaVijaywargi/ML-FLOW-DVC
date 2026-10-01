# DVC + MLflow Machine Learning Project

A small MLOps example that combines **DVC** (data versioning) with **MLflow** (experiment tracking) on the Iris dataset.

## What it demonstrates

- **Dataset versioning with DVC** – `data/iris.csv` is tracked by DVC; only the small pointer file `data/iris.csv.dvc` is committed to Git.
- **Experiment tracking with MLflow** – each run logs parameters, metrics and the trained model.
- **DVC ↔ MLflow link** – `train.py` reads the md5 hash from `data/iris.csv.dvc` and logs it as the `dataset_version` parameter, so every MLflow run records exactly which dataset version produced it.

## Project structure

```
ML-FLOW-DVC/
├── .dvc/                 # DVC configuration
├── data/
│   ├── iris.csv          # dataset (tracked by DVC, ignored by Git)
│   └── iris.csv.dvc      # DVC pointer file (tracked by Git)
├── create_dataset.py     # regenerates data/iris.csv
├── train.py              # trains a RandomForest and logs to MLflow
├── requirements.txt
├── .dvcignore
├── .gitignore
└── README.md
```

## Setup

```bash
git clone https://github.com/AadityaVijaywargi/ML-FLOW-DVC.git
cd ML-FLOW-DVC

python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## Get the data

No DVC remote is configured, so the CSV is not downloaded by `dvc pull`. Regenerate it instead:

```bash
python create_dataset.py
```

> Note: `create_dataset.py` recreates the Iris data, but the md5 hash may differ from the one stored in `iris.csv.dvc`. Run `dvc add data/iris.csv` afterwards to update the pointer to your local copy.

## Run training

```bash
python train.py
```

This prints the DVC dataset version and the test accuracy, and creates a new MLflow run.

## View results in MLflow

```bash
mlflow ui
```

Open <http://localhost:5000>, choose the **DVC_MLflow_Project** experiment and compare runs. The `dataset_version` column shows which data version each run used.

## Dataset versioning workflow

```bash
# Version 1
python create_dataset.py
dvc add data/iris.csv
git add data/iris.csv.dvc data/.gitignore
git commit -m "Add v1 of training data"
python train.py                      # run 1 -> dataset_version = hash of v1

# Version 2 (modified data)
python create_dataset.py --version 2
dvc add data/iris.csv
git add data/iris.csv.dvc
git commit -m "Add v2 of training data"
python train.py                      # run 2 -> different dataset_version

# Go back to the previous data version
git checkout HEAD~1 -- data/iris.csv.dvc
dvc checkout
```

Compare the two runs in the MLflow UI to see that each one is linked to a different dataset hash.

> `dvc checkout` restores old data only if it exists in the local DVC cache (or a remote), so run it on the same machine where you ran `dvc add`.

## Tech stack

Python · scikit-learn · pandas · Git · DVC · MLflow
