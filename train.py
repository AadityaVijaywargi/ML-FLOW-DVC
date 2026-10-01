import subprocess
import pandas as pd
import mlflow
import mlflow.sklearn

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score


# Load dataset
df = pd.read_csv("data/iris.csv")

X = df.drop("target", axis=1)
y = df["target"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Get DVC dataset version from the .dvc file
with open("data/iris.csv.dvc", "r") as f:
    dvc_file = f.read()

dvc_version = dvc_file.split("md5:")[1].split()[0]

# Set MLflow experiment
mlflow.set_experiment("DVC_MLflow_Project")

with mlflow.start_run():

    # Model parameter
    n_estimators = 100

    model = RandomForestClassifier(
        n_estimators=n_estimators,
        random_state=42
    )

    model.fit(X_train, y_train)

    # Evaluate model
    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)

    # Log DVC version and model parameters
    mlflow.log_param("dataset_version", dvc_version)
    mlflow.log_param("n_estimators", n_estimators)

    # Log metric
    mlflow.log_metric("accuracy", accuracy)

    mlflow.sklearn.log_model(
    model,
    "random_forest_model",
    skops_trusted_types=["sklearn.tree._tree.Tree"]
)

    print("DVC dataset version:", dvc_version)
    print("Accuracy:", accuracy)