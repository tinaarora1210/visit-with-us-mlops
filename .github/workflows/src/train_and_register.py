import os
import json
import joblib
import pandas as pd
import numpy as np

from sklearn.model_selection import GridSearchCV
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix
)
from sklearn.ensemble import RandomForestClassifier

import mlflow
import mlflow.sklearn
from huggingface_hub import hf_hub_download, HfApi


# ----------------------------
# Configuration
# ----------------------------
HF_DATASET_REPO = "tina1210/visit-with-us-wellness-data"
HF_MODEL_REPO = "tina1210/visit-with-us-wellness-model"
EXPERIMENT_NAME = "visit_with_us_wellness_rf"

TARGET = "ProdTaken"


# ----------------------------
# Load data from Hugging Face
# ----------------------------
def load_data():
    train_path = hf_hub_download(
        repo_id=HF_DATASET_REPO,
        filename="processed/train.csv",
        repo_type="dataset"
    )
    test_path = hf_hub_download(
        repo_id=HF_DATASET_REPO,
        filename="processed/test.csv",
        repo_type="dataset"
    )

    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    return train_df, test_df


# ----------------------------
# Train model with MLflow
# ----------------------------
def train_and_evaluate():
    train_df, test_df = load_data()

    X_train = train_df.drop(columns=[TARGET])
    y_train = train_df[TARGET]

    X_test = test_df.drop(columns=[TARGET])
    y_test = test_df[TARGET]

    categorical_cols = X_train.select_dtypes(include="object").columns.tolist()
    numerical_cols = X_train.select_dtypes(exclude="object").columns.tolist()

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numerical_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols),
        ]
    )

    model = RandomForestClassifier(random_state=42)

    pipeline = Pipeline(
        steps=[
            ("preprocess", preprocessor),
            ("clf", model)
        ]
    )

    param_grid = {
        "clf__n_estimators": [100, 200],
        "clf__max_depth": [None, 10],
        "clf__min_samples_split": [2, 5],
        "clf__min_samples_leaf": [1, 2],
    }

    mlflow.set_experiment(EXPERIMENT_NAME)

    with mlflow.start_run():
        grid = GridSearchCV(
            pipeline,
            param_grid=param_grid,
            cv=5,
            scoring="roc_auc",
            n_jobs=-1
        )
        grid.fit(X_train, y_train)

        best_model = grid.best_estimator_

        y_pred = best_model.predict(X_test)
        y_prob = best_model.predict_proba(X_test)[:, 1]

        metrics = {
            "accuracy": accuracy_score(y_test, y_pred),
            "precision": precision_score(y_test, y_pred),
            "recall": recall_score(y_test, y_pred),
            "f1": f1_score(y_test, y_pred),
            "roc_auc": roc_auc_score(y_test, y_prob),
            "pr_auc": average_precision_score(y_test, y_prob),
        }

        for k, v in metrics.items():
            mlflow.log_metric(k, v)

        mlflow.log_params(grid.best_params_)

        os.makedirs("package", exist_ok=True)
        model_path = "package/best_model.joblib"
        joblib.dump(best_model, model_path)

        mlflow.sklearn.log_model(best_model, "model")

        with open("package/metrics.json", "w") as f:
            json.dump(metrics, f, indent=2)

        with open("package/confusion_matrix.json", "w") as f:
            json.dump(confusion_matrix(y_test, y_pred).tolist(), f)

        return model_path


# ----------------------------
# Upload model to Hugging Face
# ----------------------------
def upload_model(model_path):
    api = HfApi()
    api.create_repo(repo_id=HF_MODEL_REPO, exist_ok=True)

    api.upload_file(
        path_or_fileobj=model_path,
        path_in_repo="best_model.joblib",
        repo_id=HF_MODEL_REPO,
        repo_type="model"
    )


if __name__ == "__main__":
    model_file = train_and_evaluate()
    upload_model(model_file)
