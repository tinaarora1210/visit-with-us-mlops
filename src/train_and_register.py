# src/train_and_register.py

import os
import json
import joblib
import pandas as pd
import mlflow
import mlflow.sklearn

from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, confusion_matrix
)
from sklearn.ensemble import RandomForestClassifier

from huggingface_hub import hf_hub_download, create_repo, upload_folder, whoami


# -----------------------------
# Config (Edit only these if needed)
# -----------------------------
DATASET_REPO_ID = os.environ.get("HF_DATASET_REPO", "tina1210/visit-with-us-wellness-data")
HF_MODEL_REPO_NAME = os.environ.get("HF_MODEL_REPO_NAME", "visit-with-us-wellness-model")
TARGET_COL = "ProdTaken"
RANDOM_STATE = 42

ARTIFACTS_DIR = os.path.join(os.getcwd(), "artifacts")
os.makedirs(ARTIFACTS_DIR, exist_ok=True)


def main():
    # -----------------------------
    # Step 1: Load train/test from HF dataset space
    # -----------------------------
    train_path = hf_hub_download(
        repo_id=DATASET_REPO_ID,
        repo_type="dataset",
        filename="processed/train.csv"
    )
    test_path = hf_hub_download(
        repo_id=DATASET_REPO_ID,
        repo_type="dataset",
        filename="processed/test.csv"
    )

    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    print("Loaded train/test from Hugging Face dataset space.")
    print("Train shape:", train_df.shape, "Test shape:", test_df.shape)

    X_train = train_df.drop(columns=[TARGET_COL])
    y_train = train_df[TARGET_COL]
    X_test = test_df.drop(columns=[TARGET_COL])
    y_test = test_df[TARGET_COL]

    # -----------------------------
    # Step 2: Preprocessing pipeline
    # -----------------------------
    numeric_features = X_train.select_dtypes(include=["int64", "float64", "int32", "float32"]).columns.tolist()
    categorical_features = X_train.select_dtypes(exclude=["int64", "float64", "int32", "float32"]).columns.tolist()

    numeric_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median"))
    ])

    categorical_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore"))
    ])

    preprocess = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ],
        remainder="drop"
    )

    # -----------------------------
    # Step 3: Model + grid
    # -----------------------------
    rf = RandomForestClassifier(
        random_state=RANDOM_STATE,
        n_jobs=-1,
        class_weight="balanced"
    )

    model = Pipeline(steps=[
        ("preprocess", preprocess),
        ("clf", rf)
    ])

    param_grid = {
        "clf__n_estimators": [200, 400],
        "clf__max_depth": [None, 10, 20],
        "clf__min_samples_split": [2, 5],
        "clf__min_samples_leaf": [1, 2]
    }

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

    grid = GridSearchCV(
        estimator=model,
        param_grid=param_grid,
        scoring="f1",
        cv=cv,
        n_jobs=-1,
        verbose=1
    )

    # -----------------------------
    # Step 4: MLflow tracking (local folder)
    # -----------------------------
    mlruns_dir = os.path.join(os.getcwd(), "mlruns")
    mlflow.set_tracking_uri(f"file:{mlruns_dir}")
    mlflow.set_experiment("visit_with_us_wellness_rf")

    with mlflow.start_run(run_name="rf_gridsearch"):
        grid.fit(X_train, y_train)
        best_model = grid.best_estimator_
        best_params = grid.best_params_

        mlflow.log_params(best_params)

        y_pred = best_model.predict(X_test)
        y_proba = best_model.predict_proba(X_test)[:, 1]

        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, zero_division=0)
        rec = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        roc = roc_auc_score(y_test, y_proba)
        pr_auc = average_precision_score(y_test, y_proba)
        cm = confusion_matrix(y_test, y_pred)

        mlflow.log_metric("test_accuracy", acc)
        mlflow.log_metric("test_precision", prec)
        mlflow.log_metric("test_recall", rec)
        mlflow.log_metric("test_f1", f1)
        mlflow.log_metric("test_roc_auc", roc)
        mlflow.log_metric("test_pr_auc", pr_auc)

        cm_path = os.path.join(ARTIFACTS_DIR, "confusion_matrix.json")
        with open(cm_path, "w") as f:
            json.dump(cm.tolist(), f, indent=2)
        mlflow.log_artifact(cm_path)

        local_model_path = os.path.join(ARTIFACTS_DIR, "best_model.joblib")
        joblib.dump(best_model, local_model_path)
        mlflow.sklearn.log_model(best_model, artifact_path="model")

        print("\nBest CV params:", best_params)
        print("Test metrics:")
        print("Accuracy:", round(acc, 4))
        print("Precision:", round(prec, 4))
        print("Recall:", round(rec, 4))
        print("F1:", round(f1, 4))
        print("ROC-AUC:", round(roc, 4))
        print("PR-AUC:", round(pr_auc, 4))
        print("Confusion Matrix:\n", cm)

    # -----------------------------
    # Step 5: Register model on HF Model Hub
    # -----------------------------
    username = whoami()["name"]
    MODEL_REPO_ID = f"{username}/{HF_MODEL_REPO_NAME}"
    create_repo(repo_id=MODEL_REPO_ID, repo_type="model", exist_ok=True)

    readme_text = f"""---
tags:
- tabular-classification
- sklearn
- mlops
library_name: scikit-learn
---

# Visit with Us – Wellness Package Purchase Predictor

This repository contains the best-performing trained model pipeline (preprocessing + RandomForestClassifier)
to predict whether a customer will purchase the Wellness Tourism Package (**{TARGET_COL}**).

## Files
- `best_model.joblib`: serialized sklearn Pipeline (preprocess + model)
- `metrics.json`: evaluation metrics on the held-out test set
- `confusion_matrix.json`: confusion matrix on the held-out test set
"""

    metrics = {
        "test_accuracy": float(acc),
        "test_precision": float(prec),
        "test_recall": float(rec),
        "test_f1": float(f1),
        "test_roc_auc": float(roc),
        "test_pr_auc": float(pr_auc),
        "best_params": best_params
    }

    upload_dir = os.path.join(ARTIFACTS_DIR, "hf_model_package")
    os.makedirs(upload_dir, exist_ok=True)

    with open(os.path.join(upload_dir, "README.md"), "w") as f:
        f.write(readme_text)

    joblib.dump(best_model, os.path.join(upload_dir, "best_model.joblib"))

    with open(os.path.join(upload_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    with open(os.path.join(upload_dir, "confusion_matrix.json"), "w") as f:
        json.dump(cm.tolist(), f, indent=2)

    upload_folder(
        repo_id=MODEL_REPO_ID,
        repo_type="model",
        folder_path=upload_dir,
        path_in_repo="."
    )

    print("\nBest model registered on Hugging Face Model Hub.")
    print("Model repo:", MODEL_REPO_ID)
    print("Model URL:", f"https://huggingface.co/{MODEL_REPO_ID}")
    print("MLflow runs folder:", mlruns_dir)


if __name__ == "__main__":
    main()
