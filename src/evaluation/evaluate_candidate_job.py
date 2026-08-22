import json
import os
import tarfile
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb

from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


MODEL_DIR = Path("/opt/ml/processing/model")
VALIDATION_DIR = Path("/opt/ml/processing/validation")
TEST_DIR = Path("/opt/ml/processing/test")
OUTPUT_DIR = Path("/opt/ml/processing/evaluation")

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# --------------------------------------------------
# Quality gate
# --------------------------------------------------

MIN_ROC_AUC = 0.85
MIN_RECALL = 0.70
MIN_F1 = 0.65


def find_file(directory, suffix):
    files = list(
        directory.rglob(f"*{suffix}")
    )

    if not files:
        raise FileNotFoundError(
            f"No {suffix} file found in {directory}"
        )

    return files[0]


def load_dataset(directory):

    csv_file = find_file(
        directory,
        ".csv",
    )

    print(f"Loading: {csv_file}")

    df = pd.read_csv(
        csv_file,
        header=None,
    )

    # Training format:
    # first column = target
    # remaining columns = features

    y = df.iloc[:, 0].astype(int)

    X = df.iloc[:, 1:]

    return X, y


def load_model():

    model_tar = find_file(
        MODEL_DIR,
        ".tar.gz",
    )

    extract_dir = MODEL_DIR / "extracted"

    extract_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    print(
        f"Extracting model: {model_tar}"
    )

    with tarfile.open(
        model_tar,
        "r:gz",
    ) as tar:

        tar.extractall(
            extract_dir
        )

    model_candidates = list(
        extract_dir.rglob(
            "xgboost-model"
        )
    )

    if not model_candidates:

        raise FileNotFoundError(
            "xgboost-model not found "
            "inside model.tar.gz"
        )

    model_path = model_candidates[0]

    print(
        f"Loading XGBoost model: "
        f"{model_path}"
    )

    booster = xgb.Booster()

    booster.load_model(
        str(model_path)
    )

    return booster


def predict(
    model,
    X,
):

    dmatrix = xgb.DMatrix(
        X
    )

    return model.predict(
        dmatrix
    )


def find_best_threshold(
    y_true,
    probabilities,
):

    best_threshold = 0.5
    best_f1 = -1

    # Test thresholds from 0.10 → 0.90

    for threshold in np.arange(
        0.10,
        0.91,
        0.01,
    ):

        predictions = (
            probabilities >= threshold
        ).astype(int)

        score = f1_score(
            y_true,
            predictions,
            zero_division=0,
        )

        if score > best_f1:

            best_f1 = score
            best_threshold = threshold

    return round(
        float(best_threshold),
        2,
    )


def calculate_metrics(
    y_true,
    probabilities,
    threshold,
):

    predictions = (
        probabilities >= threshold
    ).astype(int)

    return {
        "roc_auc": float(
            roc_auc_score(
                y_true,
                probabilities,
            )
        ),

        "pr_auc": float(
            average_precision_score(
                y_true,
                probabilities,
            )
        ),

        "precision": float(
            precision_score(
                y_true,
                predictions,
                zero_division=0,
            )
        ),

        "recall": float(
            recall_score(
                y_true,
                predictions,
                zero_division=0,
            )
        ),

        "f1": float(
            f1_score(
                y_true,
                predictions,
                zero_division=0,
            )
        ),

        "accuracy": float(
            accuracy_score(
                y_true,
                predictions,
            )
        ),
    }


def main():

    print(
        "Starting candidate model evaluation..."
    )

    # ----------------------------------------------
    # Load model
    # ----------------------------------------------

    model = load_model()

    # ----------------------------------------------
    # Validation
    # ----------------------------------------------

    X_val, y_val = load_dataset(
        VALIDATION_DIR
    )

    val_probabilities = predict(
        model,
        X_val,
    )

    best_threshold = (
        find_best_threshold(
            y_val,
            val_probabilities,
        )
    )

    print(
        f"Best validation threshold: "
        f"{best_threshold}"
    )

    # ----------------------------------------------
    # Test
    # ----------------------------------------------

    X_test, y_test = load_dataset(
        TEST_DIR
    )

    test_probabilities = predict(
        model,
        X_test,
    )

    metrics = calculate_metrics(
        y_test,
        test_probabilities,
        best_threshold,
    )

    # ----------------------------------------------
    # Quality gate
    # ----------------------------------------------

    passed = (
        metrics["roc_auc"]
        >= MIN_ROC_AUC
        and
        metrics["recall"]
        >= MIN_RECALL
        and
        metrics["f1"]
        >= MIN_F1
    )

    quality_gate = (
        "PASS"
        if passed
        else "FAIL"
    )

    result = {
        "threshold": best_threshold,

        "metrics": metrics,

        "quality_gate": quality_gate,

        "quality_gate_rules": {
            "min_roc_auc":
                MIN_ROC_AUC,

            "min_recall":
                MIN_RECALL,

            "min_f1":
                MIN_F1,
        },
    }

    output_file = (
        OUTPUT_DIR /
        "evaluation.json"
    )

    with open(
        output_file,
        "w",
    ) as f:

        json.dump(
            result,
            f,
            indent=4,
        )

    print("\nEvaluation results")
    print(
        json.dumps(
            result,
            indent=4,
        )
    )

    print(
        f"\nSaved to: {output_file}"
    )


if __name__ == "__main__":
    main()