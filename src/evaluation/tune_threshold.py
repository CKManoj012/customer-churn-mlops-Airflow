import json
import pickle
import tarfile
from pathlib import Path

import boto3
import mlflow
import numpy as np
import pandas as pd
import xgboost as xgb

from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
)


REGION = "ap-south-1"
PROFILE = "churn-dev"

BUCKET = "customer-churn-mlops-dev-558311101304-ap-south-1"

TRACKING_ARN = (
    "arn:aws:sagemaker:ap-south-1:558311101304:"
    "mlflow-tracking-server/customer-churn-mlflow-dev"
)

EXPERIMENT_NAME = "Customer_Churn"

WORK_DIR = Path("artifacts/threshold_tuning")

MODEL_TAR = WORK_DIR / "model.tar.gz"
MODEL_DIR = WORK_DIR / "model"

VALIDATION_FILE = WORK_DIR / "validation.csv"

THRESHOLD_FILE = (
    WORK_DIR / "best_threshold.json"
)


def split_s3_uri(uri):

    uri = uri.replace(
        "s3://",
        ""
    )

    bucket, key = uri.split(
        "/",
        1
    )

    return bucket, key


def get_latest_training_job(sm_client):

    response = sm_client.list_training_jobs(
        SortBy="CreationTime",
        SortOrder="Descending",
        MaxResults=20,
    )

    for job in response[
        "TrainingJobSummaries"
    ]:

        if (
            job["TrainingJobStatus"]
            == "Completed"
        ):
            return job["TrainingJobName"]

    raise RuntimeError(
        "No completed training job found."
    )


def download_files(
    sm_client,
    s3_client,
    training_job_name,
):

    details = (
        sm_client.describe_training_job(
            TrainingJobName=training_job_name
        )
    )

    model_uri = (
        details["ModelArtifacts"]
        ["S3ModelArtifacts"]
    )

    model_bucket, model_key = (
        split_s3_uri(model_uri)
    )

    s3_client.download_file(
        model_bucket,
        model_key,
        str(MODEL_TAR),
    )

    s3_client.download_file(
        BUCKET,
        "training/validation/validation.csv",
        str(VALIDATION_FILE),
    )

    return model_uri


def load_model():

    MODEL_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    with tarfile.open(
        MODEL_TAR,
        "r:gz",
    ) as tar:

        tar.extractall(
            MODEL_DIR
        )

    files = [
        file
        for file
        in MODEL_DIR.rglob("*")
        if file.is_file()
    ]

    for file in files:

        try:

            with open(
                file,
                "rb"
            ) as f:

                model = pickle.load(f)

            return model

        except Exception:
            pass

    for file in files:

        try:

            model = xgb.Booster()

            model.load_model(
                str(file)
            )

            return model

        except Exception:
            pass

    raise RuntimeError(
        "Unable to load XGBoost model."
    )


def find_best_threshold(
    model,
):

    df = pd.read_csv(
        VALIDATION_FILE,
        header=None,
    )

    y = df.iloc[:, 0]

    X = df.iloc[:, 1:]

    dmatrix = xgb.DMatrix(
        X
    )

    probabilities = model.predict(
        dmatrix
    )

    results = []

    thresholds = np.arange(
        0.10,
        0.71,
        0.01,
    )

    for threshold in thresholds:

        predictions = (
            probabilities >= threshold
        ).astype(int)

        precision = precision_score(
            y,
            predictions,
            zero_division=0,
        )

        recall = recall_score(
            y,
            predictions,
            zero_division=0,
        )

        f1 = f1_score(
            y,
            predictions,
            zero_division=0,
        )

        results.append(
            {
                "threshold":
                    float(threshold),

                "precision":
                    float(precision),

                "recall":
                    float(recall),

                "f1":
                    float(f1),
            }
        )

    result_df = pd.DataFrame(
        results
    )

    print(
        result_df.to_string(
            index=False
        )
    )

    # Quality requirement:
    # Recall must be at least 0.70
    candidates = result_df[
        result_df["recall"] >= 0.70
    ]

    if candidates.empty:

        print(
            "\nNo threshold achieved "
            "recall >= 0.70"
        )

        return None, result_df

    # Among thresholds satisfying recall,
    # choose the best F1.
    best = (
        candidates
        .sort_values(
            "f1",
            ascending=False,
        )
        .iloc[0]
    )

    return best, result_df


def log_to_mlflow(
    training_job_name,
    best,
):

    mlflow.set_tracking_uri(
        TRACKING_ARN
    )

    mlflow.set_experiment(
        EXPERIMENT_NAME
    )

    with mlflow.start_run(
        run_name=(
            f"{training_job_name}-"
            "threshold-tuning"
        )
    ):

        mlflow.log_param(
            "training_job_name",
            training_job_name,
        )

        mlflow.log_param(
            "threshold_strategy",
            "best_f1_with_recall_gte_0.70",
        )

        mlflow.log_param(
            "selected_threshold",
            float(
                best["threshold"]
            ),
        )

        mlflow.log_metric(
            "validation_precision",
            float(
                best["precision"]
            ),
        )

        mlflow.log_metric(
            "validation_recall",
            float(
                best["recall"]
            ),
        )

        mlflow.log_metric(
            "validation_f1",
            float(
                best["f1"]
            ),
        )

        mlflow.set_tag(
            "pipeline_stage",
            "threshold_tuning",
        )


def main():

    WORK_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    session = boto3.Session(
        profile_name=PROFILE,
        region_name=REGION,
    )

    sm_client = session.client(
        "sagemaker"
    )

    s3_client = session.client(
        "s3"
    )

    training_job_name = (
        get_latest_training_job(
            sm_client
        )
    )

    print(
        f"Training job: "
        f"{training_job_name}"
    )

    download_files(
        sm_client,
        s3_client,
        training_job_name,
    )

    model = load_model()

    best, results = (
        find_best_threshold(
            model
        )
    )

    results.to_csv(
        WORK_DIR /
        "threshold_results.csv",
        index=False,
    )

    if best is None:

        print(
            "\nThreshold tuning failed "
            "to meet recall requirement."
        )

        return

    print(
        "\nBEST THRESHOLD"
    )

    print(
        f"Threshold: "
        f"{best['threshold']:.2f}"
    )

    print(
        f"Precision: "
        f"{best['precision']:.4f}"
    )

    print(
        f"Recall: "
        f"{best['recall']:.4f}"
    )

    print(
        f"F1: "
        f"{best['f1']:.4f}"
    )

    threshold_result = {
        "threshold":
            float(
                best["threshold"]
            ),

        "validation_precision":
            float(
                best["precision"]
            ),

        "validation_recall":
            float(
                best["recall"]
            ),

        "validation_f1":
            float(
                best["f1"]
            ),
    }

    with open(
        THRESHOLD_FILE,
        "w"
    ) as f:

        json.dump(
            threshold_result,
            f,
            indent=4,
        )

    log_to_mlflow(
        training_job_name,
        best,
    )

    print(
        "\nThreshold tuning "
        "logged to MLflow."
    )


if __name__ == "__main__":
    main()