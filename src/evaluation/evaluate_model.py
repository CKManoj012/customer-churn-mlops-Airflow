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
    roc_auc_score,
    average_precision_score,
    precision_score,
    recall_score,
    f1_score,
    accuracy_score,
)


REGION = "ap-south-1"
PROFILE = "churn-dev"

BUCKET = "customer-churn-mlops-dev-558311101304-ap-south-1"

TRACKING_ARN = (
    "arn:aws:sagemaker:ap-south-1:558311101304:"
    "mlflow-tracking-server/customer-churn-mlflow-dev"
)

EXPERIMENT_NAME = "Customer_Churn"

WORK_DIR = Path("artifacts/evaluation")

MODEL_TAR = WORK_DIR / "model.tar.gz"
MODEL_DIR = WORK_DIR / "model"
TEST_FILE = WORK_DIR / "test.csv"
RESULTS_FILE = WORK_DIR / "evaluation.json"


THRESHOLD_FILE = Path(
    "artifacts/threshold_tuning/best_threshold.json"
)

def get_latest_completed_training_job(sm_client):

    response = sm_client.list_training_jobs(
        SortBy="CreationTime",
        SortOrder="Descending",
        MaxResults=20,
    )

    for job in response["TrainingJobSummaries"]:

        if job["TrainingJobStatus"] == "Completed":
            return job["TrainingJobName"]

    raise RuntimeError(
        "No completed SageMaker training job found."
    )


def split_s3_uri(uri):

    path = uri.replace("s3://", "")

    bucket, key = path.split("/", 1)

    return bucket, key


def download_model(s3_client, model_uri):

    bucket, key = split_s3_uri(model_uri)

    print(f"Downloading model: {model_uri}")

    s3_client.download_file(
        bucket,
        key,
        str(MODEL_TAR),
    )


def download_test_data(s3_client):

    key = "training/test/test.csv"

    print(
        f"Downloading test data: "
        f"s3://{BUCKET}/{key}"
    )

    s3_client.download_file(
        BUCKET,
        key,
        str(TEST_FILE),
    )

def load_prediction_threshold():

    if not THRESHOLD_FILE.exists():
        raise FileNotFoundError(
            f"Threshold file not found: {THRESHOLD_FILE}"
        )

    with open(
        THRESHOLD_FILE,
        "r"
    ) as f:

        data = json.load(f)

    return float(
        data["threshold"]
    )

def extract_model():

    MODEL_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    with tarfile.open(
        MODEL_TAR,
        "r:gz"
    ) as tar:

        tar.extractall(
            MODEL_DIR
        )


def load_xgboost_model():

    files = list(
        MODEL_DIR.rglob("*")
    )

    files = [
        file
        for file in files
        if file.is_file()
    ]

    if not files:
        raise RuntimeError(
            "No model files found after extraction."
        )

    print("\nExtracted model files:")

    for file in files:
        print(file)

    # SageMaker XGBoost commonly stores a pickled
    # Booster/model object inside model.tar.gz.
    for file in files:

        try:

            with open(
                file,
                "rb"
            ) as f:

                model = pickle.load(f)

            print(
                f"\nLoaded model from: {file}"
            )

            return model

        except Exception:
            continue

    # Fallback: try native XGBoost load_model
    for file in files:

        try:

            model = xgb.Booster()

            model.load_model(
                str(file)
            )

            print(
                f"\nLoaded XGBoost Booster from: {file}"
            )

            return model

        except Exception:
            continue

    raise RuntimeError(
        "Unable to load model from model.tar.gz"
    )


def evaluate(model):

    # Our Processing Job created:
    #
    # column 0 = churn_value
    # remaining columns = features
    # no header

    test_df = pd.read_csv(
        TEST_FILE,
        header=None
    )

    y_test = test_df.iloc[:, 0]

    X_test = test_df.iloc[:, 1:]

    dtest = xgb.DMatrix(
        X_test
    )

    probabilities = model.predict(
        dtest
    )

    threshold = load_prediction_threshold()

    predictions = (
        probabilities >= threshold
    ).astype(int)

    metrics = {
        "test_roc_auc":
            roc_auc_score(
                y_test,
                probabilities
            ),

        "test_pr_auc":
            average_precision_score(
                y_test,
                probabilities
            ),

        "test_precision":
            precision_score(
                y_test,
                predictions,
                zero_division=0,
            ),

        "test_recall":
            recall_score(
                y_test,
                predictions,
                zero_division=0,
            ),

        "test_f1":
            f1_score(
                y_test,
                predictions,
                zero_division=0,
            ),

        "test_accuracy":
            accuracy_score(
                y_test,
                predictions
            ),
    }

    return metrics, threshold


def apply_quality_gate(metrics):

    roc_auc_pass = (
        metrics["test_roc_auc"] >= 0.82
    )

    recall_pass = (
        metrics["test_recall"] >= 0.70
    )

    passed = (
        roc_auc_pass
        and recall_pass
    )

    return passed


def log_to_mlflow(
    training_job_name,
    metrics,
    quality_gate_passed,
    threshold
):

    mlflow.set_tracking_uri(
        TRACKING_ARN
    )

    mlflow.set_experiment(
        EXPERIMENT_NAME
    )

    with mlflow.start_run(
        run_name=(
            f"{training_job_name}-evaluation"
        )
    ):

        mlflow.log_param(
            "training_job_name",
            training_job_name
        )


        mlflow.log_param(
            "prediction_threshold",
            threshold
        )

        mlflow.log_param(
            "quality_gate_roc_auc",
            0.82
        )

        mlflow.log_param(
            "quality_gate_recall",
            0.70
        )

        mlflow.log_metrics(
            metrics
        )

        mlflow.set_tag(
            "pipeline_stage",
            "evaluation"
        )

        mlflow.set_tag(
            "quality_gate",
            (
                "PASS"
                if quality_gate_passed
                else "FAIL"
            )
        )


def main():

    WORK_DIR.mkdir(
        parents=True,
        exist_ok=True
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
        get_latest_completed_training_job(
            sm_client
        )
    )

    details = (
        sm_client.describe_training_job(
            TrainingJobName=training_job_name
        )
    )

    model_uri = (
        details["ModelArtifacts"]
        ["S3ModelArtifacts"]
    )

    print(
        f"Training job: {training_job_name}"
    )

    print(
        f"Model artifact: {model_uri}"
    )

    download_model(
        s3_client,
        model_uri
    )

    download_test_data(
        s3_client
    )

    extract_model()

    model = load_xgboost_model()

    metrics, threshold = evaluate(model)

    print(
        f"\nPrediction threshold: {threshold:.2f}"
    )

    print("\nEvaluation Metrics")
    print("-" * 40)

    for name, value in metrics.items():

        print(
            f"{name}: {value:.4f}"
        )

    quality_gate_passed = (
        apply_quality_gate(
            metrics
        )
    )

    print("\nQuality Gate")

    if quality_gate_passed:

        print("PASS")

    else:

        print("FAIL")

    results = {
        "training_job_name":
            training_job_name,

        "model_artifact_uri":
            model_uri,

        "metrics":
            metrics,

        "quality_gate":
            (
                "PASS"
                if quality_gate_passed
                else "FAIL"
            ),
    }

    with open(
        RESULTS_FILE,
        "w"
    ) as f:

        json.dump(
            results,
            f,
            indent=4
        )

    log_to_mlflow(
        training_job_name,
        metrics,
        quality_gate_passed,
        threshold
    )

    print(
        "\nEvaluation logged to MLflow."
    )

    print(
        f"Results saved to: "
        f"{RESULTS_FILE}"
    )


if __name__ == "__main__":
    main()