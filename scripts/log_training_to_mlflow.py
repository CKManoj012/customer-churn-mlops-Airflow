import os
import subprocess

import boto3
import mlflow


REGION = "ap-south-1"

TRACKING_ARN = (
    "arn:aws:sagemaker:ap-south-1:558311101304:"
    "mlflow-tracking-server/customer-churn-mlflow-dev"
)

EXPERIMENT_NAME = "Customer_Churn"


# --------------------------------------------------
# Make AWS credentials available to MLflow plugin
# --------------------------------------------------

os.environ["AWS_PROFILE"] = "churn-dev"
os.environ["AWS_DEFAULT_REGION"] = REGION
os.environ["AWS_REGION"] = REGION


# --------------------------------------------------
# AWS session
# --------------------------------------------------

session = boto3.Session(
    profile_name="churn-dev",
    region_name=REGION,
)

sm = session.client("sagemaker")


# --------------------------------------------------
# Find latest completed training job
# --------------------------------------------------

jobs = sm.list_training_jobs(
    SortBy="CreationTime",
    SortOrder="Descending",
    MaxResults=10,
)["TrainingJobSummaries"]

completed_jobs = [
    job
    for job in jobs
    if job["TrainingJobStatus"] == "Completed"
]

if not completed_jobs:
    raise RuntimeError(
        "No completed SageMaker training job found."
    )

training_job_name = completed_jobs[0]["TrainingJobName"]

details = sm.describe_training_job(
    TrainingJobName=training_job_name
)


# --------------------------------------------------
# Extract metadata
# --------------------------------------------------

hyperparameters = details.get(
    "HyperParameters",
    {}
)

metrics = {
    item["MetricName"]: item["Value"]
    for item in details.get(
        "FinalMetricDataList",
        []
    )
}

model_artifact = (
    details["ModelArtifacts"]["S3ModelArtifacts"]
)

train_s3 = (
    details["InputDataConfig"][0]
    ["DataSource"]["S3DataSource"]["S3Uri"]
)

validation_s3 = (
    details["InputDataConfig"][1]
    ["DataSource"]["S3DataSource"]["S3Uri"]
)


# --------------------------------------------------
# Git commit
# --------------------------------------------------

try:
    git_commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"],
        text=True,
    ).strip()

except Exception:
    git_commit = "unknown"


# --------------------------------------------------
# Connect to managed MLflow
# --------------------------------------------------

mlflow.set_tracking_uri(
    TRACKING_ARN
)

mlflow.set_experiment(
    EXPERIMENT_NAME
)


# --------------------------------------------------
# Log experiment run
# --------------------------------------------------

with mlflow.start_run(
    run_name=training_job_name
):

    mlflow.log_param(
        "training_job_name",
        training_job_name
    )

    mlflow.log_param(
        "model_type",
        "XGBoost"
    )

    mlflow.log_param(
        "train_s3_uri",
        train_s3
    )

    mlflow.log_param(
        "validation_s3_uri",
        validation_s3
    )

    mlflow.log_param(
        "model_artifact_uri",
        model_artifact
    )

    mlflow.log_param(
        "git_commit",
        git_commit
    )

    # XGBoost hyperparameters
    for key, value in hyperparameters.items():
        mlflow.log_param(
            key,
            value
        )

    # SageMaker-reported metrics
    for name, value in metrics.items():

        # MLflow metric names are easier to work with
        # if ":" is avoided.
        metric_name = (
            name
            .replace(":", "_")
            .replace("-", "_")
        )

        mlflow.log_metric(
            metric_name,
            float(value)
        )

    mlflow.set_tag(
        "aws_region",
        REGION
    )

    mlflow.set_tag(
        "pipeline_stage",
        "training"
    )

    mlflow.set_tag(
        "framework",
        "sagemaker-xgboost"
    )


print("MLflow logging completed.")
print(f"Experiment: {EXPERIMENT_NAME}")
print(f"Training job: {training_job_name}")
print(f"Model artifact: {model_artifact}")