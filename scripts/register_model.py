import json
from pathlib import Path

import boto3


REGION = "ap-south-1"
PROFILE = "churn-dev"

MODEL_PACKAGE_GROUP = "CustomerChurnModel"

EVALUATION_FILE = Path(
    "artifacts/evaluation/evaluation.json"
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


def load_evaluation():

    if not EVALUATION_FILE.exists():
        raise FileNotFoundError(
            f"Evaluation file not found: {EVALUATION_FILE}"
        )

    with open(EVALUATION_FILE, "r") as f:
        return json.load(f)


def main():

    session = boto3.Session(
        profile_name=PROFILE,
        region_name=REGION,
    )

    sm = session.client("sagemaker")

    training_job_name = (
        get_latest_completed_training_job(sm)
    )

    training_details = sm.describe_training_job(
        TrainingJobName=training_job_name
    )

    model_artifact = (
        training_details["ModelArtifacts"]
        ["S3ModelArtifacts"]
    )

    training_image = (
        training_details["AlgorithmSpecification"]
        ["TrainingImage"]
    )

    evaluation = load_evaluation()

    metrics = evaluation["metrics"]

    quality_gate = evaluation["quality_gate"]

    if quality_gate != "PASS":
        raise RuntimeError(
            "Model did not pass quality gate. "
            "Registration stopped."
        )

    print(f"Training job: {training_job_name}")
    print(f"Model artifact: {model_artifact}")
    print(f"Training image: {training_image}")

    response = sm.create_model_package(

        ModelPackageGroupName=MODEL_PACKAGE_GROUP,

        ModelPackageDescription=(
            f"Customer churn candidate from "
            f"{training_job_name}"
        ),

        ModelApprovalStatus="PendingManualApproval",

        InferenceSpecification={
            "Containers": [
                {
                    "Image": training_image,
                    "ModelDataUrl": model_artifact,
                }
            ],

            "SupportedContentTypes": [
                "text/csv"
            ],

            "SupportedResponseMIMETypes": [
                "text/csv"
            ],
        },

        CustomerMetadataProperties={
            "training_job_name":
                training_job_name,

            "test_roc_auc":
                str(metrics["test_roc_auc"]),

            "test_pr_auc":
                str(metrics["test_pr_auc"]),

            "test_precision":
                str(metrics["test_precision"]),

            "test_recall":
                str(metrics["test_recall"]),

            "test_f1":
                str(metrics["test_f1"]),

            "test_accuracy":
                str(metrics["test_accuracy"]),

            "prediction_threshold":
                "0.63",

            "quality_gate":
                quality_gate,
        },

        # Tags=[
        #     {
        #         "Key": "Project",
        #         "Value": "CustomerChurn"
        #     },
        #     {
        #         "Key": "Environment",
        #         "Value": "dev"
        #     },
        #     {
        #         "Key": "ModelType",
        #         "Value": "XGBoost"
        #     }
        # ]
    )

    print("\nModel registered successfully.")
    print(
        f"Model Package ARN: "
        f"{response['ModelPackageArn']}"
    )


if __name__ == "__main__":
    main()