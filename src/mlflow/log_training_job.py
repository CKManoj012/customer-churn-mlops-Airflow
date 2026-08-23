import argparse

import boto3
import mlflow


REGION = "ap-south-1"

TRACKING_ARN = (
    "arn:aws:sagemaker:ap-south-1:558311101304:"
    "mlflow-tracking-server/customer-churn-mlflow-dev"
)

EXPERIMENT_NAME = "Customer_Churn"


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--training-job-name",
        required=True,
    )

    args = parser.parse_args()

    training_job_name = (
        args.training_job_name
    )

    sm = boto3.client(
        "sagemaker",
        region_name=REGION,
    )

    details = sm.describe_training_job(
        TrainingJobName=training_job_name
    )

    hyperparameters = details.get(
        "HyperParameters",
        {}
    )

    final_metrics = details.get(
        "FinalMetricDataList",
        []
    )

    model_artifact_uri = (
        details["ModelArtifacts"]
        ["S3ModelArtifacts"]
    )

    train_uri = (
        details["InputDataConfig"][0]
        ["DataSource"]
        ["S3DataSource"]
        ["S3Uri"]
    )

    validation_uri = (
        details["InputDataConfig"][1]
        ["DataSource"]
        ["S3DataSource"]
        ["S3Uri"]
    )

    metrics = {
        item["MetricName"]:
            float(item["Value"])
        for item in final_metrics
    }

    # ----------------------------------------------
    # Connect to managed MLflow
    # ----------------------------------------------

    mlflow.set_tracking_uri(
        TRACKING_ARN
    )

    mlflow.set_experiment(
        EXPERIMENT_NAME
    )

    # ----------------------------------------------
    # Log training run
    # ----------------------------------------------

    with mlflow.start_run(
        run_name=training_job_name
    ) as run:

        mlflow.log_param(
            "training_job_name",
            training_job_name,
        )

        mlflow.log_param(
            "model_type",
            "XGBoost",
        )

        mlflow.log_param(
            "train_s3_uri",
            train_uri,
        )

        mlflow.log_param(
            "validation_s3_uri",
            validation_uri,
        )

        mlflow.log_param(
            "model_artifact_uri",
            model_artifact_uri,
        )

        for name, value in (
            hyperparameters.items()
        ):

            mlflow.log_param(
                name,
                value,
            )

        for name, value in (
            metrics.items()
        ):

            clean_name = (
                name
                .replace(":", "_")
                .replace("-", "_")
            )

            mlflow.log_metric(
                clean_name,
                value,
            )

        mlflow.set_tag(
            "orchestrator",
            "Amazon MWAA",
        )

        mlflow.set_tag(
            "pipeline",
            "model_training_pipeline",
        )

        mlflow.set_tag(
            "aws_region",
            REGION,
        )

        run_id = run.info.run_id

    print(
        "MLflow logging completed."
    )

    print(
        f"Run ID: {run_id}"
    )


if __name__ == "__main__":
    main()