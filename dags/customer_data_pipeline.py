from datetime import datetime, timedelta

import boto3

from airflow import DAG
from airflow.operators.python import PythonOperator


REGION = "ap-south-1"

DATA_BUCKET = (
    "customer-churn-mlops-dev-"
    "558311101304-ap-south-1"
)

GLUE_JOB_NAME = "customer-churn-etl"

SAGEMAKER_ROLE_ARN = (
    "arn:aws:iam::558311101304:"
    "role/CustomerChurnSageMakerExecutionRole"
)


SOURCE_FILES = {
    "customer_master": {
        "landing": (
            "landing/customer_master/"
            "customer_master.csv"
        ),
        "raw": (
            "raw/customer_master/"
            "customer_master.csv"
        ),
    },

    "customer_activity": {
        "landing": (
            "landing/customer_activity/"
            "customer_activity.csv"
        ),
        "raw": (
            "raw/customer_activity/"
            "customer_activity.csv"
        ),
    },

    "customer_billing": {
        "landing": (
            "landing/customer_billing/"
            "customer_billing.csv"
        ),
        "raw": (
            "raw/customer_billing/"
            "customer_billing.csv"
        ),
    },
}


# --------------------------------------------------
# AWS clients
# --------------------------------------------------

def get_s3_client():
    return boto3.client(
        "s3",
        region_name=REGION,
    )


def get_glue_client():
    return boto3.client(
        "glue",
        region_name=REGION,
    )


def get_sagemaker_client():
    return boto3.client(
        "sagemaker",
        region_name=REGION,
    )


# --------------------------------------------------
# Task 1 - Check landing files
# --------------------------------------------------

def check_source_files():

    s3 = get_s3_client()

    missing_files = []

    for dataset, paths in SOURCE_FILES.items():

        try:

            s3.head_object(
                Bucket=DATA_BUCKET,
                Key=paths["landing"],
            )

            print(
                f"Found source file: {dataset}"
            )

        except Exception:

            missing_files.append(
                paths["landing"]
            )

    if missing_files:

        raise FileNotFoundError(
            "Missing landing files: "
            + ", ".join(missing_files)
        )

    print(
        "All source files are available."
    )


# --------------------------------------------------
# Task 2 - Landing → Raw
# --------------------------------------------------

def landing_to_raw():

    s3 = get_s3_client()

    for dataset, paths in SOURCE_FILES.items():

        s3.copy_object(
            Bucket=DATA_BUCKET,

            CopySource={
                "Bucket": DATA_BUCKET,
                "Key": paths["landing"],
            },

            Key=paths["raw"],
        )

        print(
            f"Copied {dataset} to RAW."
        )


# --------------------------------------------------
# Task 3 - Start Glue ETL
# --------------------------------------------------

def run_glue_etl(**context):

    glue = get_glue_client()

    response = glue.start_job_run(
        JobName=GLUE_JOB_NAME,

        Arguments={
            "--SOURCE_BUCKET":
                DATA_BUCKET,

            "--OUTPUT_PATH":
                (
                    f"s3://{DATA_BUCKET}/"
                    "processed/customer_churn/"
                ),
        },
    )

    job_run_id = response["JobRunId"]

    print(
        f"Glue Job Run ID: "
        f"{job_run_id}"
    )

    context["ti"].xcom_push(
        key="glue_job_run_id",
        value=job_run_id,
    )


# --------------------------------------------------
# Task 4 - Wait for Glue
# --------------------------------------------------

def wait_for_glue(**context):
    import time

    glue = get_glue_client()

    job_run_id = context["ti"].xcom_pull(
        task_ids="run_glue_etl",
        key="glue_job_run_id",
    )

    if not job_run_id:
        raise RuntimeError(
            "Glue JobRunId was not received from run_glue_etl."
        )

    print(f"Waiting for Glue Job Run: {job_run_id}")

    while True:
        response = glue.get_job_run(
            JobName=GLUE_JOB_NAME,
            RunId=job_run_id,
            PredecessorsIncluded=False,
        )

        job_run = response["JobRun"]
        status = job_run["JobRunState"]

        print(
            f"Glue job {GLUE_JOB_NAME} "
            f"run {job_run_id} status: {status}"
        )

        if status == "SUCCEEDED":
            print("Glue ETL completed successfully.")
            return

        if status in {
            "FAILED",
            "STOPPED",
            "TIMEOUT",
            "ERROR",
            "EXPIRED",
        }:
            error_message = job_run.get(
                "ErrorMessage",
                "No Glue error message returned.",
            )

            raise RuntimeError(
                f"Glue ETL failed. "
                f"Status={status}. "
                f"Error={error_message}"
            )

        time.sleep(30)

# --------------------------------------------------
# Task 5 - Start SageMaker Processing
# --------------------------------------------------

def run_feature_processing(**context):

    sm = get_sagemaker_client()

    job_name = (
        "customer-churn-processing-"
        + datetime.utcnow().strftime(
            "%Y%m%d-%H%M%S"
        )
    )

    image_uri = (
        "720646828776.dkr.ecr."
        "ap-south-1.amazonaws.com/"
        "sagemaker-scikit-learn:"
        "1.2-1-cpu-py3"
    )

    response = sm.create_processing_job(

        ProcessingJobName=job_name,

        RoleArn=SAGEMAKER_ROLE_ARN,

        AppSpecification={
            "ImageUri": image_uri,

            "ContainerEntrypoint": [
                "python3",
                "/opt/ml/processing/code/feature_engineering.py",
            ],
        },

        ProcessingResources={
            "ClusterConfig": {
                "InstanceCount": 1,
                "InstanceType": "ml.m5.large",
                "VolumeSizeInGB": 10,
            }
        },

        ProcessingInputs=[
            {
                "InputName": "processed-data",
                "S3Input": {
                    "S3Uri": (
                        f"s3://{DATA_BUCKET}/"
                        "processed/customer_churn/"
                    ),
                    "LocalPath": "/opt/ml/processing/input",
                    "S3DataType": "S3Prefix",
                    "S3InputMode": "File",
                },
            },
            {
                "InputName": "processing-code",
                "S3Input": {
                    "S3Uri": (
                        f"s3://{DATA_BUCKET}/"
                        "scripts/sagemaker/"
                    ),
                    "LocalPath": "/opt/ml/processing/code",
                    "S3DataType": "S3Prefix",
                    "S3InputMode": "File",
                },
            },
        ],

        ProcessingOutputConfig={
            "Outputs": [

                {
                    "OutputName":
                        "features",

                    "S3Output": {
                        "S3Uri": (
                            f"s3://{DATA_BUCKET}/"
                            "features/customer_churn/"
                        ),

                        "LocalPath":
                            "/opt/ml/processing/features",

                        "S3UploadMode":
                            "EndOfJob",
                    },
                },

                {
                    "OutputName":
                        "train",

                    "S3Output": {
                        "S3Uri": (
                            f"s3://{DATA_BUCKET}/"
                            "training/train/"
                        ),

                        "LocalPath":
                            "/opt/ml/processing/train",

                        "S3UploadMode":
                            "EndOfJob",
                    },
                },

                {
                    "OutputName":
                        "validation",

                    "S3Output": {
                        "S3Uri": (
                            f"s3://{DATA_BUCKET}/"
                            "training/validation/"
                        ),

                        "LocalPath":
                            "/opt/ml/processing/validation",

                        "S3UploadMode":
                            "EndOfJob",
                    },
                },

                {
                    "OutputName":
                        "test",

                    "S3Output": {
                        "S3Uri": (
                            f"s3://{DATA_BUCKET}/"
                            "training/test/"
                        ),

                        "LocalPath":
                            "/opt/ml/processing/test",

                        "S3UploadMode":
                            "EndOfJob",
                    },
                },
            ]
        },

        StoppingCondition={
            "MaxRuntimeInSeconds":
                3600
        },
    )

    print(
        f"SageMaker Processing Job: "
        f"{job_name}"
    )

    context["ti"].xcom_push(
        key="processing_job_name",
        value=job_name,
    )


# --------------------------------------------------
# Task 6 - Wait for SageMaker Processing
# --------------------------------------------------

def wait_for_processing(**context):

    sm = get_sagemaker_client()

    job_name = (
        context["ti"].xcom_pull(
            task_ids="run_feature_processing",
            key="processing_job_name",
        )
    )

    waiter = sm.get_waiter(
        "processing_job_completed_or_stopped"
    )

    waiter.wait(
        ProcessingJobName=job_name,

        WaiterConfig={
            "Delay": 30,
            "MaxAttempts": 120,
        },
    )

    response = sm.describe_processing_job(
        ProcessingJobName=job_name
    )

    status = response[
        "ProcessingJobStatus"
    ]

    if status != "Completed":

        raise RuntimeError(
            f"Processing Job failed. "
            f"Status={status}"
        )

    print(
        "Feature processing completed."
    )


# --------------------------------------------------
# DAG configuration
# --------------------------------------------------

default_args = {

    "owner": "customer-churn",

    "depends_on_past": False,

    "retries": 1,

    "retry_delay":
        timedelta(minutes=5),
}


with DAG(

    dag_id="customer_data_pipeline",

    default_args=default_args,

    description=(
        "Customer churn data ingestion, "
        "Glue ETL and feature engineering"
    ),

    start_date=datetime(
        2026,
        8,
        1
    ),

    schedule=None,

    catchup=False,

    tags=[
        "customer-churn",
        "data-pipeline",
        "mlops",
    ],

) as dag:

    check_sources = PythonOperator(
        task_id="check_source_files",
        python_callable=check_source_files,
    )

    copy_raw = PythonOperator(
        task_id="landing_to_raw",
        python_callable=landing_to_raw,
    )

    start_glue = PythonOperator(
        task_id="run_glue_etl",
        python_callable=run_glue_etl,
    )

    wait_glue = PythonOperator(
        task_id="wait_for_glue",
        python_callable=wait_for_glue,
    )

    start_processing = PythonOperator(
        task_id="run_feature_processing",
        python_callable=run_feature_processing,
    )

    wait_processing = PythonOperator(
        task_id="wait_for_processing",
        python_callable=wait_for_processing,
    )


    check_sources \
        >> copy_raw \
        >> start_glue \
        >> wait_glue \
        >> start_processing \
        >> wait_processing