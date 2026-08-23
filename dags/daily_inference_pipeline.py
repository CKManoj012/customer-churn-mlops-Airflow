from datetime import datetime, timedelta
import time

import boto3

from airflow import DAG
from airflow.operators.python import PythonOperator

import csv
import io
from datetime import timezone


REGION = "ap-south-1"

DATA_BUCKET = (
    "customer-churn-mlops-dev-"
    "558311101304-ap-south-1"
)

SAGEMAKER_ROLE_ARN = (
    "arn:aws:iam::558311101304:"
    "role/CustomerChurnSageMakerExecutionRole"
)

MODEL_PACKAGE_GROUP = "CustomerChurnModel"

BATCH_INPUT_URI = (
    f"s3://{DATA_BUCKET}/"
    "batch-input/customer_churn/"
    "batch_input.csv"
)

BATCH_OUTPUT_URI = (
    f"s3://{DATA_BUCKET}/"
    "predictions/batch/"
)

INSTANCE_TYPE = "ml.m5.large"


def notify_failure(context):

    sns = boto3.client(
        "sns",
        region_name=REGION,
    )

    dag_id = context["dag"].dag_id

    task_id = (
        context["task_instance"]
        .task_id
    )

    run_id = context.get(
        "run_id",
        "unknown"
    )

    exception = context.get(
        "exception",
        "Unknown error"
    )

    message = (
        f"Airflow task failed.\n\n"
        f"DAG: {dag_id}\n"
        f"Task: {task_id}\n"
        f"Run ID: {run_id}\n"
        f"Error: {exception}"
    )

    sns.publish(
        TopicArn=(
            "arn:aws:sns:ap-south-1:"
            "558311101304:"
            "customer-churn-airflow-alerts"
        ),

        Subject=(
            f"FAILED: {dag_id} / {task_id}"
        ),

        Message=message,
    )

def get_sm_client():
    return boto3.client(
        "sagemaker",
        region_name=REGION,
    )


def get_s3_client():
    return boto3.client(
        "s3",
        region_name=REGION,
    )


# --------------------------------------------------
# 1. Find latest approved model
# --------------------------------------------------

def check_approved_model(**context):

    sm = get_sm_client()

    response = sm.list_model_packages(
        ModelPackageGroupName=MODEL_PACKAGE_GROUP,
        ModelApprovalStatus="Approved",
        SortBy="CreationTime",
        SortOrder="Descending",
        MaxResults=10,
    )

    packages = response[
        "ModelPackageSummaryList"
    ]

    if not packages:
        raise RuntimeError(
            "No approved model found in "
            f"{MODEL_PACKAGE_GROUP}"
        )

    package_arn = packages[0][
        "ModelPackageArn"
    ]

    details = sm.describe_model_package(
        ModelPackageName=package_arn
    )

    container = (
        details["InferenceSpecification"]
        ["Containers"][0]
    )

    image_uri = container["Image"]

    model_data_url = container[
        "ModelDataUrl"
    ]

    print(
        f"Approved model package: "
        f"{package_arn}"
    )

    print(
        f"Model artifact: "
        f"{model_data_url}"
    )

    ti = context["ti"]

    ti.xcom_push(
        key="model_package_arn",
        value=package_arn,
    )

    ti.xcom_push(
        key="image_uri",
        value=image_uri,
    )

    ti.xcom_push(
        key="model_data_url",
        value=model_data_url,
    )

def wait_for_batch_features(**context):

    sm = get_sm_client()

    job_name = (
        context["ti"]
        .xcom_pull(
            task_ids="prepare_batch_features",
            key="batch_prep_job_name",
        )
    )

    while True:

        response = (
            sm.describe_processing_job(
                ProcessingJobName=job_name
            )
        )

        status = response[
            "ProcessingJobStatus"
        ]

        print(
            f"Batch preparation "
            f"{job_name}: {status}"
        )

        if status == "Completed":

            print(
                "Batch features prepared successfully."
            )

            return

        if status in {
            "Failed",
            "Stopped",
        }:

            reason = response.get(
                "FailureReason",
                "No failure reason returned."
            )

            raise RuntimeError(
                f"Batch feature preparation "
                f"failed. "
                f"Status={status}. "
                f"Reason={reason}"
            )

        time.sleep(30)


def prepare_batch_features(**context):

    sm = get_sm_client()

    timestamp = datetime.utcnow().strftime(
        "%Y%m%d-%H%M%S"
    )

    job_name = (
        f"customer-churn-batch-prep-"
        f"{timestamp}"
    )

    image_uri = (
        "720646828776.dkr.ecr."
        "ap-south-1.amazonaws.com/"
        "sagemaker-scikit-learn:"
        "1.2-1-cpu-py3"
    )

    sm.create_processing_job(

        ProcessingJobName=job_name,

        RoleArn=SAGEMAKER_ROLE_ARN,

        AppSpecification={
            "ImageUri": image_uri,

            "ContainerEntrypoint": [
                "python3",
                "/opt/ml/processing/code/"
                "prepare_batch_job.py",
            ],
        },

        ProcessingResources={
            "ClusterConfig": {
                "InstanceCount": 1,
                "InstanceType":
                    "ml.m5.large",
                "VolumeSizeInGB": 10,
            }
        },

        ProcessingInputs=[

            # Latest processed customer data
            {
                "InputName":
                    "processed-data",

                "S3Input": {
                    "S3Uri": (
                        f"s3://{DATA_BUCKET}/"
                        "processed/customer_churn/"
                    ),

                    "LocalPath":
                        "/opt/ml/processing/input",

                    "S3DataType":
                        "S3Prefix",

                    "S3InputMode":
                        "File",
                },
            },

            # Training feature schema
            {
                "InputName":
                    "feature-schema",

                "S3Input": {
                    "S3Uri": (
                        f"s3://{DATA_BUCKET}/"
                        "features/customer_churn/"
                    ),

                    "LocalPath":
                        "/opt/ml/processing/schema",

                    "S3DataType":
                        "S3Prefix",

                    "S3InputMode":
                        "File",
                },
            },

            # Processing script
            {
                "InputName":
                    "processing-code",

                "S3Input": {
                    "S3Uri": (
                        f"s3://{DATA_BUCKET}/"
                        "scripts/sagemaker/inference/"
                    ),

                    "LocalPath":
                        "/opt/ml/processing/code",

                    "S3DataType":
                        "S3Prefix",

                    "S3InputMode":
                        "File",
                },
            },
        ],

        ProcessingOutputConfig={
            "Outputs": [
                {
                    "OutputName":
                        "batch-input",

                    "S3Output": {
                        "S3Uri": (
                            f"s3://{DATA_BUCKET}/"
                            "batch-input/customer_churn/"
                        ),

                        "LocalPath":
                            "/opt/ml/processing/output",

                        "S3UploadMode":
                            "EndOfJob",
                    },
                }
            ]
        },

        StoppingCondition={
            "MaxRuntimeInSeconds":
                3600
        },
    )

    print(
        f"Batch preparation job: "
        f"{job_name}"
    )

    context["ti"].xcom_push(
        key="batch_prep_job_name",
        value=job_name,
    )
# --------------------------------------------------
# 2. Verify batch input
# --------------------------------------------------

def check_batch_input():

    s3 = get_s3_client()

    key = (
        "batch-input/customer_churn/"
        "batch_input.csv"
    )



    try:
        response = s3.head_object(
            Bucket=DATA_BUCKET,
            Key=key,
        )

    except Exception as exc:
        raise FileNotFoundError(
            f"Batch input not found: "
            f"s3://{DATA_BUCKET}/{key}"
        ) from exc

    print(
        f"Batch input found."
    )

    print(
        f"Size: "
        f"{response['ContentLength']} bytes"
    )


# --------------------------------------------------
# 3. Create temporary SageMaker model
# --------------------------------------------------

def create_batch_model(**context):

    sm = get_sm_client()
    ti = context["ti"]

    image_uri = ti.xcom_pull(
        task_ids="check_approved_model",
        key="image_uri",
    )

    model_data_url = ti.xcom_pull(
        task_ids="check_approved_model",
        key="model_data_url",
    )

    timestamp = datetime.utcnow().strftime(
        "%Y%m%d-%H%M%S"
    )

    model_name = (
        f"customer-churn-batch-model-"
        f"{timestamp}"
    )

    sm.create_model(
        ModelName=model_name,

        PrimaryContainer={
            "Image": image_uri,
            "ModelDataUrl":
                model_data_url,
        },

        ExecutionRoleArn=
            SAGEMAKER_ROLE_ARN,
    )

    print(
        f"Created temporary model: "
        f"{model_name}"
    )

    ti.xcom_push(
        key="batch_model_name",
        value=model_name,
    )


# --------------------------------------------------
# 4. Start Batch Transform
# --------------------------------------------------

def run_batch_transform(**context):

    sm = get_sm_client()
    ti = context["ti"]

    model_name = ti.xcom_pull(
        task_ids="create_batch_model",
        key="batch_model_name",
    )

    timestamp = datetime.utcnow().strftime(
        "%Y%m%d-%H%M%S"
    )

    job_name = (
        f"customer-churn-batch-"
        f"{timestamp}"
    )

    sm.create_transform_job(

        TransformJobName=job_name,

        ModelName=model_name,

        TransformInput={
            "DataSource": {
                "S3DataSource": {
                    "S3DataType":
                        "S3Prefix",

                    "S3Uri":
                        BATCH_INPUT_URI,
                }
            },

            "ContentType":
                "text/csv",

            "SplitType":
                "Line",
        },

        TransformOutput={
            "S3OutputPath":
                BATCH_OUTPUT_URI,

            "Accept":
                "text/csv",
        },

        TransformResources={
            "InstanceType":
                INSTANCE_TYPE,

            "InstanceCount":
                1,
        },
    )

    print(
        f"Batch Transform Job: "
        f"{job_name}"
    )

    ti.xcom_push(
        key="transform_job_name",
        value=job_name,
    )


# --------------------------------------------------
# 5. Wait for Batch Transform
# --------------------------------------------------

def wait_for_transform(**context):

    sm = get_sm_client()

    job_name = context[
        "ti"
    ].xcom_pull(
        task_ids="run_batch_transform",
        key="transform_job_name",
    )

    while True:

        response = sm.describe_transform_job(
            TransformJobName=job_name
        )

        status = response[
            "TransformJobStatus"
        ]

        print(
            f"Transform job "
            f"{job_name}: {status}"
        )

        if status == "Completed":

            print(
                "Batch Transform completed."
            )

            return

        if status in {
            "Failed",
            "Stopped",
        }:

            reason = response.get(
                "FailureReason",
                "No failure reason returned."
            )

            raise RuntimeError(
                f"Batch Transform failed. "
                f"Status={status}. "
                f"Reason={reason}"
            )

        time.sleep(30)


# --------------------------------------------------
# 6. Verify prediction output
# --------------------------------------------------

def check_prediction_output():

    s3 = get_s3_client()

    response = s3.list_objects_v2(
        Bucket=DATA_BUCKET,
        Prefix="predictions/batch/",
    )

    objects = response.get(
        "Contents",
        []
    )

    output_files = [
        obj["Key"]
        for obj in objects
        if obj["Key"].endswith(".out")
    ]

    if not output_files:
        raise RuntimeError(
            "No Batch Transform output found."
        )

    print(
        "Batch prediction outputs:"
    )

    for key in output_files:
        print(key)


# --------------------------------------------------
# 7. Cleanup temporary model
# --------------------------------------------------

def cleanup_batch_model(**context):

    sm = get_sm_client()

    model_name = context[
        "ti"
    ].xcom_pull(
        task_ids="create_batch_model",
        key="batch_model_name",
    )

    if not model_name:
        print(
            "No temporary model found."
        )
        return

    sm.delete_model(
        ModelName=model_name
    )

    print(
        f"Deleted temporary model: "
        f"{model_name}"
    )


# --------------------------------------------------
# DAG config
# --------------------------------------------------

default_args = {
    "owner": "customer-churn",
    "depends_on_past": False,
    "retries": 1,
    "retry_delay": timedelta(
        minutes=5,
    ),
    "on_failure_callback": notify_failure,
}


def postprocess_predictions(**context):

    import pandas as pd

    s3 = get_s3_client()
    ti = context["ti"]

    model_package_arn = ti.xcom_pull(
        task_ids="check_approved_model",
        key="model_package_arn",
    )

    # --------------------------------------------------
    # Read customer IDs
    # --------------------------------------------------

    ids_response = s3.get_object(
        Bucket=DATA_BUCKET,
        Key=(
            "batch-input/customer_churn/"
            "batch_ids.csv"
        ),
    )

    ids_content = (
        ids_response["Body"]
        .read()
        .decode("utf-8")
    )

    ids_df = pd.read_csv(
        io.StringIO(ids_content)
    )

    # --------------------------------------------------
    # Find latest batch output
    # --------------------------------------------------

    response = s3.list_objects_v2(
        Bucket=DATA_BUCKET,
        Prefix="predictions/batch/",
    )

    output_files = [
        obj
        for obj in response.get("Contents", [])
        if obj["Key"].endswith(".out")
    ]

    if not output_files:
        raise RuntimeError(
            "No Batch Transform prediction file found."
        )

    # Pick most recently modified .out file
    latest_output = max(
        output_files,
        key=lambda obj: obj["LastModified"],
    )

    prediction_key = latest_output["Key"]

    print(
        f"Using prediction file: {prediction_key}"
    )

    prediction_response = s3.get_object(
        Bucket=DATA_BUCKET,
        Key=prediction_key,
    )

    prediction_content = (
        prediction_response["Body"]
        .read()
        .decode("utf-8")
    )

    predictions = pd.read_csv(
        io.StringIO(prediction_content),
        header=None,
        names=["churn_probability"],
    )

    # --------------------------------------------------
    # Validate row counts
    # --------------------------------------------------

    if len(ids_df) != len(predictions):

        raise RuntimeError(
            "Customer IDs and predictions "
            "have different row counts. "
            f"IDs={len(ids_df)}, "
            f"Predictions={len(predictions)}"
        )

    # --------------------------------------------------
    # Prediction threshold
    # --------------------------------------------------

    # Current approved candidate threshold
    threshold = 0.63

    predictions.insert(
        0,
        "customer_id",
        ids_df["customer_id"].values,
    )

    predictions[
        "churn_prediction"
    ] = (
        predictions["churn_probability"]
        >= threshold
    ).astype(int)

    predictions[
        "prediction_label"
    ] = predictions[
        "churn_prediction"
    ].map(
        {
            0: "NO_CHURN",
            1: "CHURN",
        }
    )

    predictions[
        "prediction_threshold"
    ] = threshold

    predictions[
        "model_package_arn"
    ] = model_package_arn

    scoring_timestamp = datetime.now(
        timezone.utc
    ).strftime(
        "%Y-%m-%dT%H:%M:%SZ"
    )

    predictions[
        "scoring_timestamp"
    ] = scoring_timestamp

    # --------------------------------------------------
    # Upload final CSV
    # --------------------------------------------------

    csv_buffer = io.StringIO()

    predictions.to_csv(
        csv_buffer,
        index=False,
    )

    output_key = (
        "predictions/final/"
        f"customer_churn_predictions_"
        f"{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}.csv"
    )

    s3.put_object(
        Bucket=DATA_BUCKET,
        Key=output_key,
        Body=csv_buffer.getvalue(),
        ContentType="text/csv",
    )

    print(
        f"Final predictions written to "
        f"s3://{DATA_BUCKET}/{output_key}"
    )

    print(
        f"Customers scored: {len(predictions)}"
    )

    print(
        "\nPrediction distribution:"
    )

    print(
        predictions[
            "prediction_label"
        ].value_counts()
    )

    ti.xcom_push(
        key="final_prediction_key",
        value=output_key,
    )

with DAG(

    dag_id="daily_inference_pipeline",

    default_args=default_args,

    description=(
        "Customer churn batch inference pipeline"
    ),

    start_date=datetime(
        2026,
        8,
        1
    ),

    # Manual first.
    schedule="0 6 * * *",

    catchup=False,

    tags=[
        "customer-churn",
        "batch-inference",
        "mlops",
    ],

) as dag:

    prepare_features = PythonOperator(
        task_id="prepare_batch_features",
        python_callable=prepare_batch_features,
    )

    wait_features = PythonOperator(
        task_id="wait_for_batch_features",
        python_callable=wait_for_batch_features,
    )

    approved_model = PythonOperator(
        task_id="check_approved_model",
        python_callable=check_approved_model,
    )

    batch_input = PythonOperator(
        task_id="check_batch_input",
        python_callable=check_batch_input,
    )

    create_model = PythonOperator(
        task_id="create_batch_model",
        python_callable=create_batch_model,
    )

    start_transform = PythonOperator(
        task_id="run_batch_transform",
        python_callable=run_batch_transform,
    )

    wait_transform = PythonOperator(
        task_id="wait_for_transform",
        python_callable=wait_for_transform,
    )

    output_check = PythonOperator(
        task_id="check_prediction_output",
        python_callable=check_prediction_output,
    )

    postprocess = PythonOperator(
        task_id="postprocess_predictions",
        python_callable=postprocess_predictions,
    )

    cleanup = PythonOperator(
        task_id="cleanup_batch_model",
        python_callable=cleanup_batch_model,
    )

    prepare_features \
        >> wait_features \
        >> batch_input


    [
        approved_model,
        batch_input,
    ] \
        >> create_model \
        >> start_transform \
        >> wait_transform \
        >> output_check \
        >> postprocess \
        >> cleanup