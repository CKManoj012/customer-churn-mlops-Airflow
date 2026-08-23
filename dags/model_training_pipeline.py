from datetime import datetime, timedelta
import json
import time

import boto3

from airflow import DAG
from airflow.operators.python import PythonOperator


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

TRAIN_PREFIX = "training/train/"
VALIDATION_PREFIX = "training/validation/"
TEST_PREFIX = "training/test/"

MODEL_OUTPUT_PREFIX = "models/xgboost/"

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


def get_s3_client():
    return boto3.client(
        "s3",
        region_name=REGION,
    )


def get_sm_client():
    return boto3.client(
        "sagemaker",
        region_name=REGION,
    )


# --------------------------------------------------
# 1. Check training data
# --------------------------------------------------

def check_training_data():

    s3 = get_s3_client()

    required_prefixes = [
        TRAIN_PREFIX,
        VALIDATION_PREFIX,
        TEST_PREFIX,
    ]

    for prefix in required_prefixes:

        response = s3.list_objects_v2(
            Bucket=DATA_BUCKET,
            Prefix=prefix,
            MaxKeys=2,
        )

        contents = response.get(
            "Contents",
            []
        )

        real_files = [
            obj
            for obj in contents
            if not obj["Key"].endswith("/")
        ]

        if not real_files:
            raise FileNotFoundError(
                f"No training data found under "
                f"s3://{DATA_BUCKET}/{prefix}"
            )

        print(
            f"PASS: data exists under {prefix}"
        )


# --------------------------------------------------
# 2. Start SageMaker training job
# --------------------------------------------------

def run_training_job(**context):

    sm = get_sm_client()

    timestamp = datetime.utcnow().strftime(
        "%Y%m%d-%H%M%S"
    )

    job_name = (
        f"customer-churn-xgboost-{timestamp}"
    )

    image_uri = (
        "720646828776.dkr.ecr."
        "ap-south-1.amazonaws.com/"
        "sagemaker-xgboost:3.0-5"
    )

    response = sm.create_training_job(

        TrainingJobName=job_name,

        RoleArn=SAGEMAKER_ROLE_ARN,

        AlgorithmSpecification={
            "TrainingImage": image_uri,
            "TrainingInputMode": "File",
        },

        InputDataConfig=[
            {
                "ChannelName": "train",
                "DataSource": {
                    "S3DataSource": {
                        "S3DataType": "S3Prefix",
                        "S3Uri": (
                            f"s3://{DATA_BUCKET}/"
                            f"{TRAIN_PREFIX}"
                        ),
                        "S3DataDistributionType":
                            "FullyReplicated",
                    }
                },
                "ContentType": "text/csv",
            },
            {
                "ChannelName": "validation",
                "DataSource": {
                    "S3DataSource": {
                        "S3DataType": "S3Prefix",
                        "S3Uri": (
                            f"s3://{DATA_BUCKET}/"
                            f"{VALIDATION_PREFIX}"
                        ),
                        "S3DataDistributionType":
                            "FullyReplicated",
                    }
                },
                "ContentType": "text/csv",
            },
        ],

        OutputDataConfig={
            "S3OutputPath": (
                f"s3://{DATA_BUCKET}/"
                f"{MODEL_OUTPUT_PREFIX}"
            )
        },

        ResourceConfig={
            "InstanceType": "ml.m5.large",
            "InstanceCount": 1,
            "VolumeSizeInGB": 10,
        },

        StoppingCondition={
            "MaxRuntimeInSeconds": 3600
        },

        HyperParameters={
            "objective": "binary:logistic",
            "eval_metric": "auc",
            "num_round": "300",
            "max_depth": "5",
            "eta": "0.1",
            "subsample": "0.8",
            "colsample_bytree": "0.8",

            # Use your challenger value here
            "scale_pos_weight": "2.7",

            "early_stopping_rounds": "20",
        },
    )

    print(
        f"Training job submitted: {job_name}"
    )

    context["ti"].xcom_push(
        key="training_job_name",
        value=job_name,
    )


# --------------------------------------------------
# 3. Wait for training
# --------------------------------------------------

def wait_for_training(**context):

    sm = get_sm_client()

    job_name = context["ti"].xcom_pull(
        task_ids="run_training_job",
        key="training_job_name",
    )

    while True:

        response = sm.describe_training_job(
            TrainingJobName=job_name
        )

        status = response[
            "TrainingJobStatus"
        ]

        print(
            f"Training job {job_name} "
            f"status: {status}"
        )

        if status == "Completed":

            model_uri = (
                response[
                    "ModelArtifacts"
                ]["S3ModelArtifacts"]
            )

            print(
                f"Model artifact: {model_uri}"
            )

            context["ti"].xcom_push(
                key="model_artifact_uri",
                value=model_uri,
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
                f"Training job failed. "
                f"Status={status}. "
                f"Reason={reason}"
            )

        time.sleep(30)

def run_evaluation_job(**context):

    sm = get_sm_client()

    ti = context["ti"]

    training_job_name = ti.xcom_pull(
        task_ids="run_training_job",
        key="training_job_name",
    )

    model_artifact_uri = ti.xcom_pull(
        task_ids="wait_for_training",
        key="model_artifact_uri",
    )

    if not model_artifact_uri:
        raise RuntimeError(
            "Model artifact URI was not received."
        )

    timestamp = datetime.utcnow().strftime(
        "%Y%m%d-%H%M%S"
    )

    evaluation_job_name = (
        f"customer-churn-evaluation-{timestamp}"
    )

    image_uri = (
        "720646828776.dkr.ecr."
        "ap-south-1.amazonaws.com/"
        "sagemaker-xgboost:3.0-5"
    )

    sm.create_processing_job(

        ProcessingJobName=
            evaluation_job_name,

        RoleArn=
            SAGEMAKER_ROLE_ARN,

        AppSpecification={
            "ImageUri": image_uri,

            "ContainerEntrypoint": [
                "python3",
                "/opt/ml/processing/code/"
                "evaluate_candidate_job.py",
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

            # Candidate model
            {
                "InputName": "model",

                "S3Input": {
                    "S3Uri":
                        model_artifact_uri,

                    "LocalPath":
                        "/opt/ml/processing/model",

                    "S3DataType":
                        "S3Prefix",

                    "S3InputMode":
                        "File",
                },
            },

            # Validation dataset
            {
                "InputName":
                    "validation",

                "S3Input": {
                    "S3Uri": (
                        f"s3://{DATA_BUCKET}/"
                        f"{VALIDATION_PREFIX}"
                    ),

                    "LocalPath":
                        "/opt/ml/processing/validation",

                    "S3DataType":
                        "S3Prefix",

                    "S3InputMode":
                        "File",
                },
            },

            # Test dataset
            {
                "InputName":
                    "test",

                "S3Input": {
                    "S3Uri": (
                        f"s3://{DATA_BUCKET}/"
                        f"{TEST_PREFIX}"
                    ),

                    "LocalPath":
                        "/opt/ml/processing/test",

                    "S3DataType":
                        "S3Prefix",

                    "S3InputMode":
                        "File",
                },
            },

            # Evaluation Python code
            {
                "InputName":
                    "evaluation-code",

                "S3Input": {
                    "S3Uri": (
                        f"s3://{DATA_BUCKET}/"
                        "scripts/sagemaker/"
                        "evaluation/"
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
                        "evaluation",

                    "S3Output": {
                        "S3Uri": (
                            f"s3://{DATA_BUCKET}/"
                            "evaluation/"
                            f"{training_job_name}/"
                        ),

                        "LocalPath":
                            "/opt/ml/processing/evaluation",

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
        f"Evaluation Processing Job: "
        f"{evaluation_job_name}"
    )

    ti.xcom_push(
        key="evaluation_job_name",
        value=evaluation_job_name,
    )

    ti.xcom_push(
        key="evaluation_s3_key",
        value=(
            "evaluation/"
            f"{training_job_name}/"
            "evaluation.json"
        ),
    )


def wait_for_evaluation(**context):

    sm = get_sm_client()

    job_name = context[
        "ti"
    ].xcom_pull(
        task_ids="run_evaluation_job",
        key="evaluation_job_name",
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
            f"Evaluation job "
            f"{job_name}: {status}"
        )

        if status == "Completed":

            print(
                "Model evaluation completed."
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
                f"Evaluation failed. "
                f"Status={status}. "
                f"Reason={reason}"
            )

        time.sleep(30)

def check_quality_gate(**context):

    s3 = get_s3_client()

    key = context[
        "ti"
    ].xcom_pull(
        task_ids="run_evaluation_job",
        key="evaluation_s3_key",
    )

    response = s3.get_object(
        Bucket=DATA_BUCKET,
        Key=key,
    )

    result = json.loads(
        response["Body"]
        .read()
        .decode("utf-8")
    )

    print(
        json.dumps(
            result,
            indent=4,
        )
    )

    context["ti"].xcom_push(
        key="evaluation_result",
        value=result,
    )

    if (
        result["quality_gate"]
        != "PASS"
    ):
        raise RuntimeError(
            "Candidate model failed "
            "the quality gate."
        )

    print(
        "Candidate model PASSED "
        "the quality gate."
    )
# --------------------------------------------------
# Placeholder tasks
# --------------------------------------------------

def run_mlflow_logging_job(**context):

    sm = get_sm_client()

    ti = context["ti"]

    training_job_name = ti.xcom_pull(
        task_ids="run_training_job",
        key="training_job_name",
    )

    timestamp = datetime.utcnow().strftime(
        "%Y%m%d-%H%M%S"
    )

    job_name = (
        f"customer-churn-mlflow-"
        f"{timestamp}"
    )

    image_uri = (
        "720646828776.dkr.ecr."
        "ap-south-1.amazonaws.com/"
        "sagemaker-scikit-learn:"
        "1.2-1-cpu-py3"
    )

    command = (
        "pip install "
        "-r /opt/ml/processing/code/requirements.txt "
        "&& python3 "
        "/opt/ml/processing/code/log_training_job.py "
        f"--training-job-name {training_job_name}"
    )

    sm.create_processing_job(

        ProcessingJobName=job_name,

        RoleArn=SAGEMAKER_ROLE_ARN,

        AppSpecification={
            "ImageUri": image_uri,

            "ContainerEntrypoint": [
                "bash",
                "-c",
                command,
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
                "InputName":
                    "mlflow-code",

                "S3Input": {
                    "S3Uri": (
                        f"s3://{DATA_BUCKET}/"
                        "scripts/sagemaker/mlflow/"
                    ),

                    "LocalPath":
                        "/opt/ml/processing/code",

                    "S3DataType":
                        "S3Prefix",

                    "S3InputMode":
                        "File",
                },
            }
        ],

        StoppingCondition={
            "MaxRuntimeInSeconds":
                1800
        },
    )

    print(
        f"MLflow Processing Job: "
        f"{job_name}"
    )

    ti.xcom_push(
        key="mlflow_job_name",
        value=job_name,
    )

def wait_for_mlflow_logging(**context):

    sm = get_sm_client()

    job_name = (
        context["ti"]
        .xcom_pull(
            task_ids="run_mlflow_logging_job",
            key="mlflow_job_name",
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
            f"MLflow logging job "
            f"{job_name}: {status}"
        )

        if status == "Completed":

            print(
                "MLflow logging completed."
            )

            return

        if status in {
            "Failed",
            "Stopped",
        }:

            reason = response.get(
                "FailureReason",
                "No failure reason returned.",
            )

            raise RuntimeError(
                f"MLflow logging failed. "
                f"Status={status}. "
                f"Reason={reason}"
            )

        time.sleep(30)



def tune_threshold(**context):

    print(
        "Next: tune threshold using validation set"
    )


def evaluate_candidate(**context):

    print(
        "Next: evaluate candidate on test set"
    )


def quality_gate(**context):

    print(
        "Next: apply ROC-AUC and recall quality gate"
    )


def register_model(**context):

    sm = get_sm_client()
    ti = context["ti"]

    training_job_name = ti.xcom_pull(
        task_ids="run_training_job",
        key="training_job_name",
    )

    model_artifact_uri = ti.xcom_pull(
        task_ids="wait_for_training",
        key="model_artifact_uri",
    )

    evaluation_result = ti.xcom_pull(
        task_ids="quality_gate",
        key="evaluation_result",
    )

    if not evaluation_result:
        raise RuntimeError(
            "Evaluation result was not received."
        )

    if evaluation_result["quality_gate"] != "PASS":
        raise RuntimeError(
            "Model cannot be registered because "
            "the quality gate did not pass."
        )

    threshold = evaluation_result[
        "threshold"
    ]

    metrics = evaluation_result[
        "metrics"
    ]

    image_uri = (
        "720646828776.dkr.ecr."
        "ap-south-1.amazonaws.com/"
        "sagemaker-xgboost:3.0-5"
    )

    # --------------------------------------------------
    # Make sure Model Package Group exists
    # --------------------------------------------------

    try:
        sm.describe_model_package_group(
            ModelPackageGroupName=
                MODEL_PACKAGE_GROUP
        )

        print(
            f"Model Package Group exists: "
            f"{MODEL_PACKAGE_GROUP}"
        )

    except sm.exceptions.ClientError:

        print(
            f"Creating Model Package Group: "
            f"{MODEL_PACKAGE_GROUP}"
        )

        sm.create_model_package_group(
            ModelPackageGroupName=
                MODEL_PACKAGE_GROUP,

            ModelPackageGroupDescription=(
                "Customer churn XGBoost models"
            ),
        )

    # --------------------------------------------------
    # Register model version
    # --------------------------------------------------

    response = sm.create_model_package(

        ModelPackageGroupName=
            MODEL_PACKAGE_GROUP,

        ModelPackageDescription=(
            f"Customer churn model from "
            f"{training_job_name}; "
            f"threshold={threshold}; "
            f"roc_auc={metrics['roc_auc']:.4f}; "
            f"recall={metrics['recall']:.4f}; "
            f"f1={metrics['f1']:.4f}"
        ),

        InferenceSpecification={
            "Containers": [
                {
                    "Image": image_uri,
                    "ModelDataUrl":
                        model_artifact_uri,
                }
            ],

            "SupportedContentTypes": [
                "text/csv"
            ],

            "SupportedResponseMIMETypes": [
                "text/csv"
            ],
        },

        ModelApprovalStatus=
            "PendingManualApproval",
    )

    package_arn = response[
        "ModelPackageArn"
    ]

    print(
        f"Registered model: {package_arn}"
    )

    print(
        "Approval status: PendingManualApproval"
    )

    ti.xcom_push(
        key="model_package_arn",
        value=package_arn,
    )


default_args = {
    "owner": "customer-churn",
    "depends_on_past": False,
    "retries": 1,
    "retry_delay": timedelta(
        minutes=5
    ),
    "on_failure_callback": notify_failure,
}


with DAG(

    dag_id="model_training_pipeline",

    default_args=default_args,

    description=(
        "Customer churn SageMaker training, "
        "MLflow tracking, evaluation and registration"
    ),

    start_date=datetime(
        2026,
        8,
        1
    ),

    schedule="0 3 * * 0",

    catchup=False,

    tags=[
        "customer-churn",
        "training",
        "mlops",
    ],

) as dag:

    check_data = PythonOperator(
        task_id="check_training_data",
        python_callable=check_training_data,
    )

    start_training = PythonOperator(
        task_id="run_training_job",
        python_callable=run_training_job,
    )

    wait_training = PythonOperator(
        task_id="wait_for_training",
        python_callable=wait_for_training,
    )

    start_mlflow = PythonOperator(
        task_id="run_mlflow_logging_job",
        python_callable=run_mlflow_logging_job,
    )

    wait_mlflow = PythonOperator(
        task_id="wait_for_mlflow_logging",
        python_callable=wait_for_mlflow_logging,
    )

    start_evaluation = PythonOperator(
        task_id="run_evaluation_job",
        python_callable=run_evaluation_job,
    )

    wait_evaluation = PythonOperator(
        task_id="wait_for_evaluation",
        python_callable=wait_for_evaluation,
    )

    gate = PythonOperator(
        task_id="quality_gate",
        python_callable=check_quality_gate,
    )

    register = PythonOperator(
        task_id="register_model",
        python_callable=register_model,
    )

    check_data \
        >> start_training \
        >> wait_training \
        >> start_mlflow \
        >> wait_mlflow \
        >> start_evaluation