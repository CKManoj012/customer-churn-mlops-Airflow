import boto3
import sagemaker

from sagemaker.core import image_uris
from sagemaker.core.helper.session_helper import Session
from sagemaker.core.processing import ScriptProcessor
from sagemaker.core.shapes import (
    ProcessingInput,
    ProcessingOutput,
    ProcessingS3Input,
    ProcessingS3Output,
)

REGION = "ap-south-1"

BUCKET = (
    "customer-churn-mlops-dev-"
    "558311101304-ap-south-1"
)

ROLE_ARN = (
    "arn:aws:iam::558311101304:"
    "role/CustomerChurnSageMakerExecutionRole"
)


# --------------------------------------------------
# AWS session
# --------------------------------------------------

boto_session = boto3.Session(
    profile_name="churn-dev",
    region_name=REGION,
)


sagemaker_session = Session(
    boto_session=boto_session
)

# --------------------------------------------------
# Retrieve sklearn processing image
# --------------------------------------------------

image_uri = image_uris.retrieve(
    framework="sklearn",
    region=REGION,
    version="1.2-1",
    py_version="py3",
    instance_type="ml.m5.large",
)

print(f"Processing image: {image_uri}")

# --------------------------------------------------
# Processing container
# --------------------------------------------------

processor = ScriptProcessor(
    image_uri=image_uri,
    command=["python3"],
    role=ROLE_ARN,
    instance_count=1,
    instance_type="ml.m5.large",
    sagemaker_session=sagemaker_session,
)


# --------------------------------------------------
# Run processing
# --------------------------------------------------

processor.run(
    code="src/processing/feature_engineering.py",

    inputs=[
        ProcessingInput(
            input_name="processed-data",
            s3_input=ProcessingS3Input(
                s3_uri=(
                    f"s3://{BUCKET}/"
                    "processed/customer_churn/"
                ),
                local_path="/opt/ml/processing/input",
                s3_data_type="S3Prefix",
            ),
        )
    ],

    outputs=[
        ProcessingOutput(
            output_name="features",
            s3_output=ProcessingS3Output(
                s3_uri=(
                    f"s3://{BUCKET}/"
                    "features/customer_churn/"
                ),
                local_path="/opt/ml/processing/features",
                s3_upload_mode="EndOfJob",
            ),
        ),

        ProcessingOutput(
            output_name="train",
            s3_output=ProcessingS3Output(
                s3_uri=(
                    f"s3://{BUCKET}/"
                    "training/train/"
                ),
                local_path="/opt/ml/processing/train",
                s3_upload_mode="EndOfJob",
            ),
        ),

        ProcessingOutput(
            output_name="validation",
            s3_output=ProcessingS3Output(
                s3_uri=(
                    f"s3://{BUCKET}/"
                    "training/validation/"
                ),
                local_path="/opt/ml/processing/validation",
                s3_upload_mode="EndOfJob",
            ),
        ),

        ProcessingOutput(
            output_name="test",
            s3_output=ProcessingS3Output(
                s3_uri=(
                    f"s3://{BUCKET}/"
                    "training/test/"
                ),
                local_path="/opt/ml/processing/test",
                s3_upload_mode="EndOfJob",
            ),
        ),
    ],
    wait=False,
    logs=False
)
print("SageMaker Processing Job submitted successfully.")
print(f"Job name: {processor.latest_job.name}")