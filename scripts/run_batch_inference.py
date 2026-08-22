import time

import boto3


REGION = "ap-south-1"
PROFILE = "churn-dev"

BUCKET = (
    "customer-churn-mlops-dev-"
    "558311101304-ap-south-1"
)

ROLE_ARN = (
    "arn:aws:iam::558311101304:"
    "role/CustomerChurnSageMakerExecutionRole"
)

MODEL_PACKAGE_GROUP = "CustomerChurnModel"

INSTANCE_TYPE = "ml.m5.large"


session = boto3.Session(
    profile_name=PROFILE,
    region_name=REGION,
)

sm = session.client(
    "sagemaker"
)


# --------------------------------------------------
# Find latest approved model package
# --------------------------------------------------

packages = sm.list_model_packages(
    ModelPackageGroupName=MODEL_PACKAGE_GROUP,
    ModelApprovalStatus="Approved",
    SortBy="CreationTime",
    SortOrder="Descending",
    MaxResults=10,
)

if not packages[
    "ModelPackageSummaryList"
]:
    raise RuntimeError(
        "No approved model package found."
    )

package_arn = (
    packages[
        "ModelPackageSummaryList"
    ][0]["ModelPackageArn"]
)

package = sm.describe_model_package(
    ModelPackageName=package_arn
)

container = (
    package[
        "InferenceSpecification"
    ]["Containers"][0]
)

image_uri = container["Image"]
model_data_url = container[
    "ModelDataUrl"
]


timestamp = time.strftime(
    "%Y%m%d-%H%M%S"
)

model_name = (
    f"customer-churn-batch-model-"
    f"{timestamp}"
)

transform_job_name = (
    f"customer-churn-batch-"
    f"{timestamp}"
)


# --------------------------------------------------
# Create temporary SageMaker model
# --------------------------------------------------

sm.create_model(
    ModelName=model_name,

    PrimaryContainer={
        "Image": image_uri,
        "ModelDataUrl": model_data_url,
    },

    ExecutionRoleArn=ROLE_ARN,
)

print(
    f"Created model: "
    f"{model_name}"
)


# --------------------------------------------------
# Create Batch Transform Job
# --------------------------------------------------

input_uri = (
    f"s3://{BUCKET}/"
    "batch-input/customer_churn/"
    "batch_input.csv"
)

output_uri = (
    f"s3://{BUCKET}/"
    "predictions/batch/"
)


sm.create_transform_job(

    TransformJobName=
        transform_job_name,

    ModelName=
        model_name,

    TransformInput={
        "DataSource": {
            "S3DataSource": {
                "S3DataType":
                    "S3Prefix",

                "S3Uri":
                    input_uri,
            }
        },

        "ContentType":
            "text/csv",

        "SplitType":
            "Line",
    },

    TransformOutput={
        "S3OutputPath":
            output_uri,

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
    "\nBatch Transform submitted."
)

print(
    f"Transform job: "
    f"{transform_job_name}"
)

print(
    f"Output: "
    f"{output_uri}"
)

print(
    f"Temporary model: "
    f"{model_name}"
)