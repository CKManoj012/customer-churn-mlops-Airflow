import time
import boto3


REGION = "ap-south-1"
PROFILE = "churn-dev"

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

sm = session.client("sagemaker")


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

if not packages["ModelPackageSummaryList"]:
    raise RuntimeError(
        "No approved model package found."
    )

model_package_arn = (
    packages["ModelPackageSummaryList"][0]
    ["ModelPackageArn"]
)

package = sm.describe_model_package(
    ModelPackageName=model_package_arn
)


container = (
    package["InferenceSpecification"]
    ["Containers"][0]
)

image_uri = container["Image"]

model_data_url = container["ModelDataUrl"]


timestamp = time.strftime(
    "%Y%m%d-%H%M%S"
)

model_name = (
    f"customer-churn-staging-model-{timestamp}"
)

endpoint_config_name = (
    f"customer-churn-staging-config-{timestamp}"
)

endpoint_name = "customer-churn-staging"


print(
    f"Using Model Package: "
    f"{model_package_arn}"
)

print(
    f"Model artifact: "
    f"{model_data_url}"
)


# --------------------------------------------------
# Create SageMaker Model
# --------------------------------------------------

sm.create_model(

    ModelName=model_name,

    PrimaryContainer={
        "Image": image_uri,
        "ModelDataUrl": model_data_url,
        "Environment": {},
    },

    ExecutionRoleArn=ROLE_ARN,
)

print(
    f"Created model: "
    f"{model_name}"
)


# --------------------------------------------------
# Create Endpoint Config
# --------------------------------------------------

sm.create_endpoint_config(

    EndpointConfigName=endpoint_config_name,

    ProductionVariants=[
        {
            "VariantName": "AllTraffic",
            "ModelName": model_name,
            "InitialInstanceCount": 1,
            "InstanceType": INSTANCE_TYPE,
            "InitialVariantWeight": 1.0,
        }
    ],
)

print(
    f"Created endpoint config: "
    f"{endpoint_config_name}"
)


# --------------------------------------------------
# Create or update staging endpoint
# --------------------------------------------------

existing = sm.list_endpoints(
    NameContains=endpoint_name,
    MaxResults=10,
)

exact_match = [
    item
    for item in existing["Endpoints"]
    if item["EndpointName"] == endpoint_name
]


if exact_match:

    sm.update_endpoint(
        EndpointName=endpoint_name,
        EndpointConfigName=endpoint_config_name,
    )

    print(
        f"Updating endpoint: "
        f"{endpoint_name}"
    )

else:

    sm.create_endpoint(
        EndpointName=endpoint_name,
        EndpointConfigName=endpoint_config_name,
    )

    print(
        f"Creating endpoint: "
        f"{endpoint_name}"
    )


print("\nDeployment submitted.")
print(
    f"Endpoint: "
    f"{endpoint_name}"
)