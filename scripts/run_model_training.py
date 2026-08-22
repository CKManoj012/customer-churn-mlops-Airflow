import time
import boto3

from sagemaker.core import image_uris


REGION = "ap-south-1"

BUCKET = "customer-churn-mlops-dev-558311101304-ap-south-1"

ROLE_ARN = (
    "arn:aws:iam::558311101304:"
    "role/CustomerChurnSageMakerExecutionRole"
)

INSTANCE_TYPE = "ml.m5.large"


# --------------------------------------------------
# AWS Session
# --------------------------------------------------

session = boto3.Session(
    profile_name="churn-dev",
    region_name=REGION,
)

sagemaker_client = session.client("sagemaker")


# --------------------------------------------------
# XGBoost image
# --------------------------------------------------

image_uri = image_uris.retrieve(
    framework="xgboost",
    region=REGION,
    version="3.0-5",
)

print(f"XGBoost image: {image_uri}")


# --------------------------------------------------
# Job name
# --------------------------------------------------

job_name = (
    "customer-churn-xgboost-"
    + time.strftime("%Y-%m-%d-%H-%M-%S")
)


# --------------------------------------------------
# Create Training Job
# --------------------------------------------------

response = sagemaker_client.create_training_job(

    TrainingJobName=job_name,

    RoleArn=ROLE_ARN,

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
                        f"s3://{BUCKET}/"
                        "training/train/"
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
                        f"s3://{BUCKET}/"
                        "training/validation/"
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
            f"s3://{BUCKET}/models/xgboost/"
        )
    },

    ResourceConfig={
        "InstanceType": INSTANCE_TYPE,
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

        # Replace with the ratio calculated from train.csv
        "scale_pos_weight": "2.77",

        # Stop if validation performance stops improving
        "early_stopping_rounds": "20",
    },
)


print("\nTraining Job submitted successfully.")
print(f"Job name: {job_name}")
print(f"Job ARN: {response['TrainingJobArn']}")