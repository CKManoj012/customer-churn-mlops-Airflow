from datetime import datetime, timedelta
import io
import json

import boto3
# import pandas as pd

from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.operators.trigger_dagrun import TriggerDagRunOperator
from airflow.operators.python import BranchPythonOperator
from airflow.operators.empty import EmptyOperator


REGION = "ap-south-1"

DATA_BUCKET = (
    "customer-churn-mlops-dev-"
    "558311101304-ap-south-1"
)

PREDICTION_PREFIX = (
    "predictions/final/"
)

BASELINE_KEY = (
    "monitoring/baseline/"
    "prediction_baseline.json"
)

SNS_TOPIC_ARN = (
    "arn:aws:sns:ap-south-1:"
    "558311101304:"
    "customer-churn-airflow-alerts"
)

FEATURE_BASELINE_KEY = (
    "monitoring/baseline/"
    "feature_baseline.json"
)

PROCESSED_PREFIX = (
    "processed/customer_churn/"
)

NUMERIC_FEATURES = [
    "age",
    "number_of_dependents",
    "monthly_charge",
    "total_charges",
]

CATEGORICAL_FEATURES = [
    "contract",
    "payment_method",
    "internet_service",
    "gender",
    "married",
]

MAX_NUMERIC_MEAN_CHANGE = 0.20
MAX_CATEGORY_DISTRIBUTION_CHANGE = 0.15

# --------------------------------------------------
# Drift thresholds
# --------------------------------------------------

MAX_MEAN_PROBABILITY_CHANGE = 0.10
# MAX_MEAN_PROBABILITY_CHANGE = 0.0



MAX_CHURN_RATE_CHANGE = 0.10


def get_s3_client():

    return boto3.client(
        "s3",
        region_name=REGION,
    )


def get_sns_client():

    return boto3.client(
        "sns",
        region_name=REGION,
    )


def choose_retraining_path(**context):

    result = context["ti"].xcom_pull(
        task_ids="combine_drift_signals",
        key="overall_drift_result",
    )

    if result["overall_drift"]:
        return "alert_on_drift"

    return "no_retraining_needed"

# --------------------------------------------------
# Find latest scored dataset
# --------------------------------------------------

def find_latest_predictions(
    **context
):

    s3 = get_s3_client()

    response = s3.list_objects_v2(
        Bucket=DATA_BUCKET,
        Prefix=PREDICTION_PREFIX,
    )

    objects = [
        obj
        for obj in response.get(
            "Contents",
            []
        )
        if obj["Key"].endswith(
            ".csv"
        )
    ]

    if not objects:

        raise RuntimeError(
            "No final prediction files found."
        )

    latest = max(
        objects,
        key=lambda obj:
            obj["LastModified"],
    )

    key = latest["Key"]

    print(
        f"Latest prediction file: {key}"
    )

    context["ti"].xcom_push(
        key="prediction_key",
        value=key,
    )


# --------------------------------------------------
# Calculate current statistics
# --------------------------------------------------

def calculate_prediction_stats(
    **context
):
    import pandas as pd
    s3 = get_s3_client()

    prediction_key = (
        context["ti"]
        .xcom_pull(
            task_ids=
                "find_latest_predictions",
            key="prediction_key",
        )
    )

    response = s3.get_object(
        Bucket=DATA_BUCKET,
        Key=prediction_key,
    )

    content = (
        response["Body"]
        .read()
        .decode("utf-8")
    )

    df = pd.read_csv(
        io.StringIO(content)
    )

    stats = {
        "row_count":
            int(len(df)),

        "mean_churn_probability":
            float(
                df[
                    "churn_probability"
                ].mean()
            ),

        "std_churn_probability":
            float(
                df[
                    "churn_probability"
                ].std()
            ),

        "predicted_churn_rate":
            float(
                df[
                    "churn_prediction"
                ].mean()
            ),

        "probability_p25":
            float(
                df[
                    "churn_probability"
                ].quantile(0.25)
            ),

        "probability_p50":
            float(
                df[
                    "churn_probability"
                ].quantile(0.50)
            ),

        "probability_p75":
            float(
                df[
                    "churn_probability"
                ].quantile(0.75)
            ),
    }

    print(
        json.dumps(
            stats,
            indent=4,
        )
    )

    context["ti"].xcom_push(
        key="current_stats",
        value=stats,
    )


# --------------------------------------------------
# Compare to baseline
# --------------------------------------------------

def evaluate_prediction_drift(
    **context
):

    s3 = get_s3_client()

    baseline_response = (
        s3.get_object(
            Bucket=DATA_BUCKET,
            Key=BASELINE_KEY,
        )
    )

    baseline = json.loads(
        baseline_response[
            "Body"
        ]
        .read()
        .decode("utf-8")
    )

    current = (
        context["ti"]
        .xcom_pull(
            task_ids=
                "calculate_prediction_stats",
            key="current_stats",
        )
    )

    mean_probability_change = abs(
        current[
            "mean_churn_probability"
        ]
        -
        baseline[
            "mean_churn_probability"
        ]
    )

    churn_rate_change = abs(
        current[
            "predicted_churn_rate"
        ]
        -
        baseline[
            "predicted_churn_rate"
        ]
    )

    drift_detected = (
        mean_probability_change
        >
        MAX_MEAN_PROBABILITY_CHANGE
        or
        churn_rate_change
        >
        MAX_CHURN_RATE_CHANGE
    )

    result = {
        "baseline":
            baseline,

        "current":
            current,

        "mean_probability_change":
            mean_probability_change,

        "churn_rate_change":
            churn_rate_change,

        "drift_detected":
            drift_detected,
    }

    print(
        json.dumps(
            result,
            indent=4,
        )
    )

    context["ti"].xcom_push(
        key="drift_result",
        value=result,
    )


# --------------------------------------------------
# Alert if drift detected
# --------------------------------------------------

def alert_on_drift(
    **context
):

    result = (
        context["ti"]
        .xcom_pull(
            task_ids=
                "combine_drift_signals",
            key="overall_drift_result",
        )
    )

    if not result[
        "overall_drift"
    ]:

        print(
            "No significant drift detected."
        )

        return

    sns = get_sns_client()

    message = (
        "Customer churn drift detected.\n\n"
        f"Prediction drift: "
        f"{result['prediction_drift']}\n"
        f"Feature drift: "
        f"{result['feature_drift']}\n\n"
        "Review the pipeline and consider retraining."
    )

    sns.publish(
        TopicArn=SNS_TOPIC_ARN,

        Subject=(
            "Customer Churn Model "
            "Drift Alert"
        ),

        Message=message,
    )

    print(
        "Drift alert sent through SNS."
    )

def calculate_feature_drift(**context):

    import io
    import pandas as pd

    s3 = get_s3_client()

    # ----------------------------------------------
    # Load feature baseline
    # ----------------------------------------------

    baseline_response = s3.get_object(
        Bucket=DATA_BUCKET,
        Key=FEATURE_BASELINE_KEY,
    )

    baseline = json.loads(
        baseline_response["Body"]
        .read()
        .decode("utf-8")
    )

    # ----------------------------------------------
    # Load latest processed Parquet data
    # ----------------------------------------------

    response = s3.list_objects_v2(
        Bucket=DATA_BUCKET,
        Prefix=PROCESSED_PREFIX,
    )

    parquet_files = [
        obj["Key"]
        for obj in response.get(
            "Contents",
            []
        )
        if obj["Key"].endswith(
            ".parquet"
        )
    ]

    if not parquet_files:
        raise RuntimeError(
            "No processed Parquet files found."
        )

    frames = []

    for key in parquet_files:

        obj = s3.get_object(
            Bucket=DATA_BUCKET,
            Key=key,
        )

        data = obj["Body"].read()

        frames.append(
            pd.read_parquet(
                io.BytesIO(data)
            )
        )

    df = pd.concat(
        frames,
        ignore_index=True,
    )

    result = {
        "numeric": {},
        "categorical": {},
        "drift_detected": False,
    }

    # ----------------------------------------------
    # Numeric drift
    # ----------------------------------------------

    for feature in NUMERIC_FEATURES:

        current_series = pd.to_numeric(
            df[feature],
            errors="coerce",
        )

        current_mean = float(
            current_series.mean()
        )

        baseline_mean = float(
            baseline[
                "numeric"
            ][feature]["mean"]
        )

        denominator = max(
            abs(baseline_mean),
            1e-6,
        )

        relative_change = abs(
            current_mean - baseline_mean
        ) / denominator

        drift = (
            relative_change
            > MAX_NUMERIC_MEAN_CHANGE
        )

        result[
            "numeric"
        ][feature] = {
            "baseline_mean":
                baseline_mean,

            "current_mean":
                current_mean,

            "relative_change":
                relative_change,

            "drift":
                drift,
        }

        if drift:
            result[
                "drift_detected"
            ] = True

    # ----------------------------------------------
    # Categorical drift
    # ----------------------------------------------

    for feature in CATEGORICAL_FEATURES:

        current_dist = (
            df[feature]
            .astype(str)
            .fillna("MISSING")
            .value_counts(
                normalize=True
            )
            .to_dict()
        )

        baseline_dist = baseline[
            "categorical"
        ][feature]

        categories = set(
            baseline_dist.keys()
        ) | set(
            current_dist.keys()
        )

        max_change = 0.0

        for category in categories:

            baseline_value = float(
                baseline_dist.get(
                    category,
                    0.0,
                )
            )

            current_value = float(
                current_dist.get(
                    category,
                    0.0,
                )
            )

            change = abs(
                current_value
                - baseline_value
            )

            max_change = max(
                max_change,
                change,
            )

        drift = (
            max_change
            >
            MAX_CATEGORY_DISTRIBUTION_CHANGE
        )

        result[
            "categorical"
        ][feature] = {
            "max_distribution_change":
                max_change,

            "drift":
                drift,
        }

        if drift:
            result[
                "drift_detected"
            ] = True

    print(
        json.dumps(
            result,
            indent=4,
        )
    )

    context["ti"].xcom_push(
        key="feature_drift_result",
        value=result,
    )

def combine_drift_signals(**context):

    ti = context["ti"]

    prediction_result = (
        ti.xcom_pull(
            task_ids=
                "evaluate_prediction_drift",
            key="drift_result",
        )
    )

    feature_result = (
        ti.xcom_pull(
            task_ids=
                "calculate_feature_drift",
            key="feature_drift_result",
        )
    )

    overall_drift = (
        prediction_result[
            "drift_detected"
        ]
        or
        feature_result[
            "drift_detected"
        ]
    )

    result = {
        "prediction_drift":
            prediction_result[
                "drift_detected"
            ],

        "feature_drift":
            feature_result[
                "drift_detected"
            ],

        "overall_drift":
            overall_drift,
    }

    print(
        json.dumps(
            result,
            indent=4,
        )
    )

    ti.xcom_push(
        key="overall_drift_result",
        value=result,
    )


default_args = {
    "owner": "customer-churn",
    "depends_on_past": False,
    "retries": 1,
    "retry_delay":
        timedelta(minutes=5),
}


with DAG(

    dag_id="model_monitoring_pipeline",

    default_args=default_args,

    description=(
        "Monitor customer churn "
        "prediction drift"
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
        "monitoring",
        "drift",
    ],

) as dag:

    latest_predictions = (
        PythonOperator(
            task_id=
                "find_latest_predictions",

            python_callable=
                find_latest_predictions,
        )
    )

    stats = PythonOperator(
        task_id=
            "calculate_prediction_stats",

        python_callable=
            calculate_prediction_stats,
    )

    drift = PythonOperator(
        task_id=
            "evaluate_prediction_drift",

        python_callable=
            evaluate_prediction_drift,
    )

    feature_drift = PythonOperator(
        task_id="calculate_feature_drift",
        python_callable=
            calculate_feature_drift,
    )

    combine = PythonOperator(
        task_id="combine_drift_signals",
        python_callable=
            combine_drift_signals,
    )

    branch = BranchPythonOperator(
        task_id="check_retraining_needed",
        python_callable=choose_retraining_path,
    )

    no_retraining = EmptyOperator(
        task_id="no_retraining_needed"
    )

    trigger_training = TriggerDagRunOperator(
        task_id="trigger_model_training",
        trigger_dag_id="model_training_pipeline",
        wait_for_completion=False,
    )

    alert = PythonOperator(
        task_id="alert_on_drift",

        python_callable=
            alert_on_drift,
    )


    latest_predictions >> stats >> drift

    drift >> combine
    feature_drift >> combine

    combine >> branch

    branch >> no_retraining

    branch >> alert >> trigger_training