import json
from pathlib import Path

import boto3
import pandas as pd


REGION = "ap-south-1"
PROFILE = "churn-dev"

BUCKET = (
    "customer-churn-mlops-dev-"
    "558311101304-ap-south-1"
)

PROCESSED_PREFIX = "processed/customer_churn/"

BASELINE_KEY = (
    "monitoring/baseline/"
    "feature_baseline.json"
)

LOCAL_DIR = Path(
    "artifacts/monitoring/feature_baseline"
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


def download_processed_data(s3):

    LOCAL_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    response = s3.list_objects_v2(
        Bucket=BUCKET,
        Prefix=PROCESSED_PREFIX,
    )

    parquet_objects = [
        obj
        for obj in response.get("Contents", [])
        if obj["Key"].endswith(".parquet")
    ]

    if not parquet_objects:
        raise RuntimeError(
            "No processed Parquet files found."
        )

    files = []

    for index, obj in enumerate(parquet_objects):

        local_file = (
            LOCAL_DIR /
            f"part_{index}.parquet"
        )

        s3.download_file(
            BUCKET,
            obj["Key"],
            str(local_file),
        )

        files.append(local_file)

    return files


def build_baseline(df):

    baseline = {
        "row_count": int(len(df)),
        "numeric": {},
        "categorical": {},
    }

    # ----------------------------------------------
    # Numeric baseline
    # ----------------------------------------------

    for feature in NUMERIC_FEATURES:

        series = pd.to_numeric(
            df[feature],
            errors="coerce",
        )

        baseline["numeric"][feature] = {
            "mean": float(series.mean()),
            "std": float(series.std()),
            "min": float(series.min()),
            "max": float(series.max()),
            "p25": float(series.quantile(0.25)),
            "p50": float(series.quantile(0.50)),
            "p75": float(series.quantile(0.75)),
            "missing_rate": float(
                series.isna().mean()
            ),
        }

    # ----------------------------------------------
    # Categorical baseline
    # ----------------------------------------------

    for feature in CATEGORICAL_FEATURES:

        series = (
            df[feature]
            .astype(str)
            .fillna("MISSING")
        )

        proportions = (
            series
            .value_counts(
                normalize=True
            )
            .to_dict()
        )

        baseline[
            "categorical"
        ][feature] = {
            str(key): float(value)
            for key, value
            in proportions.items()
        }

    return baseline


def main():

    session = boto3.Session(
        profile_name=PROFILE,
        region_name=REGION,
    )

    s3 = session.client("s3")

    files = download_processed_data(
        s3
    )

    df = pd.concat(
        [
            pd.read_parquet(file)
            for file in files
        ],
        ignore_index=True,
    )

    print(
        f"Baseline dataset shape: {df.shape}"
    )

    baseline = build_baseline(
        df
    )

    print(
        json.dumps(
            baseline,
            indent=4,
        )
    )

    s3.put_object(
        Bucket=BUCKET,
        Key=BASELINE_KEY,
        Body=json.dumps(
            baseline,
            indent=4,
        ),
        ContentType="application/json",
    )

    print(
        "\nFeature baseline written to:"
    )

    print(
        f"s3://{BUCKET}/{BASELINE_KEY}"
    )


if __name__ == "__main__":
    main()