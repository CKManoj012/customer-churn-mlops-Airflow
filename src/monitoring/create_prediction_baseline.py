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

PREDICTION_PREFIX = "predictions/final/"
BASELINE_KEY = (
    "monitoring/baseline/"
    "prediction_baseline.json"
)

LOCAL_DIR = Path(
    "artifacts/monitoring"
)


def get_latest_prediction_file(
    s3_client
):

    response = s3_client.list_objects_v2(
        Bucket=BUCKET,
        Prefix=PREDICTION_PREFIX,
    )

    objects = [
        obj
        for obj in response.get(
            "Contents",
            []
        )
        if obj["Key"].endswith(".csv")
    ]

    if not objects:
        raise RuntimeError(
            "No final prediction files found."
        )

    latest = max(
        objects,
        key=lambda obj: obj["LastModified"],
    )

    return latest["Key"]


def main():

    LOCAL_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    session = boto3.Session(
        profile_name=PROFILE,
        region_name=REGION,
    )

    s3 = session.client("s3")

    prediction_key = (
        get_latest_prediction_file(
            s3
        )
    )

    print(
        f"Using prediction file: "
        f"s3://{BUCKET}/{prediction_key}"
    )

    local_file = (
        LOCAL_DIR /
        "baseline_predictions.csv"
    )

    s3.download_file(
        BUCKET,
        prediction_key,
        str(local_file),
    )

    df = pd.read_csv(
        local_file
    )

    baseline = {
        "source_prediction_file":
            prediction_key,

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
        f"\nBaseline written to "
        f"s3://{BUCKET}/{BASELINE_KEY}"
    )


if __name__ == "__main__":
    main()