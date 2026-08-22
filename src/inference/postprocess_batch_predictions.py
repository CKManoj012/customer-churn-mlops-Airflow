from datetime import datetime, timezone
from pathlib import Path

import boto3
import pandas as pd


REGION = "ap-south-1"
PROFILE = "churn-dev"

BUCKET = "customer-churn-mlops-dev-558311101304-ap-south-1"

THRESHOLD = 0.63

# Raw batch output from SageMaker
PREDICTION_KEY = (
    "predictions/batch/"
    "batch_input.csv.out"
)

# We need customer IDs in exactly the same row order
# as the batch input.
# SOURCE_FEATURE_FILE = Path(
#     "artifacts/evaluation/test.csv"
# )

OUTPUT_DIR = Path(
    "artifacts/batch_inference"
)

OUTPUT_FILE = (
    OUTPUT_DIR /
    "customer_churn_predictions.csv"
)

BATCH_IDS_FILE = Path(
    "artifacts/batch_inference/batch_ids.csv"
)

def download_predictions(
    s3_client
):
    """
    Download SageMaker Batch Transform output.
    """

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    local_file = (
        OUTPUT_DIR /
        "batch_input.csv.out"
    )

    print(
        "Downloading predictions..."
    )

    s3_client.download_file(
        BUCKET,
        PREDICTION_KEY,
        str(local_file),
    )

    return local_file


def load_probabilities(
    prediction_file
):
    """
    Batch Transform writes one probability
    per input row.
    """

    predictions = pd.read_csv(
        prediction_file,
        header=None,
        names=[
            "churn_probability"
        ],
    )

    predictions[
        "churn_probability"
    ] = pd.to_numeric(
        predictions[
            "churn_probability"
        ],
        errors="raise",
    )

    return predictions




def main():

    session = boto3.Session(
        profile_name=PROFILE,
        region_name=REGION,
    )

    s3_client = session.client(
        "s3"
    )

    prediction_file = (
        download_predictions(
            s3_client
        )
    )

    predictions = (
        load_probabilities(
            prediction_file
        )
    )

    # --------------------------------------------------
    # Load real customer IDs
    # --------------------------------------------------

    customer_ids = pd.read_csv(
        BATCH_IDS_FILE
    )

    if len(customer_ids) != len(predictions):
        raise ValueError(
            "Customer ID row count does not match "
            "prediction row count. "
            f"Customer IDs={len(customer_ids)}, "
            f"Predictions={len(predictions)}"
        )

    predictions.insert(
        0,
        "customer_id",
        customer_ids["customer_id"].values,
    )
    

    # Temporary IDs for the current test run.
    # We'll replace this with real customer IDs
    # in the production batch pipeline.
    customer_ids = pd.read_csv(
        BATCH_IDS_FILE
    )

    if len(customer_ids) != len(predictions):
        raise ValueError(
            "Customer ID row count does not "
            "match prediction count. "
            f"IDs={len(customer_ids)}, "
            f"Predictions={len(predictions)}"
        )

    # predictions.insert(
    #     0,
    #     "customer_id",
    #     customer_ids["customer_id"].values,
    # )

    predictions[
        "churn_prediction"
    ] = (
        predictions[
            "churn_probability"
        ]
        >= THRESHOLD
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
    ] = THRESHOLD

    predictions[
        "scoring_timestamp"
    ] = datetime.now(
        timezone.utc
    ).isoformat()

    # Rearrange columns
    predictions = predictions[
        [
            "customer_id",
            "churn_probability",
            "churn_prediction",
            "prediction_label",
            "prediction_threshold",
            "scoring_timestamp",
        ]
    ]

    predictions.to_csv(
        OUTPUT_FILE,
        index=False,
    )

    print(
        f"\nCreated: {OUTPUT_FILE}"
    )

    print(
        f"Rows: {len(predictions)}"
    )

    print(
        "\nPrediction distribution:"
    )

    print(
        predictions[
            "prediction_label"
        ].value_counts()
    )

    print(
        "\nSample:"
    )

    print(
        predictions.head()
    )


if __name__ == "__main__":
    main()