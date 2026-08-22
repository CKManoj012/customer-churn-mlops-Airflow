import boto3
import pandas as pd


REGION = "ap-south-1"
PROFILE = "churn-dev"

BUCKET = "customer-churn-mlops-dev-558311101304-ap-south-1"

ENDPOINT_NAME = "customer-churn-staging"

THRESHOLD = 0.63

LOCAL_TEST_FILE = "artifacts/evaluation/test.csv"


def main():

    session = boto3.Session(
        profile_name=PROFILE,
        region_name=REGION,
    )

    runtime = session.client(
        "sagemaker-runtime"
    )

    # --------------------------------------------------
    # Read test dataset
    # --------------------------------------------------

    df = pd.read_csv(
        LOCAL_TEST_FILE,
        header=None,
    )

    # First column is churn_value
    actual_label = int(
        df.iloc[0, 0]
    )

    # Remaining columns are model features
    features = (
        df.iloc[0, 1:]
        .astype(float)
        .tolist()
    )

    # SageMaker XGBoost expects CSV
    payload = ",".join(
        str(value)
        for value in features
    )

    print(
        f"Sending {len(features)} features "
        f"to staging endpoint..."
    )

    # --------------------------------------------------
    # Invoke endpoint
    # --------------------------------------------------

    response = runtime.invoke_endpoint(
        EndpointName=ENDPOINT_NAME,
        ContentType="text/csv",
        Accept="text/csv",
        Body=payload,
    )

    result = (
        response["Body"]
        .read()
        .decode("utf-8")
        .strip()
    )

    probability = float(result)

    prediction = (
        1
        if probability >= THRESHOLD
        else 0
    )

    print("\n" + "=" * 50)
    print("STAGING ENDPOINT SMOKE TEST")
    print("=" * 50)

    print(
        f"Churn probability : "
        f"{probability:.4f}"
    )

    print(
        f"Threshold         : "
        f"{THRESHOLD:.2f}"
    )

    print(
        f"Prediction        : "
        f"{prediction}"
    )

    print(
        f"Actual label      : "
        f"{actual_label}"
    )

    print(
        f"Prediction class  : "
        f"{'CHURN' if prediction == 1 else 'NO CHURN'}"
    )

    print("=" * 50)

    print(
        "\nSmoke test completed successfully."
    )


if __name__ == "__main__":
    main()