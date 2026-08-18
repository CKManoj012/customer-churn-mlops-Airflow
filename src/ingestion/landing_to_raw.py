import boto3


BUCKET = "customer-churn-mlops-dev-558311101304-ap-south-1"

SOURCE_FILES = {
    "customer_master": {
        "landing": "landing/customer_master/customer_master.csv",
        "raw": "raw/customer_master/customer_master.csv",
    },
    "customer_billing": {
        "landing": "landing/customer_billing/customer_billing.csv",
        "raw": "raw/customer_billing/customer_billing.csv",
    },
    "customer_activity": {
        "landing": "landing/customer_activity/customer_activity.csv",
        "raw": "raw/customer_activity/customer_activity.csv",
    },
}


def get_s3_client():
    """
    Uses local AWS CLI profile during development.
    Later MWAA will use its execution role automatically.
    """
    session = boto3.Session(profile_name="churn-dev")

    return session.client(
        "s3",
        region_name="ap-south-1",
    )


def object_exists(s3_client, bucket, key):
    """
    Check whether an S3 object exists.
    """

    try:
        s3_client.head_object(
            Bucket=bucket,
            Key=key,
        )

        return True

    except s3_client.exceptions.ClientError as exc:

        error_code = exc.response["Error"]["Code"]

        if error_code in ["404", "NoSuchKey"]:
            return False

        raise


def check_source_files(s3_client):
    """
    Confirm all required source files exist in landing/.
    """

    print("\nChecking landing files...")

    missing_files = []

    for dataset, paths in SOURCE_FILES.items():

        landing_key = paths["landing"]

        if object_exists(
            s3_client,
            BUCKET,
            landing_key,
        ):
            print(
                f"PASS: {dataset} found at "
                f"s3://{BUCKET}/{landing_key}"
            )

        else:
            missing_files.append(landing_key)

    if missing_files:
        raise FileNotFoundError(
            "Missing source files:\n"
            + "\n".join(missing_files)
        )

    print("\nPASS: All required source files exist.")


def copy_to_raw(s3_client):
    """
    Copy each source dataset from landing/ to raw/.
    """

    print("\nCopying files to RAW layer...")

    for dataset, paths in SOURCE_FILES.items():

        landing_key = paths["landing"]
        raw_key = paths["raw"]

        copy_source = {
            "Bucket": BUCKET,
            "Key": landing_key,
        }

        s3_client.copy_object(
            Bucket=BUCKET,
            CopySource=copy_source,
            Key=raw_key,
        )

        print(
            f"COPIED: {dataset}\n"
            f"  {landing_key}\n"
            f"  -> {raw_key}"
        )


def verify_raw_files(s3_client):
    """
    Verify all expected files exist in RAW.
    """

    print("\nVerifying RAW files...")

    missing_files = []

    for dataset, paths in SOURCE_FILES.items():

        raw_key = paths["raw"]

        if object_exists(
            s3_client,
            BUCKET,
            raw_key,
        ):
            print(
                f"PASS: {dataset} available in RAW."
            )

        else:
            missing_files.append(raw_key)

    if missing_files:
        raise FileNotFoundError(
            "RAW verification failed:\n"
            + "\n".join(missing_files)
        )

    print("\nPASS: RAW ingestion completed.")


def main():

    print("=" * 60)
    print("CUSTOMER CHURN - LANDING TO RAW INGESTION")
    print("=" * 60)

    s3_client = get_s3_client()

    check_source_files(s3_client)

    copy_to_raw(s3_client)

    verify_raw_files(s3_client)

    print("\n" + "=" * 60)
    print("INGESTION SUCCESSFUL")
    print("=" * 60)


if __name__ == "__main__":
    main()