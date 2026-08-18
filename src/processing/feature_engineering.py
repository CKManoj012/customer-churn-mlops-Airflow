from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split


# --------------------------------------------------
# SageMaker Processing paths
# --------------------------------------------------

INPUT_DIR = Path("/opt/ml/processing/input")

TRAIN_DIR = Path("/opt/ml/processing/train")
VALIDATION_DIR = Path("/opt/ml/processing/validation")
TEST_DIR = Path("/opt/ml/processing/test")
FEATURE_DIR = Path("/opt/ml/processing/features")


def main():

    print("=" * 60)
    print("CUSTOMER CHURN FEATURE ENGINEERING")
    print("=" * 60)

    # --------------------------------------------------
    # Find input Parquet file
    # --------------------------------------------------

    parquet_files = list(
        INPUT_DIR.rglob("*.parquet")
    )

    if not parquet_files:
        raise FileNotFoundError(
            "No Parquet files found in processing input."
        )

    print(
        f"Found {len(parquet_files)} parquet file(s)."
    )

    # Works even if Glue eventually creates multiple parts
    df = pd.concat(
        [
            pd.read_parquet(file)
            for file in parquet_files
        ],
        ignore_index=True,
    )

    print(f"Input shape: {df.shape}")

    # --------------------------------------------------
    # Basic cleanup
    # --------------------------------------------------

    df = df.drop_duplicates(
        subset=["customer_id"]
    )

    # --------------------------------------------------
    # Feature engineering
    # --------------------------------------------------

    service_columns = [
        "phone_service",
        "multiple_lines",
        "internet_service",
        "online_security",
        "online_backup",
        "device_protection_plan",
        "premium_tech_support",
        "streaming_tv",
        "streaming_movies",
    ]

    # Count services that look active
    yes_values = {
        "yes",
        "fiber optic",
        "dsl",
    }

    def count_services(row):

        count = 0

        for column in service_columns:

            value = str(row[column]).strip().lower()

            if value in yes_values:
                count += 1

        return count

    df["number_of_services"] = df.apply(
        count_services,
        axis=1,
    )

    # Internet indicator
    df["has_internet"] = (
        df["internet_service"]
        .astype(str)
        .str.lower()
        .ne("no")
        .astype(int)
    )

    # Streaming indicator
    df["has_streaming"] = (
        (
            df["streaming_tv"]
            .astype(str)
            .str.lower()
            .eq("yes")
        )
        |
        (
            df["streaming_movies"]
            .astype(str)
            .str.lower()
            .eq("yes")
        )
    ).astype(int)

    # Security/support service count
    security_columns = [
        "online_security",
        "online_backup",
        "device_protection_plan",
        "premium_tech_support",
    ]

    df["security_service_count"] = (
        df[security_columns]
        .apply(
            lambda row: sum(
                str(value).strip().lower() == "yes"
                for value in row
            ),
            axis=1,
        )
    )

    # Monthly charge relative to number of active services
    df["monthly_charge_per_service"] = (
        df["monthly_charge"]
        /
        df["number_of_services"].clip(lower=1)
    )

    # --------------------------------------------------
    # Remove identifier
    # --------------------------------------------------

    df = df.drop(
        columns=["customer_id"]
    )

    # --------------------------------------------------
    # Separate target
    # --------------------------------------------------

    target = "churn_value"

    y = df[target]

    X = df.drop(
        columns=[target]
    )

    # --------------------------------------------------
    # Encode categorical variables
    # --------------------------------------------------

    X = pd.get_dummies(
        X,
        drop_first=False,
        dtype=int,
    )

    print(
        f"Feature count after encoding: {X.shape[1]}"
    )

    # Put target back
    model_df = X.copy()

    model_df[target] = y.values

    # --------------------------------------------------
    # 70 / 15 / 15 split
    # --------------------------------------------------

    train_df, temp_df = train_test_split(
        model_df,
        test_size=0.30,
        random_state=42,
        stratify=model_df[target],
    )

    validation_df, test_df = train_test_split(
        temp_df,
        test_size=0.50,
        random_state=42,
        stratify=temp_df[target],
    )

    print(f"Train:      {train_df.shape}")
    print(f"Validation: {validation_df.shape}")
    print(f"Test:       {test_df.shape}")

    # --------------------------------------------------
    # Create output directories
    # --------------------------------------------------

    for directory in [
        TRAIN_DIR,
        VALIDATION_DIR,
        TEST_DIR,
        FEATURE_DIR,
    ]:
        directory.mkdir(
            parents=True,
            exist_ok=True,
        )

    # --------------------------------------------------
    # Save feature dataset
    # --------------------------------------------------

    model_df.to_parquet(
        FEATURE_DIR / "features.parquet",
        index=False,
    )

    # --------------------------------------------------
    # Save training datasets
    # --------------------------------------------------

    train_df.to_csv(
        TRAIN_DIR / "train.csv",
        index=False,
    )

    validation_df.to_csv(
        VALIDATION_DIR / "validation.csv",
        index=False,
    )

    test_df.to_csv(
        TEST_DIR / "test.csv",
        index=False,
    )

    print("=" * 60)
    print("FEATURE ENGINEERING COMPLETED")
    print("=" * 60)


if __name__ == "__main__":
    main()