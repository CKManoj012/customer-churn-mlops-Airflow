from pathlib import Path

import pandas as pd


# --------------------------------------------------
# SageMaker Processing paths
# --------------------------------------------------

PROCESSED_DIR = Path(
    "/opt/ml/processing/input"
)

FEATURE_SCHEMA_DIR = Path(
    "/opt/ml/processing/schema"
)

OUTPUT_DIR = Path(
    "/opt/ml/processing/output"
)


def read_parquet_directory(directory):

    files = list(
        directory.rglob("*.parquet")
    )

    if not files:
        raise FileNotFoundError(
            f"No parquet files found under {directory}"
        )

    return pd.concat(
        [
            pd.read_parquet(file)
            for file in files
        ],
        ignore_index=True,
    )


def engineer_features(df):

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

    yes_values = {
        "yes",
        "fiber optic",
        "dsl",
    }

    df["number_of_services"] = df.apply(
        lambda row: sum(
            str(row[col])
            .strip()
            .lower()
            in yes_values
            for col in service_columns
        ),
        axis=1,
    )

    df["has_internet"] = (
        df["internet_service"]
        .astype(str)
        .str.lower()
        .ne("no")
        .astype(int)
    )

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
                str(value)
                .strip()
                .lower()
                == "yes"
                for value in row
            ),
            axis=1,
        )
    )

    df["monthly_charge_per_service"] = (
        df["monthly_charge"]
        /
        df["number_of_services"]
        .clip(lower=1)
    )

    return df


def main():

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    print(
        "Reading processed customer dataset..."
    )

    df = read_parquet_directory(
        PROCESSED_DIR
    )

    print(
        f"Processed dataset shape: {df.shape}"
    )

    # --------------------------------------------------
    # Preserve customer IDs
    # --------------------------------------------------

    customer_ids = df[
        ["customer_id"]
    ].copy()

    # --------------------------------------------------
    # Feature engineering
    # --------------------------------------------------

    df = engineer_features(
        df
    )

    # Remove identifier and target
    drop_columns = [
        "customer_id"
    ]

    if "churn_value" in df.columns:
        drop_columns.append(
            "churn_value"
        )

    X = df.drop(
        columns=drop_columns
    )

    X = pd.get_dummies(
        X,
        drop_first=False,
        dtype=int,
    )

    print(
        f"Features before alignment: "
        f"{X.shape[1]}"
    )

    # --------------------------------------------------
    # Load training feature schema
    # --------------------------------------------------

    training_features = (
        read_parquet_directory(
            FEATURE_SCHEMA_DIR
        )
    )

    training_columns = [
        column
        for column
        in training_features.columns
        if column != "churn_value"
    ]

    print(
        f"Expected training features: "
        f"{len(training_columns)}"
    )

    # --------------------------------------------------
    # Enforce same features/order as training
    # --------------------------------------------------

    X = X.reindex(
        columns=training_columns,
        fill_value=0,
    )

    print(
        f"Final batch feature shape: "
        f"{X.shape}"
    )

    if len(X) != len(customer_ids):
        raise ValueError(
            "Feature rows and customer IDs "
            "do not match."
        )

    # --------------------------------------------------
    # Write SageMaker input
    # --------------------------------------------------

    X.to_csv(
        OUTPUT_DIR /
        "batch_input.csv",
        index=False,
        header=False,
    )

    customer_ids.to_csv(
        OUTPUT_DIR /
        "batch_ids.csv",
        index=False,
    )

    print(
        "Batch preparation completed."
    )

    print(
        f"Customers: {len(X)}"
    )

    print(
        f"Features: {X.shape[1]}"
    )


if __name__ == "__main__":
    main()