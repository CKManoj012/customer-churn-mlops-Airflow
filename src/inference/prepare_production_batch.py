from pathlib import Path

import pandas as pd


PROCESSED_DATA = Path(
    "data/processed/customer_churn/"
)

OUTPUT_DIR = Path(
    "artifacts/batch_inference"
)

BATCH_INPUT = OUTPUT_DIR / "batch_input.csv"
BATCH_IDS = OUTPUT_DIR / "batch_ids.csv"


def main():

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    df = pd.read_parquet(
        PROCESSED_DATA
    )

    customer_ids = df[
        ["customer_id"]
    ].copy()

    # --------------------------------------------------
    # Same feature engineering used during training
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

    yes_values = {
        "yes",
        "fiber optic",
        "dsl",
    }

    df["number_of_services"] = df.apply(
        lambda row: sum(
            str(row[col]).strip().lower()
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

    # Remove non-feature columns
    df = df.drop(
        columns=[
            "customer_id",
            "churn_value",
        ]
    )

    # Encode categorical variables
    X = pd.get_dummies(
        df,
        drop_first=False,
        dtype=int,
    )

    # Save model input
    X.to_csv(
        BATCH_INPUT,
        index=False,
        header=False,
    )

    # Save customer ID mapping
    customer_ids.to_csv(
        BATCH_IDS,
        index=False,
    )

    print(
        f"Batch input created: {BATCH_INPUT}"
    )

    print(
        f"Customer ID mapping created: {BATCH_IDS}"
    )

    print(
        f"Rows: {len(X)}"
    )

    print(
        f"Features: {X.shape[1]}"
    )


if __name__ == "__main__":
    main()