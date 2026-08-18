from pathlib import Path

import pandas as pd


DATA_DIR = Path("data/local")


EXPECTED_SCHEMAS = {
    "customer_master.csv": [
        "customer_id",
        "gender",
        "age",
        "married",
        "number_of_dependents",
    ],

    "customer_billing.csv": [
        "customer_id",
        "contract",
        "paperless_billing",
        "payment_method",
        "monthly_charge",
        "total_charges",
        "churn_value",
    ],

    "customer_activity.csv": [
        "customer_id",
        "phone_service",
        "multiple_lines",
        "internet_service",
        "online_security",
        "online_backup",
        "device_protection_plan",
        "premium_tech_support",
        "streaming_tv",
        "streaming_movies",
    ],
}


def validate_file(filename, expected_columns):
    """
    Performs common validation checks applicable to all source datasets.
    """

    file_path = DATA_DIR / filename

    print(f"\nValidating: {filename}")

    # -------------------------------------------------
    # 1. File exists
    # -------------------------------------------------

    if not file_path.exists():
        raise FileNotFoundError(
            f"Required dataset not found: {file_path}"
        )

    # -------------------------------------------------
    # 2. Load file
    # -------------------------------------------------

    df = pd.read_csv(file_path)

    # -------------------------------------------------
    # 3. File should not be empty
    # -------------------------------------------------

    if df.empty:
        raise ValueError(
            f"{filename} contains no records."
        )

    # -------------------------------------------------
    # 4. Schema validation
    # -------------------------------------------------

    actual_columns = set(df.columns)
    expected_columns_set = set(expected_columns)

    missing_columns = expected_columns_set - actual_columns
    unexpected_columns = actual_columns - expected_columns_set

    if missing_columns:
        raise ValueError(
            f"{filename} is missing columns: "
            f"{sorted(missing_columns)}"
        )

    if unexpected_columns:
        print(
            f"WARNING: {filename} contains additional columns: "
            f"{sorted(unexpected_columns)}"
        )

    # -------------------------------------------------
    # 5. Customer ID null check
    # -------------------------------------------------

    if df["customer_id"].isna().any():
        raise ValueError(
            f"{filename}: customer_id contains NULL values."
        )

    # -------------------------------------------------
    # 6. Customer ID blank check
    # -------------------------------------------------

    blank_customer_ids = (
        df["customer_id"]
        .astype(str)
        .str.strip()
        .eq("")
    )

    if blank_customer_ids.any():
        raise ValueError(
            f"{filename}: customer_id contains blank values."
        )

    # -------------------------------------------------
    # 7. Customer ID uniqueness
    # -------------------------------------------------

    duplicate_count = df["customer_id"].duplicated().sum()

    if duplicate_count > 0:
        raise ValueError(
            f"{filename}: {duplicate_count} duplicate "
            f"customer_id values detected."
        )

    print(
        f"PASS: {filename} "
        f"({len(df)} rows, {len(df.columns)} columns)"
    )

    return df


def validate_customer_master(df):
    """
    Validation rules specific to customer_master.csv.
    """

    print("Running customer master validations...")

    # Age should be numeric
    df["age"] = pd.to_numeric(
        df["age"],
        errors="coerce"
    )

    if df["age"].isna().any():
        raise ValueError(
            "customer_master.csv: age contains "
            "missing or non-numeric values."
        )

    # Reasonable age range
    invalid_age = ~df["age"].between(0, 120)

    if invalid_age.any():
        raise ValueError(
            "customer_master.csv: age contains "
            "values outside 0-120."
        )

    # Dependents should be numeric
    df["number_of_dependents"] = pd.to_numeric(
        df["number_of_dependents"],
        errors="coerce"
    )

    if df["number_of_dependents"].isna().any():
        raise ValueError(
            "customer_master.csv: number_of_dependents "
            "contains missing or non-numeric values."
        )

    if (df["number_of_dependents"] < 0).any():
        raise ValueError(
            "customer_master.csv: number_of_dependents "
            "contains negative values."
        )

    print("PASS: customer master business rules")


def validate_customer_billing(df):
    """
    Validation rules specific to customer_billing.csv.
    """

    print("Running customer billing validations...")

    # Monthly charge
    df["monthly_charge"] = pd.to_numeric(
        df["monthly_charge"],
        errors="coerce"
    )

    if df["monthly_charge"].isna().any():
        raise ValueError(
            "customer_billing.csv: monthly_charge "
            "contains missing or non-numeric values."
        )

    if (df["monthly_charge"] < 0).any():
        raise ValueError(
            "customer_billing.csv: monthly_charge "
            "contains negative values."
        )

    # Total charges
    df["total_charges"] = pd.to_numeric(
        df["total_charges"],
        errors="coerce"
    )

    if (df["total_charges"].dropna() < 0).any():
        raise ValueError(
            "customer_billing.csv: total_charges "
            "contains negative values."
        )

    # Churn target must be binary
    invalid_churn = ~df["churn_value"].isin([0, 1])

    if invalid_churn.any():
        invalid_values = (
            df.loc[invalid_churn, "churn_value"]
            .unique()
            .tolist()
        )

        raise ValueError(
            f"customer_billing.csv: churn_value must "
            f"contain only 0 or 1. Found: {invalid_values}"
        )

    print("PASS: customer billing business rules")


def validate_customer_activity(df):
    """
    Validation rules specific to customer_activity.csv.
    """

    print("Running customer activity validations...")

    # Check important service columns for missing values
    important_columns = [
        "phone_service",
        "internet_service",
    ]

    for column in important_columns:

        if df[column].isna().any():
            raise ValueError(
                f"customer_activity.csv: "
                f"{column} contains NULL values."
            )

    print("PASS: customer activity business rules")


def validate_customer_consistency(
    customer_master,
    customer_billing,
    customer_activity,
):
    """
    Verify that the same customers exist across all three datasets.
    """

    print("\nChecking customer IDs across datasets...")

    master_ids = set(
        customer_master["customer_id"]
    )

    billing_ids = set(
        customer_billing["customer_id"]
    )

    activity_ids = set(
        customer_activity["customer_id"]
    )

    if master_ids != billing_ids:
        missing_in_billing = master_ids - billing_ids
        extra_in_billing = billing_ids - master_ids

        raise ValueError(
            "Customer IDs differ between customer_master "
            "and customer_billing.\n"
            f"Missing in billing: {len(missing_in_billing)}\n"
            f"Extra in billing: {len(extra_in_billing)}"
        )

    if master_ids != activity_ids:
        missing_in_activity = master_ids - activity_ids
        extra_in_activity = activity_ids - master_ids

        raise ValueError(
            "Customer IDs differ between customer_master "
            "and customer_activity.\n"
            f"Missing in activity: {len(missing_in_activity)}\n"
            f"Extra in activity: {len(extra_in_activity)}"
        )

    print(
        f"PASS: All three datasets contain the same "
        f"{len(master_ids)} customers."
    )


def main():

    print("=" * 60)
    print("CUSTOMER CHURN SOURCE DATA VALIDATION")
    print("=" * 60)

    validated_data = {}

    # Common validation
    for filename, expected_columns in EXPECTED_SCHEMAS.items():

        validated_data[filename] = validate_file(
            filename,
            expected_columns
        )

    customer_master = validated_data[
        "customer_master.csv"
    ]

    customer_billing = validated_data[
        "customer_billing.csv"
    ]

    customer_activity = validated_data[
        "customer_activity.csv"
    ]

    # Dataset-specific validation
    validate_customer_master(
        customer_master
    )

    validate_customer_billing(
        customer_billing
    )

    validate_customer_activity(
        customer_activity
    )

    # Cross-dataset validation
    validate_customer_consistency(
        customer_master,
        customer_billing,
        customer_activity,
    )

    print("\n" + "=" * 60)
    print("ALL SOURCE DATA VALIDATION PASSED")
    print("=" * 60)


if __name__ == "__main__":
    main()