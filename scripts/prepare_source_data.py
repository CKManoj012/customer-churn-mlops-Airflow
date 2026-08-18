from pathlib import Path

import pandas as pd


SOURCE_FILE = Path("data/local/source/telco_customer_churn.csv")
OUTPUT_DIR = Path("data/local")


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(SOURCE_FILE)

    print(f"Original dataset shape: {df.shape}")
    print("\nColumns:")
    print(df.columns.tolist())

    # Normalize common IBM Telco dataset column names
    df = df.rename(
        columns={
            "customerID": "customer_id",
            "gender": "gender",
            "SeniorCitizen": "senior_citizen",
            "Partner": "partner",
            "Dependents": "dependents",
            "tenure": "tenure",
            "PhoneService": "phone_service",
            "MultipleLines": "multiple_lines",
            "InternetService": "internet_service",
            "OnlineSecurity": "online_security",
            "OnlineBackup": "online_backup",
            "DeviceProtection": "device_protection",
            "TechSupport": "tech_support",
            "StreamingTV": "streaming_tv",
            "StreamingMovies": "streaming_movies",
            "Contract": "contract_type",
            "PaperlessBilling": "paperless_billing",
            "PaymentMethod": "payment_method",
            "MonthlyCharges": "monthly_charge",
            "TotalCharges": "total_charge",
            "Churn": "churn",
        }
    )

    # Convert churn to binary
    df["churn"] = (
        df["churn"]
        .astype(str)
        .str.strip()
        .str.lower()
        .map({"yes": 1, "no": 0})
    )

    # TotalCharges can contain blank strings
    df["total_charge"] = pd.to_numeric(
        df["total_charge"],
        errors="coerce"
    )

    # ---------------------------------------------------
    # Dataset 1: Customer Master
    # ---------------------------------------------------

    customer_master_columns = [
        "customer_id",
        "gender",
        "senior_citizen",
        "partner",
        "dependents",
        "tenure",
        "contract_type",
    ]

    customer_master = df[customer_master_columns].copy()

    # ---------------------------------------------------
    # Dataset 2: Customer Activity
    # ---------------------------------------------------

    customer_activity_columns = [
        "customer_id",
        "phone_service",
        "multiple_lines",
        "internet_service",
        "online_security",
        "online_backup",
        "device_protection",
        "tech_support",
        "streaming_tv",
        "streaming_movies",
    ]

    customer_activity = df[customer_activity_columns].copy()

    # ---------------------------------------------------
    # Dataset 3: Customer Billing
    # ---------------------------------------------------

    customer_billing_columns = [
        "customer_id",
        "paperless_billing",
        "payment_method",
        "monthly_charge",
        "total_charge",
        "churn",
    ]

    customer_billing = df[customer_billing_columns].copy()

    # Save
    customer_master.to_csv(
        OUTPUT_DIR / "customer_master.csv",
        index=False
    )

    customer_activity.to_csv(
        OUTPUT_DIR / "customer_activity.csv",
        index=False
    )

    customer_billing.to_csv(
        OUTPUT_DIR / "customer_billing.csv",
        index=False
    )

    print("\nCreated datasets:")
    print(f"customer_master   : {customer_master.shape}")
    print(f"customer_activity : {customer_activity.shape}")
    print(f"customer_billing  : {customer_billing.shape}")


if __name__ == "__main__":
    main()