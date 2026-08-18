import sys

from awsglue.context import GlueContext
from awsglue.job import Job
from awsglue.utils import getResolvedOptions

from pyspark.context import SparkContext
from pyspark.sql import functions as F
from pyspark.sql.types import (
    IntegerType,
    DoubleType,
)


# ---------------------------------------------------------
# Glue job arguments
# ---------------------------------------------------------

args = getResolvedOptions(
    sys.argv,
    [
        "JOB_NAME",
        "SOURCE_BUCKET",
        "OUTPUT_PATH",
    ],
)


SOURCE_BUCKET = args["SOURCE_BUCKET"]
OUTPUT_PATH = args["OUTPUT_PATH"]


# ---------------------------------------------------------
# Spark / Glue setup
# ---------------------------------------------------------

sc = SparkContext()

glue_context = GlueContext(sc)

spark = glue_context.spark_session

job = Job(glue_context)

job.init(
    args["JOB_NAME"],
    args,
)


# ---------------------------------------------------------
# Source paths
# ---------------------------------------------------------

MASTER_PATH = (
    f"s3://{SOURCE_BUCKET}/raw/customer_master/"
)

ACTIVITY_PATH = (
    f"s3://{SOURCE_BUCKET}/raw/customer_activity/"
)

BILLING_PATH = (
    f"s3://{SOURCE_BUCKET}/raw/customer_billing/"
)


print("Reading source datasets...")

print(f"Customer master: {MASTER_PATH}")
print(f"Customer activity: {ACTIVITY_PATH}")
print(f"Customer billing: {BILLING_PATH}")


# ---------------------------------------------------------
# Read CSV files
# ---------------------------------------------------------

customer_master = (
    spark.read
    .option("header", True)
    .option("inferSchema", True)
    .csv(MASTER_PATH)
)


customer_activity = (
    spark.read
    .option("header", True)
    .option("inferSchema", True)
    .csv(ACTIVITY_PATH)
)


customer_billing = (
    spark.read
    .option("header", True)
    .option("inferSchema", True)
    .csv(BILLING_PATH)
)


print(
    "Customer master rows:",
    customer_master.count(),
)

print(
    "Customer activity rows:",
    customer_activity.count(),
)

print(
    "Customer billing rows:",
    customer_billing.count(),
)


# ---------------------------------------------------------
# Customer master cleaning
# ---------------------------------------------------------

customer_master = (
    customer_master

    # Standardize customer ID
    .withColumn(
        "customer_id",
        F.trim(F.col("customer_id")),
    )

    # Explicit type conversion
    .withColumn(
        "age",
        F.col("age").cast(IntegerType()),
    )

    .withColumn(
        "number_of_dependents",
        F.col("number_of_dependents")
        .cast(IntegerType()),
    )

    # Remove rows without customer ID
    .filter(
        F.col("customer_id").isNotNull()
    )

    .filter(
        F.col("customer_id") != ""
    )

    # Deduplicate
    .dropDuplicates(
        ["customer_id"]
    )
)


# ---------------------------------------------------------
# Customer activity cleaning
# ---------------------------------------------------------

customer_activity = (
    customer_activity

    .withColumn(
        "customer_id",
        F.trim(F.col("customer_id")),
    )

    .filter(
        F.col("customer_id").isNotNull()
    )

    .filter(
        F.col("customer_id") != ""
    )

    .dropDuplicates(
        ["customer_id"]
    )
)


# ---------------------------------------------------------
# Customer billing cleaning
# ---------------------------------------------------------

customer_billing = (
    customer_billing

    .withColumn(
        "customer_id",
        F.trim(F.col("customer_id")),
    )

    .withColumn(
        "monthly_charge",
        F.col("monthly_charge")
        .cast(DoubleType()),
    )

    .withColumn(
        "total_charges",
        F.col("total_charges")
        .cast(DoubleType()),
    )

    .withColumn(
        "churn_value",
        F.col("churn_value")
        .cast(IntegerType()),
    )

    .filter(
        F.col("customer_id").isNotNull()
    )

    .filter(
        F.col("customer_id") != ""
    )

    .dropDuplicates(
        ["customer_id"]
    )
)


# ---------------------------------------------------------
# Missing value handling
# ---------------------------------------------------------

customer_master = (
    customer_master
    .fillna(
        {
            "number_of_dependents": 0,
        }
    )
)


customer_billing = (
    customer_billing
    .fillna(
        {
            "total_charges": 0.0,
        }
    )
)


# ---------------------------------------------------------
# Join datasets
# ---------------------------------------------------------

print("Joining datasets...")


customer_dataset = (
    customer_master

    .join(
        customer_activity,
        on="customer_id",
        how="inner",
    )

    .join(
        customer_billing,
        on="customer_id",
        how="inner",
    )
)


# ---------------------------------------------------------
# Basic final validation
# ---------------------------------------------------------

customer_dataset = (
    customer_dataset

    .filter(
        F.col("monthly_charge") >= 0
    )

    .filter(
        F.col("total_charges") >= 0
    )

    .filter(
        F.col("churn_value").isin(
            0,
            1,
        )
    )
)


final_count = customer_dataset.count()

print(
    f"Final processed row count: {final_count}"
)


# ---------------------------------------------------------
# Print schema
# ---------------------------------------------------------

customer_dataset.printSchema()


# ---------------------------------------------------------
# Write Parquet
# ---------------------------------------------------------

print(
    f"Writing processed dataset to: "
    f"{OUTPUT_PATH}"
)


(
    customer_dataset
    .write
    .mode("overwrite")
    .parquet(
        OUTPUT_PATH
    )
)


print(
    "Customer churn ETL completed successfully."
)


job.commit()