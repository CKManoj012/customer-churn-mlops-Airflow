import pandas as pd

df = pd.read_parquet(
    "data/processed/customer_churn/"
)

print(df.shape)
print(df.head())
print(df.dtypes)
print(df.isnull().sum())
print(df["churn_value"].value_counts())