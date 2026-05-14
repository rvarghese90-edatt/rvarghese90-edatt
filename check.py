from src.ingest import load_csv, load_parquet
from src.preprocess_cease import preprocess_cease
from src.preprocess_customer import preprocess_customer
from src.preprocess_calls import preprocess_calls
from src.preprocess_usage import preprocess_usage
from src.merge_features import merge_all
from src.model_training import train_churn_model
from config.settings import OUTPUT_DIR

import os

def main():

    # Load data
    df_cease = load_csv("cease.csv")
    df_customer = load_parquet("customer_info.parquet")
    df_calls = load_csv("calls.csv")
    df_usage = load_parquet("usage.parquet")

    # Preprocess
    df_cease_clean = preprocess_cease(df_cease)
    df_customer_clean = preprocess_customer(df_customer)
    df_calls_summary = preprocess_calls(df_calls)
    df_usage_summary = preprocess_usage(df_usage)

    # Merge
    df_final = merge_all(
        df_customer_clean,
        df_cease_clean,
        df_calls_summary,
        df_usage_summary
    )

    return df_final


df=main()

print(df.info())