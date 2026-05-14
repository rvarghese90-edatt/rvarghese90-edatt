from src.ingest import load_csv, load_parquet
from src.preprocess_cease import preprocess_cease
from src.preprocess_customer import preprocess_customer
from src.preprocess_calls import preprocess_calls
from src.preprocess_usage import preprocess_usage
from src.merge_features import merge_all

from src.model_preparation import split_features
from src.feature_selection import run_rfe
from src.model_training import train_models
from src.model_evaluation import save_results

from config.settings import OUTPUT_DIR
import os

def main():

    # Load raw data
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
    # Ensure target column is named correctly
    if "df_churn" in df_final.columns:
        df_final = df_final.rename(columns={"df_churn": "target"})
    
    df_final["target"] = df_final["target"].fillna(0).astype(int)
    df_final.drop(columns=["unique_customer_identifier", "datevalue"],inplace=True)

    # Save df_model
    model_path = os.path.join(OUTPUT_DIR, "df_model.csv")
    df_final.to_csv(model_path, index=False)
    print(f"Saved model dataset to {model_path}")

    # Identify categorical + numeric columns
    categorical, numeric = split_features(df_final)

    # Train models
    results, X_train, y_train = train_models(df_final, categorical, numeric)

    # Save results
    save_results(results)
    
    # RFE
    #rankings = run_rfe(X_train, y_train, categorical, numeric)
    #rankings.to_csv(os.path.join(OUTPUT_DIR, "feature_rankings.csv"), index=False)



    print("Modelling complete.")

if __name__ == "__main__":
    main()