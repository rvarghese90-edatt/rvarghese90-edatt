import pandas as pd
from src.utils import get_logger

logger = get_logger(__name__)

def preprocess_cease(df):
    logger.info("Cleaning cease data")

    df["cease_placed_date"] = pd.to_datetime(df["cease_placed_date"])
    df["cease_completed_date"] = pd.to_datetime(df["cease_completed_date"])

    df_sorted = df.sort_values(
        by=["unique_customer_identifier", "cease_completed_date"],
        ascending=True
    )

    df_clean = df_sorted.drop_duplicates(
        subset=["unique_customer_identifier"],
        keep="last"
    )

    df_clean["df_churn"] = 1
    df_clean = df_clean.drop(columns=["cease_placed_date", "reason_description"])

    logger.info(f"Cease cleaned shape: {df_clean.shape}")
    return df_clean