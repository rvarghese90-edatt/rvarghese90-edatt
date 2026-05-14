import pandas as pd
from src.utils import get_logger

logger = get_logger(__name__)

def preprocess_customer(df):
    logger.info("Preparing customer data")

    df_sorted = df.sort_values(
        by=["unique_customer_identifier", "datevalue"],
        ascending=True
    )

    df_latest = df_sorted.drop_duplicates(
        subset=["unique_customer_identifier"],
        keep="last"
    )

    logger.info(f"Customer final shape: {df_latest.shape}")
    return df_latest