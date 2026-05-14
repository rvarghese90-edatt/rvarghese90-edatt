import pandas as pd
from src.utils import get_logger

logger = get_logger(__name__)

def preprocess_calls(df):
    logger.info("Processing call data")

    df["event_date"] = pd.to_datetime(df["event_date"])

    df_summary = (
        df[["unique_customer_identifier", "talk_time_seconds", "hold_time_seconds"]]
        .groupby("unique_customer_identifier")
        .sum()
        .reset_index()
    )

    logger.info(f"Call summary shape: {df_summary.shape}")
    return df_summary