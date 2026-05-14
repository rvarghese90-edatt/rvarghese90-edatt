import pandas as pd
from src.utils import get_logger

logger = get_logger(__name__)

def preprocess_usage(df):
    logger.info("Processing broadband usage")

    # Clean weird characters and whitespace
    df["usage_download_mbs"] = (
        df["usage_download_mbs"]
        .astype(str)
        .str.replace(",", "", regex=False)
        .str.strip()
        .replace(["", " ", "None", "nan", "NaN", "--"], pd.NA)
    )

    df["usage_upload_mbs"] = (
        df["usage_upload_mbs"]
        .astype(str)
        .str.replace(",", "", regex=False)
        .str.strip()
        .replace(["", " ", "None", "nan", "NaN", "--"], pd.NA)
    )

    # Convert to numeric safely
    df["usage_download_mbs"] = pd.to_numeric(df["usage_download_mbs"], errors="coerce")
    df["usage_upload_mbs"] = pd.to_numeric(df["usage_upload_mbs"], errors="coerce")

    # Keep as float — broadband usage can be large and float is safer
    df_summary = (
        df.groupby("unique_customer_identifier")[["usage_download_mbs", "usage_upload_mbs"]]
        .sum(min_count=1)
        .reset_index()
    )

    logger.info(f"Usage summary shape: {df_summary.shape}")
    return df_summary