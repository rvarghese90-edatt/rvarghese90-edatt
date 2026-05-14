import pandas as pd
from src.utils import get_logger

logger = get_logger(__name__)

def split_features(df):
    """
    Splits columns into categorical and numeric based on dtype.
    """
    categorical = df.select_dtypes(include=["object"]).columns.tolist()
    numeric = df.select_dtypes(include=["int64", "float64", "int32"]).columns.tolist()

    if "target" in numeric:
        numeric.remove("target")

    logger.info(f"Detected categorical: {categorical}")
    logger.info(f"Detected numeric: {numeric}")

    return categorical, numeric