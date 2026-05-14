import os
import pandas as pd
from config.settings import DATA_DIR
from src.utils import get_logger

logger = get_logger(__name__)

def load_csv(filename):
    path = os.path.join(DATA_DIR, filename)
    logger.info(f"Loading CSV: {path}")
    df = pd.read_csv(path)
    logger.info(f"Loaded shape: {df.shape}")
    return df

def load_parquet(filename):
    path = os.path.join(DATA_DIR, filename)
    logger.info(f"Loading Parquet: {path}")
    df = pd.read_parquet(path)
    logger.info(f"Loaded shape: {df.shape}")
    return df