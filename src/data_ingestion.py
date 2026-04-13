# src/data_ingestion.py
import os
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import pandas as pd
import logging
import yaml
import sys

# ── Load config ───────────────────────────────────────────────
with open("config/config.yaml", "r") as f:
    config = yaml.safe_load(f)

# ── Set up logging ────────────────────────────────────────────
logging.basicConfig(
    level=config["logging"]["level"],
    format=config["logging"]["format"]
)
logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────
def load_data() -> pd.DataFrame:
    """
    Loads raw CSV from the path defined in config.yaml.
    Performs basic validation and logs a summary.
    Returns a pandas DataFrame.
    """

    raw_path = config["paths"]["raw_data"]

    # ── 1. Check file exists ──────────────────────────────────
    if not os.path.exists(raw_path):
        logger.error(f"Data file not found at: {raw_path}")
        sys.exit(1)                          # Stop everything — no point continuing

    # ── 2. Load CSV ───────────────────────────────────────────
    logger.info(f"Loading data from: {raw_path}")
    df = pd.read_csv(raw_path)
    logger.info(f"Data loaded successfully — Shape: {df.shape}")

    # ── 3. Basic validation ───────────────────────────────────
    target_col = config["data"]["target_classification"]

    if target_col not in df.columns:
        logger.error(f"Target column '{target_col}' not found in dataset.")
        sys.exit(1)

    # ── 4. Log a summary ──────────────────────────────────────
    logger.info(f"Columns       : {df.shape[1]}")
    logger.info(f"Rows          : {df.shape[0]}")
    logger.info(f"Missing values: {df.isnull().sum().sum()}")
    logger.info(f"Target distribution:\n{df[target_col].value_counts()}")

    return df


# ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    # Run this file directly to test it works
    df = load_data()
    print("\nFirst 5 rows:")
    print(df.head())