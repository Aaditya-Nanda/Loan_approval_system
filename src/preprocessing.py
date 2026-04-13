# src/preprocessing.py

import pandas as pd
import numpy as np
import yaml
import logging
import os
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from imblearn.over_sampling import SMOTE

# ── Fix working directory ─────────────────────────────────────
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# ── Load config ───────────────────────────────────────────────
with open("config/config.yaml", "r") as f:
    config = yaml.safe_load(f)

# ── Logging ───────────────────────────────────────────────────
logging.basicConfig(
    level=config["logging"]["level"],
    format=config["logging"]["format"]
)
logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────
def preprocess(df: pd.DataFrame):
    """
    Full preprocessing pipeline:
    1. Drop irrelevant columns
    2. Encode categorical features
    3. Impute missing values
    4. Scale numerical features
    5. Split into train/test
    6. Apply SMOTE on training set only
    Returns train/test splits for both classifier and regressor.
    """

    logger.info("Starting preprocessing...")

    # ── 1. Drop configured columns ────────────────────────────
    drop_cols = config["data"]["drop_columns"]
    # Only drop columns that actually exist
    drop_cols = [c for c in drop_cols if c in df.columns]
    df = df.drop(columns=drop_cols)
    logger.info(f"Dropped columns: {drop_cols}")

    # ── 2. Separate targets ───────────────────────────────────
    clf_target = config["data"]["target_classification"]   # TARGET (0/1)
    reg_target = config["data"]["target_regression"]       # AMT_CREDIT

    y_clf = df[clf_target].copy()
    y_reg = df[reg_target].copy()

    # Drop both targets from features
    df = df.drop(columns=[clf_target, reg_target])
    logger.info(f"Targets separated. Feature shape: {df.shape}")

    # ── 3. Encode categorical columns ─────────────────────────
    categorical_cols = df.select_dtypes(include=["object"]).columns.tolist()
    logger.info(f"Encoding {len(categorical_cols)} categorical columns...")

    label_encoders = {}
    for col in categorical_cols:
        le = LabelEncoder()
        # Handle NaN before encoding by filling with "Missing"
        df[col] = df[col].fillna("Missing")
        df[col] = le.fit_transform(df[col].astype(str))
        label_encoders[col] = le

    # Save label encoders for inference later
    joblib.dump(label_encoders, config["paths"]["label_encoder"])
    logger.info("Label encoders saved.")

    # ── 4. Impute missing values ──────────────────────────────
    num_strategy = config["preprocessing"]["numerical_impute_strategy"]
    logger.info(f"Imputing numerical columns with strategy: {num_strategy}")

    numerical_cols = df.select_dtypes(include=["int64", "float64"]).columns.tolist()
    num_imputer = SimpleImputer(strategy=num_strategy)
    df[numerical_cols] = num_imputer.fit_transform(df[numerical_cols])
    logger.info("Imputation complete.")

    # ── 5. Scale features ─────────────────────────────────────
    if config["preprocessing"]["scaling"]:
        logger.info("Scaling features...")
        scaler = StandardScaler()
        df[numerical_cols] = scaler.fit_transform(df[numerical_cols])
        joblib.dump(scaler, config["paths"]["scaler"])
        logger.info("Scaler saved.")

    # ── 6. Train/test split ───────────────────────────────────
    test_size    = config["data"]["test_size"]
    random_state = config["data"]["random_state"]

    X_train, X_test, y_clf_train, y_clf_test, y_reg_train, y_reg_test = \
        train_test_split(
            df, y_clf, y_reg,
            test_size=test_size,
            random_state=random_state,
            stratify=y_clf          # keeps class ratio in both splits
        )

    logger.info(f"Train size: {X_train.shape}, Test size: {X_test.shape}")
    logger.info(f"Class distribution before SMOTE: {y_clf_train.value_counts().to_dict()}")

    # ── 7. SMOTE — only on training data ─────────────────────
    # NEVER apply SMOTE to test data — that would leak information
    if config["preprocessing"]["smote"]["apply"]:
        logger.info("Applying SMOTE to training set...")
        smote = SMOTE(
            sampling_strategy=config["preprocessing"]["smote"]["sampling_strategy"],
            random_state=config["preprocessing"]["smote"]["random_state"],
            k_neighbors=config["preprocessing"]["smote"]["k_neighbors"]
        )
        X_train_clf, y_clf_train = smote.fit_resample(X_train, y_clf_train)
        logger.info(f"Class distribution after SMOTE: {pd.Series(y_clf_train).value_counts().to_dict()}")
    else:
        X_train_clf = X_train

    # ── 8. Save processed data ────────────────────────────────
    os.makedirs("data/processed", exist_ok=True)
    df_processed = df.copy()
    df_processed[clf_target] = y_clf.values
    df_processed.to_csv(config["paths"]["processed_data"], index=False)
    logger.info(f"Processed data saved to: {config['paths']['processed_data']}")

    logger.info("Preprocessing complete.")

    return (
        X_train_clf, X_test,        # classifier features
        y_clf_train, y_clf_test,    # classifier targets
        X_train,     X_test,        # regressor features (no SMOTE)
        y_reg_train, y_reg_test     # regressor targets
    )


# ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    from data_ingestion import load_data
    df = load_data()
    results = preprocess(df)
    X_train_clf, X_test, y_clf_train, y_clf_test, \
    X_train_reg, _, y_reg_train, y_reg_test = results

    print(f"\nClassifier train shape : {X_train_clf.shape}")
    print(f"Classifier test shape  : {X_test.shape}")
    print(f"Regressor train shape  : {X_train_reg.shape}")