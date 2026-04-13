# src/predict_pipeline.py

import os
import sys
import yaml
import logging
import joblib
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore")

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
def load_artifacts():
    """
    Loads all saved model artifacts from disk.
    Called once when the app starts.
    """
    logger.info("Loading model artifacts...")

    classifier     = joblib.load(config["paths"]["classifier_model"])
    regressor      = joblib.load(config["paths"]["regressor_model"])
    label_encoders = joblib.load(config["paths"]["label_encoder"])
    scaler         = joblib.load(config["paths"]["scaler"])

    logger.info("All artifacts loaded successfully.")
    return classifier, regressor, label_encoders, scaler


# ─────────────────────────────────────────────────────────────
def preprocess_input(input_dict: dict, label_encoders: dict, scaler) -> pd.DataFrame:
    """
    Applies the same transformations used during training
    to a single new applicant's data.
    """

    df = pd.DataFrame([input_dict])

    # ── Drop columns not used during training ─────────────────
    drop_cols = config["data"]["drop_columns"] + \
                [config["data"]["target_classification"],
                 config["data"]["target_regression"]]

    for col in drop_cols:
        if col in df.columns:
            df = df.drop(columns=[col])

    # ── Encode categorical columns ────────────────────────────
    for col, le in label_encoders.items():
        if col in df.columns:
            val = str(df[col].iloc[0])
            if val not in le.classes_:
                val = le.classes_[0]
            df[col] = le.transform([val])

    # ── Align columns to match training feature set ───────────
    trained_features = scaler.feature_names_in_

    # Add any missing columns as 0
    for col in trained_features:
        if col not in df.columns:
            df[col] = 0

    # Keep only columns scaler knows, in correct order
    df = df[trained_features]

    # ── Impute any missing values ─────────────────────────────
    df = df.fillna(0)

    # ── Scale ─────────────────────────────────────────────────
    df[trained_features] = scaler.transform(df[trained_features])

    return df


# ─────────────────────────────────────────────────────────────
def predict(input_dict: dict) -> dict:
    """
    Full two-stage prediction pipeline:

    Stage 1 — Classifier:
        Predicts probability of loan default.
        If default probability >= threshold → REJECTED
        If default probability <  threshold → APPROVED → Stage 2

    Stage 2 — Regressor (only if approved):
        Predicts the optimal loan amount for the applicant.

    Returns a dictionary with full prediction details.
    """

    # ── Load artifacts ────────────────────────────────────────
    classifier, regressor, label_encoders, scaler = load_artifacts()

    # ── Preprocess input ──────────────────────────────────────
    X = preprocess_input(input_dict, label_encoders, scaler)

    # ── Stage 1: Classification ───────────────────────────────
    threshold      = config["thresholds"]["classification_threshold"]
    default_proba  = classifier.predict_proba(X)[:, 1][0]
    approval_proba = 1 - default_proba
    loan_decision  = "REJECTED" if default_proba >= threshold else "APPROVED"

    logger.info(f"Default probability : {default_proba:.4f}")
    logger.info(f"Decision            : {loan_decision}")

    result = {
        "decision":           loan_decision,
        "default_proba":      round(float(default_proba), 4),
        "approval_proba":     round(float(approval_proba), 4),
        "recommended_amount": None
    }

    # ── Stage 2: Regression (approved only) ───────────────────
    if loan_decision == "APPROVED":
        recommended_amount = regressor.predict(X)[0]
        recommended_amount = max(0, recommended_amount)
        result["recommended_amount"] = round(float(recommended_amount), 2)
        logger.info(f"Recommended amount  : {recommended_amount:,.2f}")

    return result


# ─────────────────────────────────────────────────────────────
if __name__ == "__main__":

    print("=" * 50)
    print("Starting prediction test...")
    print("=" * 50)

    sample_input = {
        "NAME_CONTRACT_TYPE":         "Cash loans",
        "CODE_GENDER":                "M",
        "FLAG_OWN_CAR":               "Y",
        "FLAG_OWN_REALTY":            "Y",
        "CNT_CHILDREN":               0,
        "AMT_INCOME_TOTAL":           180000.0,
        "AMT_ANNUITY":                20000.0,
        "NAME_TYPE_SUITE":            "Unaccompanied",
        "NAME_INCOME_TYPE":           "Working",
        "NAME_EDUCATION_TYPE":        "Higher education",
        "NAME_FAMILY_STATUS":         "Married",
        "NAME_HOUSING_TYPE":          "House / apartment",
        "DAYS_BIRTH":                 -12000,
        "DAYS_EMPLOYED":              -3000,
        "FLAG_MOBIL":                 1,
        "FLAG_EMP_PHONE":             1,
        "FLAG_WORK_PHONE":            0,
        "FLAG_PHONE":                 1,
        "FLAG_EMAIL":                 0,
        "OCCUPATION_TYPE":            "Laborers",
        "CNT_FAM_MEMBERS":            2.0,
        "REGION_RATING_CLIENT":       2,
        "WEEKDAY_APPR_PROCESS_START": "MONDAY",
        "HOUR_APPR_PROCESS_START":    10,
        "REG_REGION_NOT_LIVE_REGION": 0,
        "REG_REGION_NOT_WORK_REGION": 0,
        "LIVE_REGION_NOT_WORK_REGION": 0,
        "REG_CITY_NOT_LIVE_CITY":     0,
        "REG_CITY_NOT_WORK_CITY":     0,
        "LIVE_CITY_NOT_WORK_CITY":    0,
        "ORGANIZATION_TYPE":          "Business Entity Type 3",
        "EXT_SOURCE_1":               0.5,
        "EXT_SOURCE_2":               0.6,
        "EXT_SOURCE_3":               0.5,
        "OBS_30_CNT_SOCIAL_CIRCLE":   2.0,
        "DEF_30_CNT_SOCIAL_CIRCLE":   0.0,
        "OBS_60_CNT_SOCIAL_CIRCLE":   2.0,
        "DEF_60_CNT_SOCIAL_CIRCLE":   0.0,
        "DAYS_LAST_PHONE_CHANGE":     -300.0,
        "FLAG_DOCUMENT_3":            1,
        "AMT_REQ_CREDIT_BUREAU_YEAR": 1.0,
    }

    result = predict(sample_input)

    print("\n── Prediction Result ──")
    print(f"Decision            : {result['decision']}")
    print(f"Default Probability : {result['default_proba']:.2%}")
    print(f"Approval Probability: {result['approval_proba']:.2%}")

    if result["recommended_amount"]:
        print(f"Recommended Amount  : {result['recommended_amount']:,.2f}")
    else:
        print("Recommended Amount  : N/A (Loan Rejected)")

    print("=" * 50)