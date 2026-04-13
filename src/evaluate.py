# src/evaluate.py

import os
import yaml
import logging
import joblib
import numpy as np
import matplotlib
matplotlib.use("Agg")   # non-interactive backend, safe for all environments
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

from sklearn.metrics import (
    roc_auc_score, roc_curve,
    confusion_matrix, classification_report,
    mean_squared_error, r2_score
)

from data_ingestion import load_data
from preprocessing import preprocess

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

os.makedirs("reports", exist_ok=True)


# ─────────────────────────────────────────────────────────────
def evaluate_classifier(model, X_test, y_test):
    """
    Full classifier evaluation:
    - ROC-AUC score
    - Confusion matrix
    - Classification report
    - ROC curve plot saved to reports/
    """

    logger.info("=" * 50)
    logger.info("Evaluating Classifier...")
    logger.info("=" * 50)

    threshold = config["thresholds"]["classification_threshold"]

    # ── Predictions ───────────────────────────────────────────
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    y_pred       = (y_pred_proba >= threshold).astype(int)

    # ── Metrics ───────────────────────────────────────────────
    auc = roc_auc_score(y_test, y_pred_proba)
    cm  = confusion_matrix(y_test, y_pred)
    cr  = classification_report(y_test, y_pred, target_names=["Approved", "Rejected"])

    logger.info(f"ROC-AUC Score : {auc:.4f}")
    logger.info(f"Threshold used: {threshold}")
    logger.info(f"\nConfusion Matrix:\n{cm}")
    logger.info(f"\nClassification Report:\n{cr}")

    # ── Unpack confusion matrix ───────────────────────────────
    tn, fp, fn, tp = cm.ravel()
    logger.info(f"True Negatives  (Correctly Approved) : {tn:,}")
    logger.info(f"False Positives (Wrongly Rejected)   : {fp:,}")
    logger.info(f"False Negatives (Missed Defaults)    : {fn:,}")
    logger.info(f"True Positives  (Correctly Rejected) : {tp:,}")

    # ── ROC Curve Plot ────────────────────────────────────────
    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)

    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color="steelblue", lw=2,
             label=f"ROC Curve (AUC = {auc:.4f})")
    plt.plot([0, 1], [0, 1], color="gray", linestyle="--", label="Random Classifier")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve — Loan Default Classifier")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig("reports/roc_curve.png", dpi=150)
    plt.close()
    logger.info("ROC curve saved to: reports/roc_curve.png")

    return auc, cm


# ─────────────────────────────────────────────────────────────
def evaluate_regressor(model, X_test, y_test):
    """
    Full regressor evaluation:
    - RMSE, MAE, R² score
    - Actual vs Predicted plot saved to reports/
    """

    logger.info("=" * 50)
    logger.info("Evaluating Regressor...")
    logger.info("=" * 50)

    # ── Predictions ───────────────────────────────────────────
    y_pred = model.predict(X_test)

    # ── Metrics ───────────────────────────────────────────────
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae  = np.mean(np.abs(y_test - y_pred))
    r2   = r2_score(y_test, y_pred)

    logger.info(f"RMSE : {rmse:,.2f}")
    logger.info(f"MAE  : {mae:,.2f}")
    logger.info(f"R²   : {r2:.4f}")

    # ── Actual vs Predicted Plot ──────────────────────────────
    sample_size = min(1000, len(y_test))
    indices     = np.random.choice(len(y_test), sample_size, replace=False)

    plt.figure(figsize=(8, 6))
    plt.scatter(
        np.array(y_test)[indices],
        y_pred[indices],
        alpha=0.4, color="steelblue", s=15
    )
    plt.plot(
        [y_test.min(), y_test.max()],
        [y_test.min(), y_test.max()],
        color="red", linestyle="--", lw=2, label="Perfect Prediction"
    )
    plt.xlabel("Actual Loan Amount")
    plt.ylabel("Predicted Loan Amount")
    plt.title("Actual vs Predicted Loan Amount")
    plt.legend()
    plt.tight_layout()
    plt.savefig("reports/actual_vs_predicted.png", dpi=150)
    plt.close()
    logger.info("Actual vs Predicted plot saved to: reports/actual_vs_predicted.png")

    return rmse, mae, r2


# ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    # Load data and preprocess
    df = load_data()

    (X_train_clf, X_test,
     y_clf_train, y_clf_test,
     X_train_reg, _,
     y_reg_train, y_reg_test) = preprocess(df)

    # Load saved models
    classifier = joblib.load(config["paths"]["classifier_model"])
    regressor  = joblib.load(config["paths"]["regressor_model"])

    # Evaluate both
    auc, cm        = evaluate_classifier(classifier, X_test, y_clf_test)
    rmse, mae, r2  = evaluate_regressor(regressor, X_test, y_reg_test)

    print("\n── Final Evaluation Summary ──")
    print(f"Classifier  ROC-AUC : {auc:.4f}")
    print(f"Regressor   RMSE    : {rmse:,.2f}")
    print(f"Regressor   R²      : {r2:.4f}")