# src/train_classifier.py

import os
import yaml
import logging
import joblib
import numpy as np
import optuna
import warnings
warnings.filterwarnings("ignore")

from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import cross_val_score

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

# Silence Optuna's own logs — we handle logging ourselves
optuna.logging.set_verbosity(optuna.logging.WARNING)


# ─────────────────────────────────────────────────────────────
def objective(trial, X_train, y_train):
    """
    Optuna objective function.
    Each 'trial' is one experiment with a different set of hyperparameters.
    Optuna learns from each trial to suggest better params next time.
    """

    model_type = config["classifier"]["model_type"]

    if model_type == "xgboost":
        params = {
            "n_estimators":      trial.suggest_int("n_estimators", 100, 500),
            "max_depth":         trial.suggest_int("max_depth", 3, 10),
            "learning_rate":     trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "subsample":         trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree":  trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "min_child_weight":  trial.suggest_int("min_child_weight", 1, 10),
            "use_label_encoder": False,
            "eval_metric":       "logloss",
            "random_state":      config["data"]["random_state"],
            "n_jobs":            -1
        }
        model = XGBClassifier(**params)

    elif model_type == "random_forest":
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 100, 500),
            "max_depth":    trial.suggest_int("max_depth", 3, 20),
            "min_samples_split": trial.suggest_int("min_samples_split", 2, 10),
            "random_state": config["data"]["random_state"],
            "n_jobs":       -1
        }
        model = RandomForestClassifier(**params)

    # 3-fold cross validation score on training data
    scores = cross_val_score(
        model, X_train, y_train,
        cv=3,
        scoring="roc_auc",
        n_jobs=-1
    )
    return scores.mean()


# ─────────────────────────────────────────────────────────────
def train_classifier(X_train, y_train, X_test, y_clf_test):
    """
    Runs Optuna hyperparameter search, trains final model
    on best params, evaluates on test set, saves model.
    """

    logger.info("=" * 50)
    logger.info("Starting classifier training with Optuna...")
    logger.info(f"Model type : {config['classifier']['model_type']}")
    logger.info(f"Trials     : {config['classifier']['optuna']['n_trials']}")
    logger.info("=" * 50)

    # ── Optuna Study ──────────────────────────────────────────
    # A "study" is a collection of trials
    # direction="maximize" because higher ROC-AUC = better
    study = optuna.create_study(
        direction=config["classifier"]["optuna"]["direction"]
    )

    study.optimize(
        lambda trial: objective(trial, X_train, y_train),
        n_trials=config["classifier"]["optuna"]["n_trials"],
        show_progress_bar=True
    )

    best_params = study.best_params
    best_score  = study.best_value
    logger.info(f"Best ROC-AUC (CV): {best_score:.4f}")
    logger.info(f"Best params: {best_params}")

    # ── Train final model on full training set ────────────────
    logger.info("Training final model with best params...")

    model_type = config["classifier"]["model_type"]

    if model_type == "xgboost":
        best_params.update({
            "use_label_encoder": False,
            "eval_metric": "logloss",
            "random_state": config["data"]["random_state"],
            "n_jobs": -1
        })
        final_model = XGBClassifier(**best_params)

    elif model_type == "random_forest":
        best_params.update({
            "random_state": config["data"]["random_state"],
            "n_jobs": -1
        })
        final_model = RandomForestClassifier(**best_params)

    final_model.fit(X_train, y_train)

    # ── Evaluate on test set ──────────────────────────────────
    y_pred_proba = final_model.predict_proba(X_test)[:, 1]
    test_auc = roc_auc_score(y_clf_test, y_pred_proba)
    logger.info(f"Test ROC-AUC: {test_auc:.4f}")

    # ── Save model ────────────────────────────────────────────
    os.makedirs("models/classifier", exist_ok=True)
    joblib.dump(final_model, config["paths"]["classifier_model"])
    logger.info(f"Classifier saved to: {config['paths']['classifier_model']}")

    return final_model, test_auc


# ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    df = load_data()

    (X_train_clf, X_test,
     y_clf_train, y_clf_test,
     X_train_reg, _,
     y_reg_train, y_reg_test) = preprocess(df)

    model, auc = train_classifier(X_train_clf, y_clf_train, X_test, y_clf_test)
    print(f"\nFinal Test ROC-AUC: {auc:.4f}")