# src/train_regressor.py

import os
import yaml
import logging
import joblib
import numpy as np
import optuna
import warnings
warnings.filterwarnings("ignore")

from xgboost import XGBRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, r2_score
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

optuna.logging.set_verbosity(optuna.logging.WARNING)


# ─────────────────────────────────────────────────────────────
def objective(trial, X_train, y_train):
    """
    Optuna objective for regression.
    We minimize RMSE — lower is better.
    """

    model_type = config["regressor"]["model_type"]

    if model_type == "xgboost":
        params = {
            "n_estimators":     trial.suggest_int("n_estimators", 100, 500),
            "max_depth":        trial.suggest_int("max_depth", 3, 10),
            "learning_rate":    trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "subsample":        trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
            "random_state":     config["data"]["random_state"],
            "n_jobs":           -1
        }
        model = XGBRegressor(**params)

    elif model_type == "ridge":
        params = {
            "alpha": trial.suggest_float("alpha", 0.01, 100.0, log=True)
        }
        model = Ridge(**params)

    # Negative RMSE because cross_val_score maximizes by default
    scores = cross_val_score(
        model, X_train, y_train,
        cv=3,
        scoring="neg_root_mean_squared_error",
        n_jobs=-1
    )
    return scores.mean()   # will be negative; Optuna maximizes this = minimizes RMSE


# ─────────────────────────────────────────────────────────────
def train_regressor(X_train, y_train, X_test, y_reg_test):
    """
    Runs Optuna search, trains final regressor,
    evaluates on test set, saves model.
    Only runs on approved loans (TARGET == 0).
    """

    logger.info("=" * 50)
    logger.info("Starting regressor training with Optuna...")
    logger.info(f"Model type : {config['regressor']['model_type']}")
    logger.info(f"Trials     : {config['regressor']['optuna']['n_trials']}")
    logger.info("=" * 50)

    # ── Optuna Study ──────────────────────────────────────────
    # direction="maximize" because we're maximizing negative RMSE
    study = optuna.create_study(
        direction=config["regressor"]["optuna"]["direction"]
    )

    study.optimize(
        lambda trial: objective(trial, X_train, y_train),
        n_trials=config["regressor"]["optuna"]["n_trials"],
        show_progress_bar=True
    )

    best_params = study.best_params
    best_score  = study.best_value
    logger.info(f"Best CV RMSE : {abs(best_score):,.2f}")
    logger.info(f"Best params  : {best_params}")

    # ── Train final model ─────────────────────────────────────
    logger.info("Training final regressor with best params...")

    model_type = config["regressor"]["model_type"]

    if model_type == "xgboost":
        best_params.update({
            "random_state": config["data"]["random_state"],
            "n_jobs": -1
        })
        final_model = XGBRegressor(**best_params)

    elif model_type == "ridge":
        final_model = Ridge(**best_params)

    final_model.fit(X_train, y_train)

    # ── Evaluate ──────────────────────────────────────────────
    y_pred = final_model.predict(X_test)

    rmse = np.sqrt(mean_squared_error(y_reg_test, y_pred))
    r2   = r2_score(y_reg_test, y_pred)

    logger.info(f"Test RMSE : {rmse:,.2f}")
    logger.info(f"Test R²   : {r2:.4f}")

    # ── Save model ────────────────────────────────────────────
    os.makedirs("models/regressor", exist_ok=True)
    joblib.dump(final_model, config["paths"]["regressor_model"])
    logger.info(f"Regressor saved to: {config['paths']['regressor_model']}")

    return final_model, rmse, r2


# ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    df = load_data()

    (X_train_clf, X_test,
     y_clf_train, y_clf_test,
     X_train_reg, _,
     y_reg_train, y_reg_test) = preprocess(df)

    model, rmse, r2 = train_regressor(X_train_reg, y_reg_train, X_test, y_reg_test)

    print(f"\nFinal Test RMSE : {rmse:,.2f}")
    print(f"Final Test R²   : {r2:.4f}")