# app/streamlit_app.py

import sys
import os

# ── Path setup so app can find src/ modules ───────────────────
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
os.chdir(os.path.join(os.path.dirname(__file__), '..'))

import streamlit as st
from components.input_form import render_input_form
from components.results_display import render_results
from predict_pipeline import predict

# ── Page config ───────────────────────────────────────────────
st.set_page_config(
    page_title="Loan Approval System",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ── Header ────────────────────────────────────────────────────
st.title("🏦 Loan Approval & Valuation System")
st.markdown(
    "A two-stage ML system that predicts **loan approval** and "
    "recommends an **optimal loan amount** for approved applicants."
)
st.divider()

# ── Sidebar info ──────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ℹ️ About")
    st.markdown(
        "This system uses two ML models:\n\n"
        "**Stage 1 — Classifier**\n"
        "XGBoost model trained on 300K+ loan applications "
        "to predict default risk.\n\n"
        "**Stage 2 — Regressor**\n"
        "XGBoost model that recommends the optimal "
        "loan amount for approved applicants.\n\n"
        "**Dataset:** Home Credit Default Risk (Kaggle)"
    )
    st.divider()
    st.markdown("**Model Performance**")
    st.metric("Classifier ROC-AUC", "0.7418")
    st.metric("Regressor R²", "0.7099")

# ── Main form ─────────────────────────────────────────────────
input_data = render_input_form()

# ── Prediction ────────────────────────────────────────────────
if input_data is not None:
    with st.spinner("🔄 Running prediction pipeline..."):
        try:
            result = predict(input_data)
            render_results(result, input_data)
        except Exception as e:
            st.error(f"Prediction error: {e}")
            st.exception(e)