# app/components/results_display.py

import streamlit as st

def render_results(result: dict, input_data: dict):
    """
    Renders the prediction result in a clean UI.
    """

    st.divider()
    st.subheader("📊 Prediction Results")

    decision = result["decision"]
    default_proba  = result["default_proba"]
    approval_proba = result["approval_proba"]
    amount         = result["recommended_amount"]

    # ── Decision Banner ───────────────────────────────────────
    if decision == "APPROVED":
        st.success("✅ Loan APPROVED", icon="✅")
    else:
        st.error("❌ Loan REJECTED", icon="❌")

    st.divider()

    # ── Probability Metrics ───────────────────────────────────
    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            label="✅ Approval Probability",
            value=f"{approval_proba:.2%}"
        )
    with col2:
        st.metric(
            label="⚠️ Default Risk",
            value=f"{default_proba:.2%}"
        )
    with col3:
        if amount:
            st.metric(
                label="💰 Recommended Loan Amount",
                value=f"₹{amount:,.0f}"
            )
        else:
            st.metric(
                label="💰 Recommended Loan Amount",
                value="N/A"
            )

    st.divider()

    # ── Risk Gauge ────────────────────────────────────────────
    st.markdown("#### 🎯 Risk Assessment")
    st.progress(
        value=approval_proba,
        text=f"Creditworthiness Score: {approval_proba:.2%}"
    )

    # ── Approval Details ──────────────────────────────────────
    if decision == "APPROVED" and amount:
        st.divider()
        st.markdown("#### 💼 Loan Offer Details")

        col4, col5, col6 = st.columns(3)
        annual_income = input_data.get("AMT_INCOME_TOTAL", 0)
        annuity       = input_data.get("AMT_ANNUITY", 0)

        with col4:
            st.info(f"**Recommended Amount**\n\n₹{amount:,.0f}")
        with col5:
            dti = (annuity / annual_income * 100) if annual_income > 0 else 0
            st.info(f"**Debt-to-Income Ratio**\n\n{dti:.1f}%")
        with col6:
            months = int(amount / annuity) if annuity > 0 else 0
            st.info(f"**Estimated Tenure**\n\n{months} months")

    # ── Rejection Advice ──────────────────────────────────────
    if decision == "REJECTED":
        st.divider()
        st.warning(
            "**Why was this loan rejected?**\n\n"
            "The model identified a high probability of default based on the "
            "provided financial profile. Common factors include low external "
            "credit scores, high debt-to-income ratio, or insufficient income "
            "relative to the requested loan amount.\n\n"
            "**Suggestions:** Improve credit scores, reduce existing debt, "
            "or apply for a lower loan amount."
        )