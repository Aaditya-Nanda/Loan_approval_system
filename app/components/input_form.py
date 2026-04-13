# app/components/input_form.py

import streamlit as st

def render_input_form():
    """
    Renders the loan applicant input form.
    Returns a dictionary of user inputs.
    """

    st.subheader("📋 Applicant Information")

    with st.form("loan_form"):

        # ── Personal Information ──────────────────────────────
        st.markdown("#### 👤 Personal Details")
        col1, col2, col3 = st.columns(3)

        with col1:
            gender = st.selectbox("Gender", ["M", "F"])
            own_car = st.selectbox("Owns a Car?", ["Y", "N"])
            own_realty = st.selectbox("Owns Real Estate?", ["Y", "N"])

        with col2:
            cnt_children = st.number_input("Number of Children", 0, 20, 0)
            cnt_fam_members = st.number_input("Family Members", 1, 20, 2)
            age_years = st.number_input("Age (years)", 18, 100, 35)

        with col3:
            family_status = st.selectbox("Family Status", [
                "Married", "Single / not married",
                "Civil marriage", "Separated", "Widow"
            ])
            housing_type = st.selectbox("Housing Type", [
                "House / apartment", "With parents",
                "Municipal apartment", "Rented apartment",
                "Office apartment", "Co-op apartment"
            ])
            education = st.selectbox("Education", [
                "Higher education", "Secondary / secondary special",
                "Incomplete higher", "Lower secondary", "Academic degree"
            ])

        st.divider()

        # ── Financial Information ─────────────────────────────
        st.markdown("#### 💰 Financial Details")
        col4, col5 = st.columns(2)

        with col4:
            income = st.number_input(
                "Annual Income (₹)", 10000, 10000000, 180000, step=5000
            )
            annuity = st.number_input(
                "Loan Annuity (₹)", 1000, 500000, 20000, step=1000
            )
            income_type = st.selectbox("Income Type", [
                "Working", "Commercial associate",
                "Pensioner", "State servant", "Unemployed"
            ])

        with col5:
            occupation = st.selectbox("Occupation Type", [
                "Laborers", "Core staff", "Accountants", "Managers",
                "Drivers", "Sales staff", "Cleaning staff", "Cooking staff",
                "Private service staff", "Medicine staff", "Security staff",
                "High skill tech staff", "Waiters/barmen staff",
                "Low-skill Laborers", "Realty agents", "Secretaries",
                "IT staff", "HR staff"
            ])
            organization = st.selectbox("Organization Type", [
                "Business Entity Type 3", "School", "Government",
                "Religion", "Other", "Medicine", "Business Entity Type 2",
                "Self-employed", "Transport: type 2", "Construction",
                "Housing", "Kindergarten", "Trade: type 7"
            ])
            contract_type = st.selectbox("Contract Type", [
                "Cash loans", "Revolving loans"
            ])

        st.divider()

        # ── Credit History ────────────────────────────────────
        st.markdown("#### 📊 Credit & External Scores")
        col6, col7, col8 = st.columns(3)

        with col6:
            ext_source_1 = st.slider("External Score 1", 0.0, 1.0, 0.5)
        with col7:
            ext_source_2 = st.slider("External Score 2", 0.0, 1.0, 0.6)
        with col8:
            ext_source_3 = st.slider("External Score 3", 0.0, 1.0, 0.5)

        st.divider()

        # ── Employment ────────────────────────────────────────
        st.markdown("#### 💼 Employment Details")
        col9, col10 = st.columns(2)

        with col9:
            days_employed = st.number_input(
                "Years Employed", 0, 50, 5
            )
        with col10:
            region_rating = st.selectbox("Region Rating", [1, 2, 3])

        st.divider()

        submitted = st.form_submit_button(
            "🔍 Predict Loan Eligibility",
            use_container_width=True,
            type="primary"
        )

    if submitted:
        return {
            "NAME_CONTRACT_TYPE":           contract_type,
            "CODE_GENDER":                  gender,
            "FLAG_OWN_CAR":                 own_car,
            "FLAG_OWN_REALTY":              own_realty,
            "CNT_CHILDREN":                 cnt_children,
            "AMT_INCOME_TOTAL":             float(income),
            "AMT_ANNUITY":                  float(annuity),
            "NAME_TYPE_SUITE":              "Unaccompanied",
            "NAME_INCOME_TYPE":             income_type,
            "NAME_EDUCATION_TYPE":          education,
            "NAME_FAMILY_STATUS":           family_status,
            "NAME_HOUSING_TYPE":            housing_type,
            "DAYS_BIRTH":                   -(age_years * 365),
            "DAYS_EMPLOYED":                -(days_employed * 365),
            "FLAG_MOBIL":                   1,
            "FLAG_EMP_PHONE":               1,
            "FLAG_WORK_PHONE":              0,
            "FLAG_PHONE":                   1,
            "FLAG_EMAIL":                   0,
            "OCCUPATION_TYPE":              occupation,
            "CNT_FAM_MEMBERS":              float(cnt_fam_members),
            "REGION_RATING_CLIENT":         region_rating,
            "WEEKDAY_APPR_PROCESS_START":   "MONDAY",
            "HOUR_APPR_PROCESS_START":      10,
            "REG_REGION_NOT_LIVE_REGION":   0,
            "REG_REGION_NOT_WORK_REGION":   0,
            "LIVE_REGION_NOT_WORK_REGION":  0,
            "REG_CITY_NOT_LIVE_CITY":       0,
            "REG_CITY_NOT_WORK_CITY":       0,
            "LIVE_CITY_NOT_WORK_CITY":      0,
            "ORGANIZATION_TYPE":            organization,
            "EXT_SOURCE_1":                 ext_source_1,
            "EXT_SOURCE_2":                 ext_source_2,
            "EXT_SOURCE_3":                 ext_source_3,
            "OBS_30_CNT_SOCIAL_CIRCLE":     2.0,
            "DEF_30_CNT_SOCIAL_CIRCLE":     0.0,
            "OBS_60_CNT_SOCIAL_CIRCLE":     2.0,
            "DEF_60_CNT_SOCIAL_CIRCLE":     0.0,
            "DAYS_LAST_PHONE_CHANGE":       -300.0,
            "FLAG_DOCUMENT_3":              1,
            "AMT_REQ_CREDIT_BUREAU_YEAR":   1.0,
        }

    return None