# ==========================================================
# DischargeFlow AI
# Live Discharge Risk Command Center
# Classification + Regression Integrated
# Deployment-Ready Version (Streamlit Cloud Safe)
# ==========================================================

import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from openai import OpenAI
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer

# ----------------------------------------------------------
# Page Config
# ----------------------------------------------------------

st.set_page_config(page_title="DischargeFlow AI", layout="wide")
st.title("DischargeFlow AI")
st.caption("Live discharge delay monitoring and operational command board.")

# ----------------------------------------------------------
# OpenAI Configuration (Streamlit Cloud Safe)
# ----------------------------------------------------------

if "OPENAI_API_KEY" not in st.secrets:
    st.error("OpenAI API key not configured in Streamlit Secrets.")
    st.stop()

client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

# ----------------------------------------------------------
# Load Data
# ----------------------------------------------------------

@st.cache_data
def load_data():
    df = pd.read_csv("fictitious_dataset_FINAL.csv")
    df.columns = df.columns.str.strip()
    df["delayed_over_200mins"] = (
        df["Discharge Duration (minutes)"] > 200
    ).astype(int)
    return df

df = load_data()

# ----------------------------------------------------------
# Train Models (Classifier + Regressor)
# ----------------------------------------------------------

@st.cache_resource
def train_models(df):

    target_class = "delayed_over_200mins"
    target_reg = "Discharge Duration (minutes)"

    X = df.drop(columns=[target_class, target_reg])
    y_class = df[target_class]
    y_reg = df[target_reg]

    numeric_features = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
    categorical_features = X.select_dtypes(include=["object"]).columns.tolist()

    preprocessor = ColumnTransformer([
        ("num", Pipeline([
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler())
        ]), numeric_features),
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False),
         categorical_features)
    ])

    clf_model = Pipeline([
        ("pre", preprocessor),
        ("clf", GradientBoostingClassifier(random_state=42))
    ])

    reg_model = Pipeline([
        ("pre", preprocessor),
        ("reg", GradientBoostingRegressor(random_state=42))
    ])

    X_train, X_test, y_train_class, y_test_class = train_test_split(
        X, y_class, test_size=0.2, random_state=42
    )

    _, _, y_train_reg, y_test_reg = train_test_split(
        X, y_reg, test_size=0.2, random_state=42
    )

    clf_model.fit(X_train, y_train_class)
    reg_model.fit(X_train, y_train_reg)

    # Residual standard deviation for CI
    train_preds = reg_model.predict(X_train)
    residuals = y_train_reg - train_preds
    residual_std = np.std(residuals)

    return clf_model, reg_model, residual_std

clf_model, reg_model, residual_std = train_models(df)

# ----------------------------------------------------------
# Risk Registry
# ----------------------------------------------------------

if "risk_registry" not in st.session_state:
    st.session_state.risk_registry = pd.DataFrame(
        columns=[
            "MRN",
            "Risk Level",
            "Order DateTime",
            "Delay Probability",
            "Projected Minutes",
            "Elapsed Minutes"
        ]
    )

# ----------------------------------------------------------
# Severity Logic
# ----------------------------------------------------------

def assign_severity(prob):
    if prob >= 0.85:
        return "Critical", "🔴"
    elif prob >= 0.70:
        return "High", "🟠"
    elif prob >= 0.40:
        return "Moderate", "🟡"
    else:
        return "Low", "🟢"

# ----------------------------------------------------------
# AI Advisory Generator
# ----------------------------------------------------------

def generate_advisory(prob, projected_minutes, patient_data):

    severity, _ = assign_severity(prob)

    prompt = f"""
You are advising a HOSPITAL DISCHARGE TEAM.

This is operational discharge guidance only.

Patient Profile:
- Age: {patient_data['Patient Age']}
- Length of Stay: {patient_data['Length of Stay (days)']}
- Diagnosis: {patient_data['Primary Diagnosis (Description)']}
- Delay Probability: {round(prob,2)}
- Projected Discharge Duration: {int(projected_minutes)} minutes
- Risk Level: {severity}

Use EXACTLY these section headers:

Operational Risks
Clinical Coordination Actions
Discharge Process Actions
Escalation Plan

Under each header:
- Use bullet points
- Keep bullets short and actionable
- No JSON
- No quotation marks
- No paragraphs
"""

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system",
             "content": "You generate operational discharge playbooks in bullet format."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.2
    )

    return response.choices[0].message.content.strip()

# ----------------------------------------------------------
# Patient Input Form
# ----------------------------------------------------------

st.markdown("## Patient Input")

with st.form("patient_form", clear_on_submit=True):

    mrn = st.text_input("Patient MRN")

    col1, col2 = st.columns(2)

    with col1:
        los = st.number_input("Length of Stay (days)", 0, 100, 5)
        doctors = st.slider("Number of Doctors Involved", 1, 30, 2)

    with col2:
        bill = st.number_input("Current Bill (PHP)", 0, 2000000, 50000)
        age = st.slider("Patient Age", 0, 120, 40)

    diagnosis_input = st.text_input("Enter Diagnosis")

    st.markdown("### Discharge Order Timing")

    col_date, col_time = st.columns(2)

    with col_date:
        order_date = st.date_input("Discharge Order Date")

    with col_time:
        order_time = st.time_input("Discharge Order Time",
                                   step=timedelta(minutes=1))

    submitted = st.form_submit_button("Generate Advisory")

# ----------------------------------------------------------
# Generate Prediction + Advisory
# ----------------------------------------------------------

if submitted:

    if not mrn:
        st.error("Please enter MRN.")
    else:

        baseline_row = {}

        for col in df.select_dtypes(include=["int64", "float64"]).columns:
            if col not in ["delayed_over_200mins",
                           "Discharge Duration (minutes)"]:
                baseline_row[col] = df[col].median()

        for col in df.select_dtypes(include=["object"]).columns:
            baseline_row[col] = df[col].mode()[0]

        baseline_row["Length of Stay (days)"] = los
        baseline_row["Number of Doctors Involved"] = doctors
        baseline_row["Current Bill (PHP)"] = bill
        baseline_row["Patient Age"] = age
        baseline_row["Primary Diagnosis (Description)"] = diagnosis_input

        input_df = pd.DataFrame([baseline_row])

        # Classification
        prob = clf_model.predict_proba(input_df)[0][1]

        # Regression
        projected_minutes = reg_model.predict(input_df)[0]

        # Confidence interval
        ci_lower = projected_minutes - (1.96 * residual_std)
        ci_upper = projected_minutes + (1.96 * residual_std)

        severity, badge = assign_severity(prob)

        st.subheader("Prediction")
        st.progress(float(prob))
        st.markdown(f"Delay Probability: {round(prob,2)}")
        st.markdown(f"Projected Duration: {int(projected_minutes)} minutes")
        st.markdown(
            f"Expected Range: {int(ci_lower)}–{int(ci_upper)} minutes (95% CI)"
        )

        if severity == "Critical":
            st.error("🔴 CRITICAL – Early Discharge Team Intervention Recommended")
        elif severity == "High":
            st.warning("🟠 HIGH – Active Coordination Required")
        elif severity == "Moderate":
            st.info("🟡 MODERATE – Close Monitoring Recommended")
        else:
            st.success("🟢 LOW – Standard Discharge Workflow")

        # Update Board
        order_datetime = datetime.combine(order_date, order_time)
        elapsed = int((datetime.now() - order_datetime).total_seconds() / 60)

        new_row = pd.DataFrame([{
            "MRN": mrn,
            "Risk Level": severity,
            "Order DateTime": order_datetime,
            "Delay Probability": round(prob, 3),
            "Projected Minutes": int(projected_minutes),
            "Elapsed Minutes": elapsed
        }])

        st.session_state.risk_registry = (
            st.session_state.risk_registry[
                st.session_state.risk_registry["MRN"] != mrn
            ]
        )

        st.session_state.risk_registry = pd.concat(
            [st.session_state.risk_registry, new_row],
            ignore_index=True
        )

        advisory = generate_advisory(prob, projected_minutes, baseline_row)

        st.markdown("## Discharge Team Operational Advisory")
        st.markdown(advisory)

# ----------------------------------------------------------
# Display Risk Command Board
# ----------------------------------------------------------

if not st.session_state.risk_registry.empty:

    display_df = st.session_state.risk_registry.copy()

    display_df = display_df.sort_values(
        by="Delay Probability",
        ascending=False
    ).reset_index(drop=True)

    def highlight_elapsed(val):
        if val > 90:
            return "background-color: #ff4d4d; color: white;"
        elif 61 <= val <= 90:
            return "background-color: #fff176;"
        else:
            return ""

    styled_df = display_df.style.applymap(
        highlight_elapsed,
        subset=["Elapsed Minutes"]
    )

    st.markdown("## Discharge Risk Command Board")
    st.dataframe(styled_df, use_container_width=True, hide_index=True)