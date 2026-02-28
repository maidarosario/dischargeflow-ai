# ==========================================================
# DischargeFlow AI
# Regression-Based Discharge Intelligence System
# ==========================================================

import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from openai import OpenAI
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer

# ----------------------------------------------------------
# Page Config
# ----------------------------------------------------------

st.set_page_config(page_title="DischargeFlow AI", layout="wide")
st.title("DischargeFlow AI")
st.caption("Live discharge duration forecasting and operational command board.")

# ----------------------------------------------------------
# OpenAI (Streamlit Cloud Safe)
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
    return pd.read_csv("fictitious_dataset_FINAL.csv")

df = load_data()

# ----------------------------------------------------------
# Train REGRESSION Model (Single Model Only)
# ----------------------------------------------------------

@st.cache_resource
def train_regression(df):

    target = "Discharge Duration (minutes)"

    X = df.drop(columns=[target])
    y = df[target]

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

    model = Pipeline([
        ("pre", preprocessor),
        ("reg", GradientBoostingRegressor(random_state=42))
    ])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model.fit(X_train, y_train)

    return model, numeric_features, categorical_features

reg_model, numeric_features, categorical_features = train_regression(df)

# ----------------------------------------------------------
# Risk Logic (Derived from Minutes)
# ----------------------------------------------------------

def assign_severity(minutes):
    if minutes >= 300:
        return "Critical", "🔴"
    elif minutes >= 200:
        return "High", "🟠"
    elif minutes >= 150:
        return "Moderate", "🟡"
    else:
        return "Low", "🟢"

# ----------------------------------------------------------
# AI Advisory (Restored – Bullet Format)
# ----------------------------------------------------------

def generate_advisory(projected_minutes, patient_data):

    severity, _ = assign_severity(projected_minutes)

    prompt = f"""
You are advising a HOSPITAL DISCHARGE TEAM.

Goal: prevent discharge delays and reduce projected duration.

This is operational guidance only.

Patient Profile:
Age: {patient_data['Patient Age']}
Length of Stay: {patient_data['Length of Stay (days)']}
Diagnosis: {patient_data['Primary Diagnosis (Description)']}
Projected Discharge Duration: {projected_minutes} minutes
Risk Level: {severity}

Use EXACTLY these headers:

Operational Risks
Clinical Coordination Actions
Discharge Process Actions
Escalation Plan

Under each header:
- Use short bullet points
- No JSON
- No quotation marks
- No paragraphs
"""

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system",
             "content": "You generate structured discharge operational playbooks."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.2
    )

    return response.choices[0].message.content.strip()

# ----------------------------------------------------------
# Risk Registry (Command Board Storage)
# ----------------------------------------------------------

if "risk_registry" not in st.session_state:
    st.session_state.risk_registry = pd.DataFrame(
        columns=[
            "MRN",
            "Risk Level",
            "Order DateTime",
            "Projected Minutes",
            "Elapsed Minutes",
            "Updated Projected Minutes"
        ]
    )

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

    col_date, col_time = st.columns(2)

    with col_date:
        order_date = st.date_input("Discharge Order Date")

    with col_time:
        order_time = st.time_input(
            "Discharge Order Time",
            step=timedelta(minutes=1)
        )

    submitted = st.form_submit_button("Generate Forecast")

# ----------------------------------------------------------
# Generate Prediction
# ----------------------------------------------------------

if submitted:

    if not mrn:
        st.error("Please enter MRN.")
    else:

        baseline_row = {}

        for col in numeric_features:
            baseline_row[col] = df[col].median()

        for col in categorical_features:
            baseline_row[col] = df[col].mode()[0]

        baseline_row["Length of Stay (days)"] = los
        baseline_row["Number of Doctors Involved"] = doctors
        baseline_row["Current Bill (PHP)"] = bill
        baseline_row["Patient Age"] = age
        baseline_row["Primary Diagnosis (Description)"] = diagnosis_input

        input_df = pd.DataFrame([baseline_row])

        projected_minutes = reg_model.predict(input_df)[0]
        projected_minutes = int(round(projected_minutes))

        severity, badge = assign_severity(projected_minutes)

        # ------------------------------
        # Display Forecast
        # ------------------------------

        st.subheader("Projection")

        st.markdown(f"Projected Duration: {projected_minutes} minutes")

        if severity == "Critical":
            st.error("🔴 CRITICAL – Immediate Discharge Team Escalation Required")
        elif severity == "High":
            st.warning("🟠 HIGH – Active Coordination Required")
        elif severity == "Moderate":
            st.info("🟡 MODERATE – Close Monitoring")
        else:
            st.success("🟢 LOW – Standard Workflow")

        # ------------------------------
        # Add to Command Board
        # ------------------------------

        order_datetime = datetime.combine(order_date, order_time)
        elapsed = int((datetime.now() - order_datetime).total_seconds() / 60)

        updated_projection = max(projected_minutes - elapsed, 0)

        new_row = pd.DataFrame([{
            "MRN": mrn,
            "Risk Level": severity,
            "Order DateTime": order_datetime,
            "Projected Minutes": projected_minutes,
            "Elapsed Minutes": elapsed,
            "Updated Projected Minutes": updated_projection
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

        # ------------------------------
        # AI Advisory Restored
        # ------------------------------

        advisory = generate_advisory(projected_minutes, baseline_row)

        st.markdown("## Discharge Team Operational Advisory")
        st.markdown(advisory)

# ----------------------------------------------------------
# Display Command Board
# ----------------------------------------------------------

if not st.session_state.risk_registry.empty:

    board = st.session_state.risk_registry.copy()

    board["Projected Minutes"] = pd.to_numeric(
        board["Projected Minutes"], errors="coerce"
    ).fillna(0).astype(int)

    board["Updated Projected Minutes"] = pd.to_numeric(
        board["Updated Projected Minutes"], errors="coerce"
    ).fillna(0).astype(int)

    board = board.sort_values(
        by="Projected Minutes",
        ascending=False
    ).reset_index(drop=True)

    st.markdown("## Discharge Risk Command Board")
    st.dataframe(board, use_container_width=True, hide_index=True)