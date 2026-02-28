# ==========================================================
# DischargeFlow AI
# Regression-Based Live Discharge Forecast Engine
# Dynamic Reforecast Version
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
st.caption("Live regression-based discharge duration forecasting system.")

# ----------------------------------------------------------
# OpenAI Configuration
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
    return df

df = load_data()

# ----------------------------------------------------------
# Train Regression Model (Includes Elapsed Feature)
# ----------------------------------------------------------

@st.cache_resource
def train_regression_model(df):

    target = "Discharge Duration (minutes)"

    # Add synthetic elapsed feature for training stability
    df["Elapsed Minutes"] = 0

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

reg_model, numeric_features, categorical_features = train_regression_model(df)

# ----------------------------------------------------------
# Session State Board
# ----------------------------------------------------------

if "risk_registry" not in st.session_state:
    st.session_state.risk_registry = pd.DataFrame(
        columns=[
            "MRN",
            "Order DateTime",
            "Projected Minutes",
            "Feature Snapshot"
        ]
    )

# ----------------------------------------------------------
# AI Advisory Generator
# ----------------------------------------------------------

def generate_advisory(projected_minutes, patient_data):

    prompt = f"""
You are advising a hospital discharge team.

This is operational guidance only.

Patient:
Age: {patient_data['Patient Age']}
Length of Stay: {patient_data['Length of Stay (days)']}
Diagnosis: {patient_data['Primary Diagnosis (Description)']}
Projected Total Discharge Duration: {int(projected_minutes)} minutes

Use EXACTLY these headers:

Operational Risks
Clinical Coordination Actions
Discharge Process Actions
Escalation Plan

Under each header:
- Use bullet points
- Short actionable lines
- No JSON
- No quotation marks
- No parentheses
"""

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system",
             "content": "You generate structured operational discharge playbooks."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.2
    )

    return response.choices[0].message.content.strip()

# ----------------------------------------------------------
# Patient Input Form
# ----------------------------------------------------------

st.markdown("## Patient Input")

with st.form("patient_form"):

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
# Generate Forecast
# ----------------------------------------------------------

if submitted:

    if not mrn:
        st.error("Please enter MRN.")
    else:

        baseline = {}

        for col in numeric_features:
            if col == "Elapsed Minutes":
                baseline[col] = 0
            else:
                baseline[col] = df[col].median()

        for col in categorical_features:
            baseline[col] = df[col].mode()[0]

        baseline["Length of Stay (days)"] = los
        baseline["Number of Doctors Involved"] = doctors
        baseline["Current Bill (PHP)"] = bill
        baseline["Patient Age"] = age
        baseline["Primary Diagnosis (Description)"] = diagnosis_input
        baseline["Elapsed Minutes"] = 0

        input_df = pd.DataFrame([baseline])

        projected_minutes = reg_model.predict(input_df)[0]
        projected_minutes = int(round(projected_minutes))

        order_datetime = datetime.combine(order_date, order_time)

        st.subheader("Prediction")
        st.markdown(f"Projected Duration: {projected_minutes} minutes")

        # Store snapshot
        new_row = pd.DataFrame([{
            "MRN": mrn,
            "Order DateTime": order_datetime,
            "Projected Minutes": projected_minutes,
            "Feature Snapshot": baseline
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

        advisory = generate_advisory(projected_minutes, baseline)

        st.markdown("## Discharge Team Operational Advisory")
        st.markdown(advisory)

## ----------------------------------------------------------
# Live Dynamic Reforecast Board (No Snapshot Storage)
# ----------------------------------------------------------

if not st.session_state.risk_registry.empty:

    board = st.session_state.risk_registry.copy()
    now = datetime.now()

    updated_rows = []

    for _, row in board.iterrows():

        elapsed = int((now - row["Order DateTime"]).total_seconds() / 60)
        elapsed = max(elapsed, 0)

        # Rebuild feature row using stored projected baseline logic
        baseline = {}

        for col in numeric_features:
            if col == "Elapsed Minutes":
                baseline[col] = elapsed
            else:
                baseline[col] = df[col].median()

        for col in categorical_features:
            baseline[col] = df[col].mode()[0]

        temp_df = pd.DataFrame([baseline])

        new_projection = reg_model.predict(temp_df)[0]
        new_projection = int(round(new_projection))

        updated_rows.append({
            "MRN": row["MRN"],
            "Order DateTime": row["Order DateTime"],
            "Projected Minutes": row["Projected Minutes"],
            "Elapsed Minutes": elapsed,
            "Updated Projected Minutes": new_projection
        })

    updated_df = pd.DataFrame(updated_rows)

    st.markdown("## Discharge Risk Command Board")
    st.dataframe(
        updated_df.sort_values(by="Updated Projected Minutes", ascending=False),
        use_container_width=True,
        hide_index=True
    )