# ==========================================================
# DischargeFlow AI
# Live Discharge Risk Command Center
# Regression-Based Duration Forecasting
# Philippines Timezone Enabled
# ==========================================================

import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import pytz
from openai import OpenAI

from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer

# ----------------------------------------------------------
# PAGE CONFIG
# ----------------------------------------------------------

st.set_page_config(page_title="DischargeFlow AI", layout="wide")

st.title("DischargeFlow AI")
st.markdown("""
Live discharge duration forecasting using regression modeling.

**Highlights**
- Regression-based projected discharge duration (minutes)
- Philippine Time (Asia/Manila) dynamic clock
- Live elapsed time recalculation
- Row-specific duration updates (no shared projections)
- AI operational discharge advisory
""")

# ----------------------------------------------------------
# OPENAI (Streamlit Cloud Safe)
# ----------------------------------------------------------

if "OPENAI_API_KEY" not in st.secrets:
    st.error("OpenAI API key not configured in Streamlit Secrets.")
    st.stop()

client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

# ----------------------------------------------------------
# LOAD DATA
# ----------------------------------------------------------

@st.cache_data
def load_data():
    df = pd.read_csv("fictitious_dataset_FINAL.csv")
    return df

df = load_data()

# ----------------------------------------------------------
# TRAIN REGRESSION MODEL
# ----------------------------------------------------------

@st.cache_resource
def train_regression_model(df):

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

    return model

reg_model = train_regression_model(df)

# ----------------------------------------------------------
# AI ADVISORY
# ----------------------------------------------------------

def generate_advisory(projected_minutes, patient_data):

    prompt = f"""
You are advising a hospital discharge operations team.

This is NOT clinical advice.
This is operational discharge workflow guidance.

Patient Profile:
- Age: {patient_data['Patient Age']}
- Length of Stay: {patient_data['Length of Stay (days)']}
- Diagnosis: {patient_data['Primary Diagnosis (Description)']}
- Projected Discharge Duration: {int(projected_minutes)} minutes

Use EXACTLY these section headers:

Operational Risks
Clinical Coordination Actions
Discharge Process Actions
Escalation Plan

Rules:
- Use bullet points
- Keep bullets short
- No JSON
- No quotation marks
- No parentheses
"""

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system",
             "content": "You generate concise hospital discharge playbooks."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.2
    )

    return response.choices[0].message.content.strip()

# ----------------------------------------------------------
# SESSION STATE BOARD
# ----------------------------------------------------------

if "risk_registry" not in st.session_state:
    st.session_state.risk_registry = pd.DataFrame(
        columns=[
            "MRN",
            "Order DateTime",
            "Projected Minutes",
            "Elapsed Minutes",
            "Updated Projected Minutes"
        ]
    )

# ----------------------------------------------------------
# PATIENT INPUT
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

    diagnosis_input = st.text_input("Primary Diagnosis (Description)")

    st.markdown("### Discharge Order Timing")

    col_date, col_time = st.columns(2)

    with col_date:
        order_date = st.date_input("Discharge Order Date")

    with col_time:
        order_time = st.time_input("Discharge Order Time")

    submitted = st.form_submit_button("Generate Forecast")

# ----------------------------------------------------------
# GENERATE FORECAST
# ----------------------------------------------------------

if submitted:

    if not mrn:
        st.error("Please enter MRN.")
    else:

        baseline_row = {}

        for col in df.columns:
            if col != "Discharge Duration (minutes)":
                if df[col].dtype == "object":
                    baseline_row[col] = df[col].mode()[0]
                else:
                    baseline_row[col] = df[col].median()

        baseline_row["Length of Stay (days)"] = los
        baseline_row["Number of Doctors Involved"] = doctors
        baseline_row["Current Bill (PHP)"] = bill
        baseline_row["Patient Age"] = age
        baseline_row["Primary Diagnosis (Description)"] = diagnosis_input

        input_df = pd.DataFrame([baseline_row])

        projected_minutes = reg_model.predict(input_df)[0]
        projected_minutes = int(max(projected_minutes, 0))

        # PH TIMEZONE
        ph_tz = pytz.timezone("Asia/Manila")
        now = datetime.now(ph_tz)

        order_datetime = ph_tz.localize(
            datetime.combine(order_date, order_time)
        )

        elapsed = max(
            int((now - order_datetime).total_seconds() // 60),
            0
        )

        updated_projection = max(projected_minutes - elapsed, 0)

        # Display
        st.subheader("Forecast")
        st.markdown(f"Projected Duration: {projected_minutes} minutes")
        st.markdown(f"Elapsed Minutes: {elapsed}")
        st.markdown(f"Updated Projected Minutes: {updated_projection}")

        # Add to board
        new_row = pd.DataFrame([{
            "MRN": mrn,
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

        # AI Advisory
        advisory = generate_advisory(projected_minutes, baseline_row)

        st.markdown("## Discharge Team Operational Advisory")
        st.markdown(advisory)

# ----------------------------------------------------------
# LIVE COMMAND BOARD
# ----------------------------------------------------------

if not st.session_state.risk_registry.empty:

    ph_tz = pytz.timezone("Asia/Manila")
    now = datetime.now(ph_tz)

    board = st.session_state.risk_registry.copy()

    for idx, row in board.iterrows():

        order_dt = row["Order DateTime"]

        elapsed = max(
            int((now - order_dt).total_seconds() // 60),
            0
        )

        board.at[idx, "Elapsed Minutes"] = elapsed

        original = int(row["Projected Minutes"])

        updated = max(original - elapsed, 0)

        board.at[idx, "Updated Projected Minutes"] = int(updated)

    st.markdown("## Discharge Risk Command Board")

    st.dataframe(
        board.sort_values("Updated Projected Minutes", ascending=False),
        use_container_width=True,
        hide_index=True
    )