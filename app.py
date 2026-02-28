# ==========================================================
# DischargeFlow AI
# Regression-Only Stable Version
# Philippine Timezone Safe
# ==========================================================

import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo
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

st.markdown("""
### 🚀 Version Highlights
- Regression-based projected discharge duration
- Live Philippine timezone tracking (Asia/Manila)
- Real-time elapsed time recalculation
- Updated projected minutes auto-adjusted
- Operational discharge team advisory
""")

st.caption("Live discharge duration forecasting and operational command board.")

# ----------------------------------------------------------
# Philippine Timezone
# ----------------------------------------------------------

PH_TZ = ZoneInfo("Asia/Manila")

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
# Train Regression Model
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
# Advisory Generator
# ----------------------------------------------------------

def generate_advisory(projected_minutes, patient_data):

    prompt = f"""
You are advising a hospital discharge operations team.

This is operational discharge guidance only.
NOT clinical treatment advice.

Patient:
- Age: {patient_data['Patient Age']}
- Length of Stay: {patient_data['Length of Stay (days)']}
- Diagnosis: {patient_data['Primary Diagnosis (Description)']}
- Projected Discharge Duration (minutes): {projected_minutes}

Use EXACT section headers:

Operational Risks
Clinical Coordination Actions
Discharge Process Actions
Escalation Plan

Use bullet points.
No JSON.
No quotation marks.
Short actionable bullets only.
"""

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system",
             "content": "You generate operational discharge bullet-point playbooks."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.2
    )

    return response.choices[0].message.content.strip()

# ----------------------------------------------------------
# Risk Registry (Session State)
# ----------------------------------------------------------

if "risk_registry" not in st.session_state:
    st.session_state.risk_registry = pd.DataFrame(
        columns=[
            "MRN",
            "Order DateTime",
            "Projected Minutes"
        ]
    )

# ----------------------------------------------------------
# Patient Input
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
        now_ph = datetime.now(PH_TZ)
        order_time = st.time_input(
            "Discharge Order Time",
            value=now_ph.time(),
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

        # Build baseline row
        baseline_row = {}

        for col in df.columns:
            if col != "Discharge Duration (minutes)":
                if df[col].dtype in ["int64", "float64"]:
                    baseline_row[col] = df[col].median()
                else:
                    baseline_row[col] = df[col].mode()[0]

        baseline_row["Length of Stay (days)"] = los
        baseline_row["Number of Doctors Involved"] = doctors
        baseline_row["Current Bill (PHP)"] = bill
        baseline_row["Patient Age"] = age
        baseline_row["Primary Diagnosis (Description)"] = diagnosis_input

        input_df = pd.DataFrame([baseline_row])

        projected_minutes = int(reg_model.predict(input_df)[0])

        # Save to board
        order_datetime = datetime.combine(order_date, order_time)
        order_datetime = order_datetime.replace(tzinfo=PH_TZ)

        new_row = pd.DataFrame([{
            "MRN": mrn,
            "Order DateTime": order_datetime,
            "Projected Minutes": projected_minutes
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

        # Display Forecast
        st.subheader("Forecast")
        st.markdown(f"Projected Duration: {projected_minutes} minutes")

        advisory = generate_advisory(projected_minutes, baseline_row)

        st.markdown("## Discharge Team Operational Advisory")
        st.markdown(advisory)

# ----------------------------------------------------------
# Display Command Board
# ----------------------------------------------------------

if not st.session_state.risk_registry.empty:

    board = st.session_state.risk_registry.copy()

    now_ph = datetime.now(PH_TZ)

    board["Elapsed Minutes"] = (
        (now_ph - board["Order DateTime"])
        .dt.total_seconds() / 60
    ).astype(int)

    board["Elapsed Minutes"] = board["Elapsed Minutes"].clip(lower=0)

    board["Updated Projected Minutes"] = (
        board["Projected Minutes"] - board["Elapsed Minutes"]
    ).clip(lower=0).astype(int)

    board_display = board.copy()
    board_display["Order DateTime"] = board_display["Order DateTime"].dt.strftime("%Y-%m-%d %H:%M")

    st.markdown("## Discharge Risk Command Board")
    st.dataframe(
        board_display.sort_values("Updated Projected Minutes", ascending=False),
        use_container_width=True,
        hide_index=True
    )