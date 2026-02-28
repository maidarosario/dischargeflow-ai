# ==========================================================
# DischargeFlow AI
# Option A – Regression-Based Dynamic Forecast
# Philippine Timezone Safe Version
# ==========================================================

import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
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
### Version Highlights
• Regression-based projected discharge duration  
• Dynamic re-forecast using elapsed time  
• Philippine timezone aligned  
• Operational discharge team advisory  
• Live risk command board  
""")

# ----------------------------------------------------------
# Philippine Time
# ----------------------------------------------------------

ph_tz = ZoneInfo("Asia/Manila")
now_ph = datetime.now(ph_tz)

# ----------------------------------------------------------
# OpenAI (Optional Advisory)
# ----------------------------------------------------------

if "OPENAI_API_KEY" in st.secrets:
    client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])
else:
    client = None

# ----------------------------------------------------------
# Load Data
# ----------------------------------------------------------

@st.cache_data
def load_data():
    df = pd.read_csv("fictitious_dataset_FINAL.csv")
    return df

df = load_data()

# ----------------------------------------------------------
# Train Regression Model
# ----------------------------------------------------------

@st.cache_resource
def train_model(df):

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

reg_model = train_model(df)

# ----------------------------------------------------------
# Risk Categorization
# ----------------------------------------------------------

def assign_risk(minutes):
    if minutes >= 240:
        return "High", "🟠"
    elif minutes >= 180:
        return "Moderate", "🟡"
    else:
        return "Low", "🟢"

# ----------------------------------------------------------
# Risk Registry Storage
# ----------------------------------------------------------

if "risk_registry" not in st.session_state:
    st.session_state.risk_registry = pd.DataFrame(
        columns=[
            "MRN",
            "Order DateTime",
            "Baseline Features",
            "Original Projected Minutes"
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

    diagnosis_input = st.text_input("Primary Diagnosis")

    st.markdown("## Discharge Order Timing")

    col_date, col_time = st.columns(2)

    with col_date:
        order_date = st.date_input(
            "Discharge Order Date",
            value=now_ph.date()
        )

    with col_time:
        order_time = st.time_input(
            "Discharge Order Time",
            value=now_ph.time().replace(second=0, microsecond=0)
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

        for col in df.select_dtypes(include=["int64", "float64"]).columns:
            baseline_row[col] = df[col].median()

        for col in df.select_dtypes(include=["object"]).columns:
            baseline_row[col] = df[col].mode()[0]

        baseline_row["Length of Stay (days)"] = los
        baseline_row["Number of Doctors Involved"] = doctors
        baseline_row["Current Bill (PHP)"] = bill
        baseline_row["Patient Age"] = age
        baseline_row["Primary Diagnosis (Description)"] = diagnosis_input

        input_df = pd.DataFrame([baseline_row])

        projected_minutes = reg_model.predict(input_df)[0]
        projected_minutes = int(round(projected_minutes))

        risk_level, badge = assign_risk(projected_minutes)

        order_datetime = datetime.combine(order_date, order_time)
        order_datetime = order_datetime.replace(tzinfo=ph_tz)

        st.subheader("Initial Forecast")
        st.markdown(f"Projected Duration: **{projected_minutes} minutes**")
        st.markdown(f"{badge} {risk_level}")

        # Save to registry
        new_row = pd.DataFrame([{
            "MRN": mrn,
            "Order DateTime": order_datetime,
            "Baseline Features": baseline_row,
            "Original Projected Minutes": projected_minutes
        }])

        st.session_state.risk_registry = pd.concat(
            [st.session_state.risk_registry, new_row],
            ignore_index=True
        )

        # Advisory
        if client:
            prompt = f"""
You are advising a hospital discharge team.

Projected Duration: {projected_minutes} minutes
Risk Level: {risk_level}

Provide sections:
Operational Risks
Clinical Coordination Actions
Discharge Process Actions
Escalation Plan

Bullet format only.
"""
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "Hospital discharge operations advisor."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.2
            )

            advisory = response.choices[0].message.content.strip()

            st.markdown("## Discharge Team Operational Advisory")
            st.markdown(advisory)

# ----------------------------------------------------------
# Risk Command Board
# ----------------------------------------------------------

if not st.session_state.risk_registry.empty:

    st.markdown("## Discharge Risk Command Board")

    board = []

    for _, row in st.session_state.risk_registry.iterrows():

        snapshot = row["Baseline Features"]
        order_dt = row["Order DateTime"]

        now_ph = datetime.now(ph_tz)
        elapsed = int((now_ph - order_dt).total_seconds() / 60)
        elapsed = max(elapsed, 0)

        snapshot["Elapsed Minutes"] = elapsed

        temp_df = pd.DataFrame([snapshot])
        new_projection = reg_model.predict(temp_df)[0]
        new_projection = int(round(new_projection))

        risk_level, badge = assign_risk(new_projection)

        board.append({
            "MRN": row["MRN"],
            "Order DateTime": order_dt.strftime("%Y-%m-%d %H:%M"),
            "Elapsed Minutes": elapsed,
            "Original Projected Minutes": row["Original Projected Minutes"],
            "Updated Projected Minutes": new_projection,
            "Risk Level": f"{badge} {risk_level}"
        })

    board_df = pd.DataFrame(board)

    st.dataframe(
        board_df,
        use_container_width=True,
        hide_index=True
    )