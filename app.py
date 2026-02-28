# ==========================================================
# DischargeFlow AI
# Stable Dynamic Projection Architecture
# Classification + Regression + Acceleration Risk
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
import time

# ----------------------------------------------------------
# Page Config
# ----------------------------------------------------------

st.set_page_config(page_title="DischargeFlow AI", layout="wide")
st.title("DischargeFlow AI")

st.markdown("""
### System Highlights
- Real-time discharge delay risk prediction  
- Projected discharge duration in minutes  
- Dynamic elapsed-time adjustment  
- Automatic escalation monitoring  
- Acceleration risk detection  
- Live command board updates  
""")

# ----------------------------------------------------------
# Auto Refresh (30 seconds)
# ----------------------------------------------------------

if "last_refresh" not in st.session_state:
    st.session_state.last_refresh = time.time()

if time.time() - st.session_state.last_refresh > 30:
    st.session_state.last_refresh = time.time()
    st.rerun()

# ----------------------------------------------------------
# OpenAI Setup
# ----------------------------------------------------------

if "OPENAI_API_KEY" not in st.secrets:
    st.error("OpenAI API key not configured.")
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

    df["Elapsed Minutes"] = (
        df["Discharge Duration (minutes)"] *
        np.random.uniform(0.3, 0.9, size=len(df))
    ).astype(int)

    return df

df = load_data()

# ----------------------------------------------------------
# Train Models
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

    X_train, _, y_train_class, _ = train_test_split(
        X, y_class, test_size=0.2, random_state=42
    )

    _, _, y_train_reg, _ = train_test_split(
        X, y_reg, test_size=0.2, random_state=42
    )

    clf_model.fit(X_train, y_train_class)
    reg_model.fit(X_train, y_train_reg)

    return clf_model, reg_model

clf_model, reg_model = train_models(df)

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
            "Original Projected Minutes",
            "Updated Projected Minutes",
            "Elapsed Minutes",
            "Acceleration Risk",
            "Feature Snapshot"
        ]
    )

# ----------------------------------------------------------
# Severity Logic
# ----------------------------------------------------------

def assign_severity(prob):
    if prob >= 0.85:
        return "Critical"
    elif prob >= 0.70:
        return "High"
    elif prob >= 0.40:
        return "Moderate"
    else:
        return "Low"

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
        order_time = st.time_input("Discharge Order Time",
                                   step=timedelta(minutes=1))

    submitted = st.form_submit_button("Generate Advisory")

# ----------------------------------------------------------
# Generate Prediction
# ----------------------------------------------------------

if submitted:

    if not mrn:
        st.error("Please enter MRN.")
    else:

        order_datetime = datetime.combine(order_date, order_time)
        elapsed = int((datetime.now() - order_datetime).total_seconds() / 60)

        # Build feature snapshot from training structure
        feature_snapshot = df.drop(
            columns=["delayed_over_200mins",
                     "Discharge Duration (minutes)"]
        ).iloc[0].copy()

        feature_snapshot["Length of Stay (days)"] = los
        feature_snapshot["Number of Doctors Involved"] = doctors
        feature_snapshot["Current Bill (PHP)"] = bill
        feature_snapshot["Patient Age"] = age
        feature_snapshot["Primary Diagnosis (Description)"] = diagnosis_input
        feature_snapshot["Elapsed Minutes"] = elapsed

        input_df = pd.DataFrame([feature_snapshot])

        prob = clf_model.predict_proba(input_df)[0][1]
        projected_minutes = reg_model.predict(input_df)[0]

        severity = assign_severity(prob)

        new_row = pd.DataFrame([{
            "MRN": mrn,
            "Risk Level": severity,
            "Order DateTime": order_datetime,
            "Delay Probability": round(prob, 3),
            "Original Projected Minutes": int(projected_minutes),
            "Updated Projected Minutes": int(projected_minutes),
            "Elapsed Minutes": elapsed,
            "Acceleration Risk": "NO",
            "Feature Snapshot": feature_snapshot.to_dict()
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

# ----------------------------------------------------------
# Command Board
# ----------------------------------------------------------

if not st.session_state.risk_registry.empty:

    display_df = st.session_state.risk_registry.copy()

    updated_proj = []
    updated_elapsed = []
    acceleration_flags = []

    for idx, row in display_df.iterrows():

        elapsed_now = int(
            (datetime.now() - row["Order DateTime"]).total_seconds() / 60
        )

        snapshot = row["Feature Snapshot"]
        snapshot["Elapsed Minutes"] = elapsed_now

        temp_df = pd.DataFrame([snapshot])
        new_projection = reg_model.predict(temp_df)[0]

        updated_proj.append(int(new_projection))
        updated_elapsed.append(elapsed_now)

        if elapsed_now >= 0.8 * new_projection:
            acceleration_flags.append("YES")
        else:
            acceleration_flags.append("NO")

    display_df["Elapsed Minutes"] = updated_elapsed
    display_df["Updated Projected Minutes"] = updated_proj
    display_df["Acceleration Risk"] = acceleration_flags

    display_df = display_df.sort_values(
        by="Delay Probability",
        ascending=False
    ).reset_index(drop=True)

    def highlight_accel(val):
        if val == "YES":
            return "background-color: #ff6b6b; color: white;"
        return ""

    styled_df = display_df.style.applymap(
        highlight_accel,
        subset=["Acceleration Risk"]
    )

    st.markdown("## Discharge Risk Command Board")
    st.dataframe(
        styled_df.drop(columns=["Feature Snapshot"]),
        use_container_width=True,
        hide_index=True
    )