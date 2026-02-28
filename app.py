# ==========================================================
# DischargeFlow AI
# Live Discharge Risk Command Center
# Deployment-Ready Version (Streamlit Cloud Safe)
# ==========================================================

import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from openai import OpenAI
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
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
    df["delayed_over_200mins"] = (
        df["Discharge Duration (minutes)"] > 200
    ).astype(int)
    return df

df = load_data()

# ----------------------------------------------------------
# Train Model (Retrospective – All Columns)
# ----------------------------------------------------------

@st.cache_resource
def train_model(df):

    target = "delayed_over_200mins"

    X = df.drop(columns=[target, "Discharge Duration (minutes)"])
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
        ("clf", GradientBoostingClassifier(random_state=42))
    ])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model.fit(X_train, y_train)

    return model

model = train_model(df)
clf = model.named_steps["clf"]

# ----------------------------------------------------------
# Risk Registry
# ----------------------------------------------------------

if "risk_registry" not in st.session_state:
    st.session_state.risk_registry = pd.DataFrame(
        columns=["MRN", "Risk Level", "Order DateTime",
                 "Delay Probability", "Elapsed Minutes"]
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

def generate_advisory(prob, patient_data):

    severity, _ = assign_severity(prob)

    prompt = f"""
You are advising a HOSPITAL DISCHARGE TEAM.

Their goal is to ensure this patient is discharged ON TIME and prevent operational delays.

This is NOT clinical treatment advice.
This is operational discharge guidance only.

Patient Profile:
- Age: {patient_data['Patient Age']}
- Length of Stay: {patient_data['Length of Stay (days)']}
- Diagnosis: {patient_data['Primary Diagnosis (Description)']}
- Delay Probability: {round(prob,2)}
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

        doctors = st.slider(
            "Number of Doctors Involved",
            min_value=1,
            max_value=30,
            value=2
        )

    with col2:
        bill = st.number_input("Current Bill (PHP)", 0, 2000000, 50000)

        age = st.slider(
            "Patient Age",
            min_value=0,
            max_value=120,
            value=40
        )

    diagnosis_input = st.text_input("Enter Diagnosis")

    st.markdown("### Discharge Order Timing")

    col_date, col_time = st.columns(2)

    with col_date:
        order_date = st.date_input("Discharge Order Date")

    with col_time:
        order_time = st.time_input(
            "Discharge Order Time",
            step=timedelta(minutes=1)
        )

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

        input_transformed = model.named_steps["pre"].transform(input_df)
        input_transformed = np.array(input_transformed).astype(float)

        prob = clf.predict_proba(input_transformed)[0][1]
        severity, badge = assign_severity(prob)

        st.subheader("Prediction")
        st.progress(float(prob))
        st.markdown(f"Delay Probability: {round(prob,2)}")

        if severity == "Critical":
            st.error("🔴 CRITICAL – Early Discharge Team Intervention Recommended")
        elif severity == "High":
            st.warning("🟠 HIGH – Active Coordination Required")
        elif severity == "Moderate":
            st.info("🟡 MODERATE – Close Monitoring Recommended")
        else:
            st.success("🟢 LOW – Standard Discharge Workflow")

        # Add to board
        order_datetime = datetime.combine(order_date, order_time)
        elapsed = int((datetime.now() - order_datetime).total_seconds() / 60)

        new_row = pd.DataFrame([{
            "MRN": mrn,
            "Risk Level": severity,
            "Order DateTime": order_datetime,
            "Delay Probability": round(prob, 3),
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

        # Generate AI Advisory
        advisory = generate_advisory(prob, baseline_row)

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

    # Conditional coloring
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