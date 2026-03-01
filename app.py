# ==========================================================
# DischargeFlow AI – Upgraded Version
# Upgrades: 1B (Elapsed Training), 2 (Metrics + AI Explain),
# 3 (Persistence + Completion Capture)
# ==========================================================

import streamlit as st
import pandas as pd
import sqlite3
from datetime import datetime
from zoneinfo import ZoneInfo
from openai import OpenAI

from sklearn.model_selection import GroupShuffleSplit
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, r2_score

# ----------------------------------------------------------
# CONFIG
# ----------------------------------------------------------

st.set_page_config(page_title="DischargeFlow AI", layout="wide")
st.title("DischargeFlow AI – Learning System Version")

PH_TZ = ZoneInfo("Asia/Manila")

client = None
if "OPENAI_API_KEY" in st.secrets:
    client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

# ----------------------------------------------------------
# DATABASE SETUP (Upgrade 3)
# ----------------------------------------------------------

conn = sqlite3.connect("dischargeflow.db", check_same_thread=False)
c = conn.cursor()

c.execute("""
CREATE TABLE IF NOT EXISTS discharge_records (
    MRN TEXT,
    OrderDateTime TEXT,
    CompletionDateTime TEXT,
    FinalDuration REAL,
    LastPrediction REAL,
    Error REAL
)
""")
conn.commit()

# ----------------------------------------------------------
# LOAD & EXPAND DATA (Upgrade 1B)
# ----------------------------------------------------------

@st.cache_data
def load_and_expand():

    df = pd.read_csv("fictitious_dataset_FINAL.csv")

    expanded_rows = []

    for idx, row in df.iterrows():

        final_duration = row["Discharge Duration (minutes)"]

        for elapsed in range(0, int(final_duration), 60):

            remaining = final_duration - elapsed

            new_row = row.copy()
            new_row["Elapsed Minutes"] = elapsed
            new_row["Remaining Minutes"] = remaining
            new_row["CaseID"] = idx

            expanded_rows.append(new_row)

    expanded_df = pd.DataFrame(expanded_rows)

    return expanded_df

df = load_and_expand()

# ----------------------------------------------------------
# TRAIN MODEL (Group Split)
# ----------------------------------------------------------

@st.cache_resource
def train_model(df):

    target = "Remaining Minutes"

    X = df.drop(columns=[
        "Discharge Duration (minutes)",
        "Remaining Minutes"
    ])

    y = df[target]
    groups = df["CaseID"]

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

    splitter = GroupShuffleSplit(test_size=0.2, random_state=42)

    train_idx, test_idx = next(splitter.split(X, y, groups))

    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    return model, X.columns.tolist(), mae, r2

reg_model, feature_columns, mae, r2 = train_model(df)

# ----------------------------------------------------------
# DISPLAY MODEL PERFORMANCE (Upgrade 2)
# ----------------------------------------------------------

st.markdown("## Model Performance")

with st.expander("View Model Performance Details"):
    
    st.write(f"**Mean Absolute Error:** {round(mae,2)} minutes")
    st.write(f"**R² Score:** {round(r2,3)}")

    if client:
        explanation_prompt = f"""
Explain to a hospital discharge operations team what these results mean:

Mean Absolute Error: {round(mae,2)} minutes
R² Score: {round(r2,3)}

Explain:
- What MAE means in operational terms
- What R² means in simple language
- Whether this is strong, moderate, or weak performance
Keep it concise.
"""

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "Hospital operations ML explainer."},
            {"role": "user", "content": explanation_prompt}
        ],
        temperature=0.2
    )

    st.markdown("### AI Interpretation for Discharge Team")
    st.markdown(response.choices[0].message.content)

# ----------------------------------------------------------
# SESSION STATE
# ----------------------------------------------------------
required_cols = [
    "MRN",
    "OrderDateTime",
    "Baseline",
    "LastPrediction"
]

if "risk_registry" not in st.session_state:
    st.session_state.risk_registry = pd.DataFrame(columns=required_cols)

if not set(required_cols).issubset(
        set(st.session_state.risk_registry.columns)):
    st.session_state.risk_registry = pd.DataFrame(columns=required_cols)

# ----------------------------------------------------------
# PATIENT INPUT
# ----------------------------------------------------------

st.markdown("## New Patient Forecast")

with st.form("patient_form"):

    mrn = st.text_input("MRN")
    los = st.number_input("Length of Stay", 0, 100, 5)
    doctors = st.slider("Doctors", 1, 30, 2)
    bill = st.number_input("Current Bill", 0, 2000000, 50000)
    age = st.slider("Age", 0, 120, 40)
    diagnosis = st.text_input("Diagnosis")

    submit = st.form_submit_button("Generate Forecast")

if submit and mrn:

    baseline = df.iloc[0].copy()

    baseline["Length of Stay (days)"] = los
    baseline["Number of Doctors Involved"] = doctors
    baseline["Current Bill (PHP)"] = bill
    baseline["Patient Age"] = age
    baseline["Primary Diagnosis (Description)"] = diagnosis
    baseline["Elapsed Minutes"] = 0

    remaining = reg_model.predict(pd.DataFrame([baseline]))[0]

    order_dt = datetime.now(PH_TZ)

    st.session_state.risk_registry = pd.concat([
        st.session_state.risk_registry,
        pd.DataFrame([{
            "MRN": mrn,
            "OrderDateTime": order_dt,
            "Baseline": baseline,
            "LastPrediction": remaining
        }])
    ], ignore_index=True)

# ----------------------------------------------------------
# COMMAND BOARD + DISCHARGE BUTTON (Upgrade 3)
# ----------------------------------------------------------

st.markdown("## Command Board")

now = datetime.now(PH_TZ)

for idx, row in st.session_state.risk_registry.iterrows():

    # --- SAFETY GUARD ---
    if "Baseline" not in row or pd.isna(row["Baseline"]) is None:
        continue
    snapshot = row["Baseline"].copy()
    elapsed = int((now - row["OrderDateTime"]).total_seconds() / 60)
    elapsed = max(elapsed, 0)

    snapshot["Elapsed Minutes"] = elapsed

    updated_remaining = reg_model.predict(pd.DataFrame([snapshot]))[0]

    st.write(f"MRN: {row['MRN']}")
    st.write(f"Elapsed: {elapsed} mins")
    st.write(f"Updated Remaining: {int(updated_remaining)} mins")

    if st.button(f"Mark Discharged - {row['MRN']}"):

        completion_dt = datetime.now(PH_TZ)
        final_duration = (completion_dt - row["OrderDateTime"]).total_seconds() / 60

        error = final_duration - row["LastPrediction"]

        c.execute("""
        INSERT INTO discharge_records
        VALUES (?, ?, ?, ?, ?, ?)
        """, (
            row["MRN"],
            row["OrderDateTime"].isoformat(),
            completion_dt.isoformat(),
            final_duration,
            row["LastPrediction"],
            error
        ))

        conn.commit()

        st.success(f"Stored. Error: {round(error,2)} minutes")
