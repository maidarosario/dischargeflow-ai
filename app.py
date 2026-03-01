# ==========================================================
# DischargeFlow AI – Stable Learning System Version
# Upgrades 1B + 2 + 3 (Clean UI Restored)
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
st.title("DischargeFlow AI")

PH_TZ = ZoneInfo("Asia/Manila")

client = None
if "OPENAI_API_KEY" in st.secrets:
    client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

# ----------------------------------------------------------
# DATABASE (Upgrade 3)
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

    return pd.DataFrame(expanded_rows)

df = load_and_expand()

# ----------------------------------------------------------
# TRAIN MODEL
# ----------------------------------------------------------

@st.cache_resource
def train_model(df):

    target = "Remaining Minutes"

    X = df.drop(columns=["Discharge Duration (minutes)", "Remaining Minutes"])
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
# MODEL PERFORMANCE (Upgrade 2)
# ----------------------------------------------------------

with st.expander("View Model Performance Details"):

    st.write(f"Mean Absolute Error: {round(mae,2)} minutes")
    st.write(f"R² Score: {round(r2,3)}")

    if client:
        explanation_prompt = f"""
Explain to a hospital discharge operations team:

MAE: {round(mae,2)} minutes
R²: {round(r2,3)}

Explain:
- What MAE means operationally
- What R² means
- Whether performance is strong, moderate, or weak
Keep concise.
"""
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "Hospital ML explainer."},
                {"role": "user", "content": explanation_prompt}
            ],
            temperature=0.2
        )
        st.markdown(response.choices[0].message.content)

# ----------------------------------------------------------
# SESSION STATE
# ----------------------------------------------------------

required_cols = ["MRN", "OrderDateTime", "Baseline", "LastPrediction"]

if "risk_registry" not in st.session_state:
    st.session_state.risk_registry = pd.DataFrame(columns=required_cols)

if not set(required_cols).issubset(st.session_state.risk_registry.columns):
    st.session_state.risk_registry = pd.DataFrame(columns=required_cols)

# ----------------------------------------------------------
# PATIENT INPUT
# ----------------------------------------------------------

st.markdown("## New Patient Forecast")

with st.form("patient_form"):

    mrn = st.text_input("MRN")
    los = st.number_input("Length of Stay (days)", 0, 100, 5)
    doctors = st.slider("Number of Doctors Involved", 1, 30, 2)
    bill = st.number_input("Current Bill (PHP)", 0, 2000000, 50000)
    age = st.slider("Patient Age", 0, 120, 40)
    diagnosis = st.text_input("Primary Diagnosis (Description)")

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

    # -----------------------------
    # INITIAL FORECAST + ADVISORY
    # -----------------------------

    st.subheader("Initial Forecast")
    st.markdown(f"Projected Remaining Time: **{int(remaining)} minutes**")

    if remaining >= 240:
        risk = "High"
    elif remaining >= 180:
        risk = "Moderate"
    else:
        risk = "Low"

    st.markdown(f"Risk Level: **{risk}**")

    if client:
        advisory_prompt = f"""
Projected Remaining Time: {int(remaining)} minutes
Risk Level: {risk}

Provide:
Operational Risks
Clinical Coordination Actions
Discharge Process Actions
Escalation Plan

Bullet format only.
"""
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "Hospital discharge advisor."},
                {"role": "user", "content": advisory_prompt}
            ],
            temperature=0.2
        )

        st.markdown("## Discharge Team Operational Advisory")
        st.markdown(response.choices[0].message.content)

# ----------------------------------------------------------
# COMMAND BOARD (Restored Table)
# ----------------------------------------------------------

st.markdown("## Command Board")

board_rows = []
now = datetime.now(PH_TZ)

# Build structured rows
for idx, row in st.session_state.risk_registry.iterrows():

    if "Baseline" not in row.index or row["Baseline"] is None:
        continue

    snapshot = row["Baseline"].copy()
    elapsed = int((now - row["OrderDateTime"]).total_seconds() / 60)
    elapsed = max(elapsed, 0)

    snapshot["Elapsed Minutes"] = elapsed
    updated_remaining = reg_model.predict(pd.DataFrame([snapshot]))[0]

    # Risk logic
    if updated_remaining >= 240:
        risk = "High"
    elif updated_remaining >= 180:
        risk = "Moderate"
    else:
        risk = "Low"

    board_rows.append({
        "MRN": row["MRN"],
        "OrderDateTime": row["OrderDateTime"],
        "Risk": risk,
        "OriginalProjection": int(row["LastPrediction"]),
        "Elapsed": elapsed,
        "Remaining": int(updated_remaining)
    })

# Convert to DataFrame and sort
board_df = pd.DataFrame(board_rows)

if not board_df.empty:

    board_df = board_df.sort_values(
        "Remaining", ascending=False
    ).reset_index(drop=True)

    # Header
    header_cols = st.columns([1, 2, 2, 1, 2, 1, 2])
    headers = [
        "Rank",
        "MRN",
        "Order DateTime",
        "Risk",
        "Original (mins)",
        "Remaining (mins)",
        "Action"
    ]

    for col, header in zip(header_cols, headers):
        col.markdown(f"**{header}**")

    # Rows
    for i, row in board_df.iterrows():

        cols = st.columns([1, 2, 2, 1, 2, 1, 2])

        cols[0].write(i + 1)
        cols[1].write(row["MRN"])
        cols[2].write(row["OrderDateTime"].strftime("%Y-%m-%d %H:%M"))
        cols[3].write(row["Risk"])
        cols[4].write(row["OriginalProjection"])
        cols[5].write(row["Remaining"])

        if cols[6].button("Mark Discharged", key=f"discharge_{row['MRN']}"):

            completion_dt = datetime.now(PH_TZ)

            original_row = st.session_state.risk_registry[
                st.session_state.risk_registry["MRN"] == row["MRN"]
            ].iloc[0]

            final_duration = (
                completion_dt - original_row["OrderDateTime"]
            ).total_seconds() / 60

            error = final_duration - original_row["LastPrediction"]

            c.execute("""
            INSERT INTO discharge_records
            VALUES (?, ?, ?, ?, ?, ?)
            """, (
                row["MRN"],
                original_row["OrderDateTime"].isoformat(),
                completion_dt.isoformat(),
                final_duration,
                original_row["LastPrediction"],
                error
            ))

            conn.commit()

            # Remove from active board
            st.session_state.risk_registry = (
                st.session_state.risk_registry[
                    st.session_state.risk_registry["MRN"] != row["MRN"]
                ]
            )

            st.success(f"{row['MRN']} discharged. Error: {round(error,2)} mins")
            st.rerun()

