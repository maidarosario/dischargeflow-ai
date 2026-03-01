# ==========================================================
# DischargeFlow AI – Stable Operational Version
# Elapsed-Aware + Delta Status + Dual Tables
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
# DATABASE
# ----------------------------------------------------------

conn = sqlite3.connect("dischargeflow.db", check_same_thread=False)
c = conn.cursor()

c.execute("""
CREATE TABLE IF NOT EXISTS discharge_records (
    MRN TEXT,
    OrderDateTime TEXT,
    CompletionDateTime TEXT,
    FinalDuration REAL,
    OriginalProjection REAL,
    Error REAL
)
""")
conn.commit()

# ----------------------------------------------------------
# LOAD + EXPAND DATA
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

    return model, mae, r2

reg_model, mae, r2 = train_model(df)

# ----------------------------------------------------------
# MODEL PERFORMANCE (Hidden)
# ----------------------------------------------------------

with st.expander("View Model Performance Details"):
    st.write(f"MAE: {round(mae,2)} minutes")
    st.write(f"R²: {round(r2,3)}")

# ----------------------------------------------------------
# SESSION STATE
# ----------------------------------------------------------

if "risk_registry" not in st.session_state:
    st.session_state.risk_registry = pd.DataFrame(
        columns=["MRN", "OrderDateTime", "Baseline", "OriginalProjection"]
    )

if "discharged_registry" not in st.session_state:
    st.session_state.discharged_registry = pd.DataFrame(
        columns=[
            "MRN",
            "OrderDateTime",
            "OriginalProjection",
            "FinalDuration",
            "Error"
        ]
    )

# ----------------------------------------------------------
# PATIENT INPUT
# ----------------------------------------------------------

st.markdown("## New Patient Forecast")

with st.form("patient_form", clear_on_submit=True):

    mrn = st.text_input("MRN")
    los = st.number_input("Length of Stay (days)", 0, 100, 5)
    doctors = st.slider("Number of Doctors Involved", 1, 30, 2)
    bill = st.number_input("Current Bill (PHP)", 0, 2000000, 50000)
    age = st.slider("Patient Age", 0, 120, 40)
    diagnosis = st.text_input("Primary Diagnosis")

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
            "OriginalProjection": int(remaining)
        }])
    ], ignore_index=True)

    st.subheader("Initial Forecast")
    st.markdown(f"Projected Remaining Time: **{int(remaining)} minutes**")
    # ----------------------------------------------------------
    # Discharge Team AI Advisory
    # ----------------------------------------------------------
    
    if client:
    
        advisory_prompt = f"""
    You are advising a hospital discharge operations team.
    
    Projected Remaining Time: {int(remaining)} minutes
    
    Provide structured bullet points under:
    
    Operational Risks
    Clinical Coordination Actions
    Discharge Process Actions
    Escalation Plan
    
    Keep it concise and practical.
    """
    
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "Hospital discharge operations advisor."},
                {"role": "user", "content": advisory_prompt}
            ],
            temperature=0.2
        )

    st.markdown("## Discharge Team Operational Advisory")
    st.markdown(response.choices[0].message.content)

# ----------------------------------------------------------
# COMMAND BOARD
# ----------------------------------------------------------

st.markdown("## Active Discharge Board")

board_rows = []
now = datetime.now(PH_TZ)

for _, row in st.session_state.risk_registry.iterrows():

    snapshot = row["Baseline"].copy()
    elapsed = int((now - row["OrderDateTime"]).total_seconds() / 60)
    elapsed = max(elapsed, 0)

    snapshot["Elapsed Minutes"] = elapsed
    updated_remaining = reg_model.predict(pd.DataFrame([snapshot]))[0]

    expected_remaining = row["OriginalProjection"] - elapsed
    delta = int(updated_remaining - expected_remaining)

    if delta > 30:
        status = "Delayed – Behind Expected Timeline"
    elif delta > 10:
        status = "Slight Delay"
    elif delta < -30:
        status = "Significantly Ahead"
    elif delta < -10:
        status = "Ahead of Schedule"
    else:
        status = "On Track"

    board_rows.append({
        "MRN": row["MRN"],
        "OrderDateTime": row["OrderDateTime"],
        "OriginalProjection": row["OriginalProjection"],
        "Remaining": int(updated_remaining),
        "Status": status
    })

board_df = pd.DataFrame(board_rows)

if not board_df.empty:

    board_df = board_df.sort_values("Remaining", ascending=False).reset_index(drop=True)

    headers = st.columns([1,2,2,2,2,3,2])
    header_labels = ["Rank","MRN","Order DateTime","Original","Remaining","Status","Action"]

    for col, label in zip(headers, header_labels):
        col.markdown(f"**{label}**")

    for i, row in board_df.iterrows():

        cols = st.columns([1,2,2,2,2,3,2])

        cols[0].write(i+1)
        cols[1].write(row["MRN"])
        cols[2].write(row["OrderDateTime"].strftime("%Y-%m-%d %H:%M"))
        cols[3].write(row["OriginalProjection"])
        cols[4].write(row["Remaining"])
        cols[5].write(row["Status"])

        if cols[6].button("Mark Discharged", key=f"dis_{row['MRN']}"):

            completion_dt = datetime.now(PH_TZ)

            original_row = st.session_state.risk_registry[
                st.session_state.risk_registry["MRN"] == row["MRN"]
            ].iloc[0]

            final_duration = (
                completion_dt - original_row["OrderDateTime"]
            ).total_seconds() / 60

            error = final_duration - original_row["OriginalProjection"]

            c.execute("""
            INSERT INTO discharge_records
            VALUES (?, ?, ?, ?, ?, ?)
            """, (
                row["MRN"],
                original_row["OrderDateTime"].isoformat(),
                completion_dt.isoformat(),
                final_duration,
                original_row["OriginalProjection"],
                error
            ))
            conn.commit()

            st.session_state.discharged_registry = pd.concat([
                st.session_state.discharged_registry,
                pd.DataFrame([{
                    "MRN": row["MRN"],
                    "OrderDateTime": original_row["OrderDateTime"],
                    "OriginalProjection": original_row["OriginalProjection"],
                    "FinalDuration": int(final_duration),
                    "Error": round(error,2)
                }])
            ], ignore_index=True)

            st.session_state.risk_registry = (
                st.session_state.risk_registry[
                    st.session_state.risk_registry["MRN"] != row["MRN"]
                ]
            )

            st.rerun()

# ----------------------------------------------------------
# DISCHARGED TABLE
# ----------------------------------------------------------

st.markdown("## Discharged Patients")

if not st.session_state.discharged_registry.empty:

    discharged_df = st.session_state.discharged_registry.copy()
    discharged_df["OrderDateTime"] = discharged_df["OrderDateTime"].apply(
        lambda x: x.strftime("%Y-%m-%d %H:%M")
    )

    st.dataframe(
        discharged_df.sort_values("OrderDateTime", ascending=False),
        use_container_width=True,
        hide_index=True
    )

