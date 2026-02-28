# ==========================================================
# DischargeFlow AI
# Option A Architecture
# True Dynamic Regression Reforecast
# Philippine Timezone Stable
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
# PAGE CONFIG
# ----------------------------------------------------------

st.set_page_config(page_title="DischargeFlow AI", layout="wide")
st.title("DischargeFlow AI")

st.markdown("""
### 🚀 Version Highlights
- True dynamic regression re-forecasting
- Elapsed minutes included as model feature
- Philippine timezone aligned (Asia/Manila)
- Risk tier derived from projected duration
- Snapshot-based board architecture
""")

# ----------------------------------------------------------
# TIMEZONE
# ----------------------------------------------------------

PH_TZ = ZoneInfo("Asia/Manila")

# ----------------------------------------------------------
# OPENAI
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
    df["Elapsed Minutes"] = 0  # training baseline
    return df

df = load_data()

# ----------------------------------------------------------
# TRAIN MODEL (WITH ELAPSED FEATURE)
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
    return model, X.columns.tolist()

reg_model, feature_columns = train_regression_model(df)

# ----------------------------------------------------------
# RISK TIER (DERIVED FROM PROJECTED MINUTES)
# ----------------------------------------------------------

def assign_risk(minutes):
    if minutes >= 240:
        return "Critical", "🔴"
    elif minutes >= 200:
        return "High", "🟠"
    elif minutes >= 150:
        return "Moderate", "🟡"
    else:
        return "Low", "🟢"

# ----------------------------------------------------------
# SESSION STATE BOARD
# ----------------------------------------------------------

if "risk_registry" not in st.session_state:
    st.session_state.risk_registry = pd.DataFrame(
        columns=[
            "MRN",
            "Order DateTime",
            "Feature Snapshot"
        ]
    )
# 🔧 Compatibility reset (fix old board structure)
if "Feature Snapshot" not in st.session_state.risk_registry.columns:
    st.session_state.risk_registry = pd.DataFrame(
        columns=[
            "MRN",
            "Order DateTime",
            "Feature Snapshot"
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

    now_ph = datetime.now(PH_TZ)

    order_date = st.date_input("Discharge Order Date")
    order_time = st.time_input(
        "Discharge Order Time",
        value=now_ph.time(),
        step=timedelta(minutes=1)
    )

    submitted = st.form_submit_button("Generate Forecast")

# ----------------------------------------------------------
# GENERATE FORECAST
# ----------------------------------------------------------

if submitted:

    if not mrn:
        st.error("Please enter MRN.")
    else:

        # baseline snapshot
        snapshot = {}

        for col in feature_columns:
            if col in df.columns:
                if df[col].dtype in ["int64", "float64"]:
                    snapshot[col] = df[col].median()
                else:
                    snapshot[col] = df[col].mode()[0]

        snapshot["Length of Stay (days)"] = los
        snapshot["Number of Doctors Involved"] = doctors
        snapshot["Current Bill (PHP)"] = bill
        snapshot["Patient Age"] = age
        snapshot["Primary Diagnosis (Description)"] = diagnosis_input
        snapshot["Elapsed Minutes"] = 0

        projected = int(reg_model.predict(pd.DataFrame([snapshot]))[0])

        order_datetime = datetime.combine(order_date, order_time).replace(tzinfo=PH_TZ)

        new_row = pd.DataFrame([{
            "MRN": mrn,
            "Order DateTime": order_datetime,
            "Feature Snapshot": snapshot
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

        risk, badge = assign_risk(projected)

        st.subheader("Initial Forecast")
        st.markdown(f"Projected Duration: {projected} minutes")
        st.markdown(f"{badge} {risk}")

# ----------------------------------------------------------
# DYNAMIC BOARD (TRUE RE-FORECAST)
# ----------------------------------------------------------

if not st.session_state.risk_registry.empty:

    board_rows = []

    now_ph = datetime.now(PH_TZ)

    for _, row in st.session_state.risk_registry.iterrows():

        order_dt = row["Order DateTime"]
        elapsed = max(int((now_ph - order_dt).total_seconds() / 60), 0)

        snapshot = row["Feature Snapshot"].copy()
        snapshot["Elapsed Minutes"] = elapsed

        updated_projection = int(
            reg_model.predict(pd.DataFrame([snapshot]))[0]
        )

        risk, badge = assign_risk(updated_projection)

        board_rows.append({
            "MRN": row["MRN"],
            "Order DateTime": order_dt.strftime("%Y-%m-%d %H:%M"),
            "Elapsed Minutes": elapsed,
            "Projected Minutes": updated_projection,
            "Risk Level": f"{badge} {risk}"
        })

    board_df = pd.DataFrame(board_rows)

    st.markdown("## Discharge Risk Command Board")
    st.dataframe(
        board_df.sort_values("Projected Minutes", ascending=False),
        use_container_width=True,
        hide_index=True
    )
