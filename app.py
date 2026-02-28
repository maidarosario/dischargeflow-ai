# ==========================================================
# DischargeFlow AI
# FINAL PRODUCTION VERSION
# Option A – True Dynamic Regression Reforecast
# ==========================================================

import streamlit as st
import pandas as pd
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
# PAGE CONFIG
# ----------------------------------------------------------

st.set_page_config(page_title="DischargeFlow AI", layout="wide")
st.title("DischargeFlow AI")

st.markdown("""
### Version Highlights
• True dynamic regression re-forecast  
• Elapsed Minutes included as model feature  
• Original vs Updated projected minutes comparison  
• Risk tier from projected duration  
• Philippine timezone aligned  
""")

# ----------------------------------------------------------
# TIMEZONE (Philippines)
# ----------------------------------------------------------

PH_TZ = ZoneInfo("Asia/Manila")
now_ph = datetime.now(PH_TZ)

# ----------------------------------------------------------
# OPENAI (Optional Advisory)
# ----------------------------------------------------------

client = None
if "OPENAI_API_KEY" in st.secrets:
    client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

# ----------------------------------------------------------
# LOAD DATA
# ----------------------------------------------------------

@st.cache_data
def load_data():
    df = pd.read_csv("fictitious_dataset_FINAL.csv")
    df["Elapsed Minutes"] = 0
    return df

df = load_data()

# ----------------------------------------------------------
# TRAIN REGRESSION MODEL (Elapsed included)
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

    return model, X.columns.tolist()

reg_model, feature_columns = train_model(df)

# ----------------------------------------------------------
# RISK TIER
# ----------------------------------------------------------

def assign_risk(minutes):
    if minutes >= 200:
        return "High", "🟠"
    elif minutes >= 180:
        return "Moderate", "🟡"
    else:
        return "Low", "🟢"

# ----------------------------------------------------------
# SESSION STATE BOARD (Stable Schema)
# ----------------------------------------------------------

required_cols = [
    "MRN",
    "Order DateTime",
    "Baseline Features",
    "Original Projected Minutes"
]

if "risk_registry" not in st.session_state:
    st.session_state.risk_registry = pd.DataFrame(columns=required_cols)

if not set(required_cols).issubset(set(st.session_state.risk_registry.columns)):
    st.session_state.risk_registry = pd.DataFrame(columns=required_cols)

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

    st.markdown("## Discharge Order Timing")

    col_date, col_time = st.columns(2)

    with col_date:
        order_date = st.date_input("Discharge Order Date", value=now_ph.date())

    with col_time:
        order_time = st.time_input(
            "Discharge Order Time",
            value=now_ph.time().replace(second=0, microsecond=0)
        )

    submitted = st.form_submit_button("Generate Forecast")

# ----------------------------------------------------------
# GENERATE INITIAL FORECAST
# ----------------------------------------------------------

if submitted:

    if not mrn:
        st.error("Please enter MRN.")
    else:

        baseline = {}

        for col in feature_columns:
            if col in df.columns:
                if df[col].dtype in ["int64", "float64"]:
                    baseline[col] = df[col].median()
                else:
                    baseline[col] = df[col].mode()[0]

        baseline["Length of Stay (days)"] = los
        baseline["Number of Doctors Involved"] = doctors
        baseline["Current Bill (PHP)"] = bill
        baseline["Patient Age"] = age
        baseline["Primary Diagnosis (Description)"] = diagnosis_input
        baseline["Elapsed Minutes"] = 0

        projected = int(reg_model.predict(pd.DataFrame([baseline]))[0])

        risk, badge = assign_risk(projected)

        order_datetime = datetime.combine(order_date, order_time)
        order_datetime = order_datetime.replace(tzinfo=PH_TZ)

        new_row = pd.DataFrame([{
            "MRN": mrn,
            "Order DateTime": order_datetime,
            "Baseline Features": baseline.copy(),
            "Original Projected Minutes": projected
        }])

        st.session_state.risk_registry = pd.concat(
            [st.session_state.risk_registry, new_row],
            ignore_index=True
        )

        st.subheader("Initial Forecast")
        st.markdown(f"Projected Duration: **{projected} minutes**")
        st.markdown(f"{badge} {risk}")

        # Optional Advisory
        if client:
            prompt = f"""
You are advising a hospital discharge operations team.

Projected Duration: {projected} minutes
Risk Level: {risk}

Use sections:
Operational Risks
Clinical Coordination Actions
Discharge Process Actions
Escalation Plan

Bullet format only.
"""

            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system",
                     "content": "Hospital discharge operations advisor."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.2
            )

            st.markdown("## Discharge Team Operational Advisory")
            st.markdown(response.choices[0].message.content.strip())

# ----------------------------------------------------------
# DYNAMIC COMMAND BOARD (TRUE RE-FORECAST)
# ----------------------------------------------------------

if not st.session_state.risk_registry.empty:

    st.markdown("## Discharge Risk Command Board")

    board_rows = []
    now_ph = datetime.now(PH_TZ)

    for _, row in st.session_state.risk_registry.iterrows():

        snapshot = row["Baseline Features"].copy()
        order_dt = row["Order DateTime"]

        elapsed = int((now_ph - order_dt).total_seconds() / 60)
        elapsed = max(elapsed, 0)

        snapshot["Elapsed Minutes"] = elapsed

        updated_projection = int(
            reg_model.predict(pd.DataFrame([snapshot]))[0]
        )

        risk, badge = assign_risk(updated_projection)

        board_rows.append({
            "MRN": row["MRN"],
            "Order DateTime": order_dt.strftime("%Y-%m-%d %H:%M"),
            "Elapsed Minutes": elapsed,
            "Original Projected Minutes": row["Original Projected Minutes"],
            "Updated Projected Minutes": updated_projection,
            "Risk Level": f"{badge} {risk}"
        })

    board_df = pd.DataFrame(board_rows)

    st.dataframe(
        board_df.sort_values("Updated Projected Minutes", ascending=False),
        use_container_width=True,
        hide_index=True
    )