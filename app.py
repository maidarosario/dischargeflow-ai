# ==========================================================
# DischargeFlow AI
# Stable Production Version
# Classification + Regression + Acceleration Risk
# ==========================================================

import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
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
st.caption("Live discharge delay monitoring and operational command board.")

st.markdown("""
### System Highlights
- Real-time discharge delay probability
- Projected discharge duration (minutes)
- Dynamic elapsed-time recalculation
- Acceleration risk detection
- Live command board updates
""")

# ----------------------------------------------------------
# Auto Refresh (30 sec)
# ----------------------------------------------------------

if "last_refresh" not in st.session_state:
    st.session_state.last_refresh = time.time()

if time.time() - st.session_state.last_refresh > 30:
    st.session_state.last_refresh = time.time()
    st.rerun()

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

    # synthetic elapsed for training
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

    clf = Pipeline([
        ("pre", preprocessor),
        ("clf", GradientBoostingClassifier(random_state=42))
    ])

    reg = Pipeline([
        ("pre", preprocessor),
        ("reg", GradientBoostingRegressor(random_state=42))
    ])

    X_train, _, y_train_class, _ = train_test_split(
        X, y_class, test_size=0.2, random_state=42
    )

    _, _, y_train_reg, _ = train_test_split(
        X, y_reg, test_size=0.2, random_state=42
    )

    clf.fit(X_train, y_train_class)
    reg.fit(X_train, y_train_reg)

    return clf, reg

clf_model, reg_model = train_models(df)

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
# Initialize Risk Registry
# ----------------------------------------------------------

REQUIRED_COLUMNS = [
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

if "risk_registry" not in st.session_state:
    st.session_state.risk_registry = pd.DataFrame(columns=REQUIRED_COLUMNS)
else:
    # ensure schema safety
    for col in REQUIRED_COLUMNS:
        if col not in st.session_state.risk_registry.columns:
            st.session_state.risk_registry[col] = None

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

    submitted = st.form_submit_button("Generate Prediction")

# ----------------------------------------------------------
# Generate Prediction
# ----------------------------------------------------------

if submitted:

    if not mrn:
        st.error("Please enter MRN.")
    else:

        order_datetime = datetime.combine(order_date, order_time)
        elapsed = int((datetime.now() - order_datetime).total_seconds() / 60)

        # Build feature row using training structure
        feature_template = df.drop(
            columns=["delayed_over_200mins",
                     "Discharge Duration (minutes)"]
        ).iloc[0].copy()

        feature_template["Length of Stay (days)"] = los
        feature_template["Number of Doctors Involved"] = doctors
        feature_template["Current Bill (PHP)"] = bill
        feature_template["Patient Age"] = age
        feature_template["Primary Diagnosis (Description)"] = diagnosis_input
        feature_template["Elapsed Minutes"] = elapsed

        input_df = pd.DataFrame([feature_template])

        prob = clf_model.predict_proba(input_df)[0][1]
        projected = reg_model.predict(input_df)[0]

        severity = assign_severity(prob)

        st.subheader("Prediction")
        st.progress(float(prob))
        st.markdown(f"Delay Probability: {round(prob,2)}")
        st.markdown(f"Projected Duration: {int(projected)} minutes")

        new_row = pd.DataFrame([{
            "MRN": mrn,
            "Risk Level": severity,
            "Order DateTime": order_datetime,
            "Delay Probability": round(prob,3),
            "Original Projected Minutes": int(projected),
            "Updated Projected Minutes": int(projected),
            "Elapsed Minutes": elapsed,
            "Acceleration Risk": "NO",
            "Feature Snapshot": feature_template.to_dict()
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

    board = st.session_state.risk_registry.copy()

    updated_proj = []
    updated_elapsed = []
    acceleration_flags = []

    for _, row in board.iterrows():

        snapshot = row.get("Feature Snapshot")

        if not isinstance(snapshot, dict):
            updated_proj.append(row["Updated Projected Minutes"])
            updated_elapsed.append(row["Elapsed Minutes"])
            acceleration_flags.append("NO")
            continue

        elapsed_now = int(
            (datetime.now() - row["Order DateTime"]).total_seconds() / 60
        )

        snapshot["Elapsed Minutes"] = elapsed_now

        temp_df = pd.DataFrame([snapshot])
        new_proj = reg_model.predict(temp_df)[0]

        updated_proj.append(int(new_proj))
        updated_elapsed.append(elapsed_now)

        if elapsed_now >= 0.8 * new_proj:
            acceleration_flags.append("YES")
        else:
            acceleration_flags.append("NO")

    board["Elapsed Minutes"] = updated_elapsed
    board["Updated Projected Minutes"] = updated_proj
    board["Acceleration Risk"] = acceleration_flags

    board = board.sort_values(
        by="Delay Probability",
        ascending=False
    ).reset_index(drop=True)

    def highlight_accel(val):
        if val == "YES":
            return "background-color: #ff6b6b; color: white;"
        return ""

    styled = board.style.applymap(
        highlight_accel,
        subset=["Acceleration Risk"]
    )

    st.markdown("## Discharge Risk Command Board")
    st.dataframe(
        styled.drop(columns=["Feature Snapshot"]),
        use_container_width=True,
        hide_index=True
    )