import joblib
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Credit Card Fraud Detector", page_icon="💳", layout="wide")


@st.cache_resource
def load_assets():
    return joblib.load("models/xgb_model.joblib"), joblib.load("models/scaler.joblib")


@st.cache_data
def load_samples():
    return pd.read_csv("data/sample_transactions.csv")


model, scaler = load_assets()
samples = load_samples()
FEATURES = [c for c in samples.columns if c != "Class"]


def score(df):
    X = df[FEATURES].copy()
    X[["Time", "Amount"]] = scaler.transform(X[["Time", "Amount"]])
    return model.predict_proba(X)[:, 1]


st.title("💳 Credit Card Fraud Detector")
st.caption("XGBoost trained on 284,807 transactions (only 0.17% fraud). Test AUPRC: 0.879")

threshold = st.sidebar.slider(
    "Fraud threshold", 0.01, 0.99, 0.50, 0.01,
    help="Flag a transaction as fraud if its probability is above this. "
         "Lower = catch more fraud but more false alarms.",
)
st.sidebar.write("Lower threshold: catches more fraud, more false alarms.")
st.sidebar.write("Higher threshold: fewer false alarms, misses more fraud.")

tab1, tab2 = st.tabs(["Try a sample transaction", "Upload your own CSV"])

with tab1:
    st.write("Pick a real transaction from the held-out test set and see what the model says.")
    choice = st.radio("Show me a", ["Genuine transaction", "Fraudulent transaction"], horizontal=True)
    pool = samples[samples["Class"] == (1 if choice.startswith("Fraud") else 0)].reset_index(drop=True)
    idx = st.number_input("Transaction number", min_value=0, max_value=len(pool) - 1, value=0, step=1)
    row = pool.iloc[[int(idx)]]
    prob = float(score(row)[0])

    c1, c2, c3 = st.columns(3)
    c1.metric("Amount", f"{row['Amount'].iloc[0]:.2f}")
    c2.metric("Fraud probability", f"{prob:.1%}")
    c3.metric("Actual label", "Fraud" if row["Class"].iloc[0] == 1 else "Genuine")

    if prob >= threshold:
        st.error("🚨 Flagged as FRAUD")
    else:
        st.success("✅ Looks genuine")

    with st.expander("See the raw transaction values"):
        st.dataframe(row[FEATURES])

with tab2:
    st.write("Upload a CSV with the same columns as the Kaggle dataset (Time, V1 to V28, Amount).")
    up = st.file_uploader("CSV file", type="csv")
    if up:
        data = pd.read_csv(up)
        missing = [c for c in FEATURES if c not in data.columns]
        if missing:
            st.error(f"Missing columns: {missing}")
        else:
            data["fraud_probability"] = score(data)
            data["flagged"] = data["fraud_probability"] >= threshold
            m1, m2 = st.columns(2)
            m1.metric("Transactions scored", f"{len(data):,}")
            m2.metric("Flagged as fraud", int(data["flagged"].sum()))
            st.dataframe(data[data["flagged"]].sort_values("fraud_probability", ascending=False))
            st.download_button("Download all results", data.to_csv(index=False), "scored_transactions.csv")
    else:
        st.info("No file handy? Download this sample and upload it back.")
        st.download_button("Download sample CSV", samples.drop(columns="Class").to_csv(index=False),
                           "sample_transactions.csv")