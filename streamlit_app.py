import streamlit as st
import pandas as pd
import joblib
import os

# Load trained model
model_path = "credit_default_model.joblib"

model = joblib.load(model_path)

st.set_page_config(
    page_title="Credit Card Default Prediction",
    page_icon="💳",
    layout="wide"
)

st.title("💳 Credit Card Default Prediction")

st.write(
    "Predict the probability of credit card default "
    "using a tuned Random Forest model."
)

st.success("Model loaded successfully!")
# ============================================================
# CUSTOMER INFORMATION
# ============================================================

st.header("Customer Information")

limit_bal = st.number_input(
    "Credit Limit (LIMIT_BAL)",
    min_value=0.0,
    value=20000.0
)

sex = st.selectbox(
    "Sex (SEX)",
    options=[1, 2],
    format_func=lambda x: "Male" if x == 1 else "Female"
)

education = st.number_input(
    "Education (EDUCATION)",
    min_value=0,
    value=2,
    step=1
)

marriage = st.number_input(
    "Marriage Status (MARRIAGE)",
    min_value=0,
    value=2,
    step=1
)

age = st.number_input(
    "Age (AGE)",
    min_value=18,
    max_value=100,
    value=24,
    step=1
)
# ============================================================
# PAYMENT HISTORY
# ============================================================

st.header("Payment History")

pay_0 = st.number_input("PAY_0", value=2, step=1)
pay_2 = st.number_input("PAY_2", value=2, step=1)
pay_3 = st.number_input("PAY_3", value=-1, step=1)
pay_4 = st.number_input("PAY_4", value=-1, step=1)
pay_5 = st.number_input("PAY_5", value=-2, step=1)
pay_6 = st.number_input("PAY_6", value=-2, step=1)

# ============================================================
# BILL AMOUNTS
# ============================================================

st.header("Bill Amounts")

bill_amt1 = st.number_input("BILL_AMT1", value=3913.0)
bill_amt2 = st.number_input("BILL_AMT2", value=3102.0)
bill_amt3 = st.number_input("BILL_AMT3", value=689.0)
bill_amt4 = st.number_input("BILL_AMT4", value=0.0)
bill_amt5 = st.number_input("BILL_AMT5", value=0.0)
bill_amt6 = st.number_input("BILL_AMT6", value=0.0)


# ============================================================
# PREVIOUS PAYMENTS
# ============================================================

st.header("Previous Payments")

pay_amt1 = st.number_input("PAY_AMT1", value=0.0)
pay_amt2 = st.number_input("PAY_AMT2", value=689.0)
pay_amt3 = st.number_input("PAY_AMT3", value=0.0)
pay_amt4 = st.number_input("PAY_AMT4", value=0.0)
pay_amt5 = st.number_input("PAY_AMT5", value=0.0)
pay_amt6 = st.number_input("PAY_AMT6", value=0.0)

# ============================================================
# PREDICTION
# ============================================================

st.header("Prediction")

if st.button("Predict Default Risk"):

    new_customer = pd.DataFrame({
        'LIMIT_BAL': [limit_bal],
        'SEX': [sex],
        'EDUCATION': [education],
        'MARRIAGE': [marriage],
        'AGE': [age],
        'PAY_0': [pay_0],
        'PAY_2': [pay_2],
        'PAY_3': [pay_3],
        'PAY_4': [pay_4],
        'PAY_5': [pay_5],
        'PAY_6': [pay_6],
        'BILL_AMT1': [bill_amt1],
        'BILL_AMT2': [bill_amt2],
        'BILL_AMT3': [bill_amt3],
        'BILL_AMT4': [bill_amt4],
        'BILL_AMT5': [bill_amt5],
        'BILL_AMT6': [bill_amt6],
        'PAY_AMT1': [pay_amt1],
        'PAY_AMT2': [pay_amt2],
        'PAY_AMT3': [pay_amt3],
        'PAY_AMT4': [pay_amt4],
        'PAY_AMT5': [pay_amt5],
        'PAY_AMT6': [pay_amt6]
    })

    prediction = model.predict(new_customer)
    probability = model.predict_proba(new_customer)[0][1]

    if prediction[0] == 1:
        st.error("⚠️ Predicted: Customer may default")
    else:
        st.success("✅ Predicted: Customer may not default")

    st.metric(
        "Default Probability",
        f"{probability:.2%}"
    )