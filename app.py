import joblib
import pandas as pd

# Load the trained model
import os
import joblib

model_path = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "credit_default_model.joblib"
)

model = joblib.load(model_path)

print("Model loaded successfully!")

print("Model loaded successfully!")
print("Model type:", type(model))
print("Number of features:", model.n_features_in_)

# Example input
data = {
    'LIMIT_BAL': [20000],
    'SEX': [2],
    'EDUCATION': [2],
    'MARRIAGE': [2],
    'AGE': [24],
    'PAY_0': [2],
    'PAY_2': [2],
    'PAY_3': [-1],
    'PAY_4': [-1],
    'PAY_5': [-2],
    'PAY_6': [-2],
    'BILL_AMT1': [3913],
    'BILL_AMT2': [3102],
    'BILL_AMT3': [689],
    'BILL_AMT4': [0],
    'BILL_AMT5': [0],
    'BILL_AMT6': [0],
    'PAY_AMT1': [0],
    'PAY_AMT2': [689],
    'PAY_AMT3': [0],
    'PAY_AMT4': [0],
    'PAY_AMT5': [0],
    'PAY_AMT6': [0]
}

input_data = pd.DataFrame(data)

prediction = model.predict(input_data)
probability = model.predict_proba(input_data)[0][1]

print("Prediction:", prediction[0])
print("Default probability:", probability)

# ============================================================
# USER INPUT SECTION
# ============================================================

print("\nEnter customer details for a new prediction:")

limit_bal = float(input("LIMIT_BAL: "))
sex = int(input("SEX: "))
education = int(input("EDUCATION: "))
marriage = int(input("MARRIAGE: "))
age = int(input("AGE: "))

pay_0 = int(input("PAY_0: "))
pay_2 = int(input("PAY_2: "))
pay_3 = int(input("PAY_3: "))
pay_4 = int(input("PAY_4: "))
pay_5 = int(input("PAY_5: "))
pay_6 = int(input("PAY_6: "))

bill_amt1 = float(input("BILL_AMT1: "))
bill_amt2 = float(input("BILL_AMT2: "))
bill_amt3 = float(input("BILL_AMT3: "))
bill_amt4 = float(input("BILL_AMT4: "))
bill_amt5 = float(input("BILL_AMT5: "))
bill_amt6 = float(input("BILL_AMT6: "))

pay_amt1 = float(input("PAY_AMT1: "))
pay_amt2 = float(input("PAY_AMT2: "))
pay_amt3 = float(input("PAY_AMT3: "))
pay_amt4 = float(input("PAY_AMT4: "))
pay_amt5 = float(input("PAY_AMT5: "))
pay_amt6 = float(input("PAY_AMT6: "))

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

new_prediction = model.predict(new_customer)
new_probability = model.predict_proba(new_customer)[0][1]

print("\nPrediction:", new_prediction[0])
print("Default probability:", new_probability)