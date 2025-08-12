import streamlit as st
import numpy as np
import joblib

# Load model and scaler
model = joblib.load("rf_model.joblib")
scaler = joblib.load("rf_scaler.joblib")

st.title("Instagram Fake Account Detector")

# Inputs (raw values)
followers = st.number_input("Followers", min_value=0)
following = st.number_input("Following", min_value=0)
uname_len = st.number_input("Username Length", min_value=0)
uname_has_num = st.selectbox("Username has number?", ["No", "Yes"])
fname_has_num = st.selectbox("Full Name has number?", ["No", "Yes"])
fname_len = st.number_input("Full Name Length", min_value=0)
private = st.selectbox("Private Account?", ["No", "Yes"])
new_acc = st.selectbox("New Account?", ["No", "Yes"])

if st.button("🔍 Predict"):
    # Convert Yes/No to 1/0
    uname_has_num_val = 1 if uname_has_num == "Yes" else 0
    fname_has_num_val = 1 if fname_has_num == "Yes" else 0
    private_val = 1 if private == "Yes" else 0
    new_acc_val = 1 if new_acc == "Yes" else 0

    # Arrange in correct order
    data = np.array([[followers, following, uname_len, uname_has_num_val,
                      fname_has_num_val, fname_len, private_val, new_acc_val]])
    
    # Scale raw inputs using the same scaler as training
    data_scaled = scaler.transform(data)
    
    # Predict
    pred = model.predict(data_scaled)[0]
    prob = model.predict_proba(data_scaled)[0][1]

    if pred == 1:
        st.error(f"🚨 FAKE Account (Confidence: {prob*100:.2f}%)")
    else:
        st.success(f"✅ Real Account (Confidence: {(1-prob)*100:.2f}%)")
