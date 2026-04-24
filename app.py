import streamlit as st
from src.predict import predict_churn

st.title("Customer Churn Prediction (KNN)")
 
age = st.number_input("Age", min_value=18, max_value=80)
gender = st.selectbox("Gender", ["Male", "Female"])
plan = st.selectbox("Plan Type", ["Prepaid", "Postpaid"])
tenure = st.number_input("Tenure (months)", min_value=1, max_value=60)
usage = st.number_input("Monthly Usage", min_value=50, max_value=1000)

if st.button("Predict churn"):
    result = predict_churn(age, gender, plan, tenure, usage)
    if result == 1:
        st.error("Customer is likely to churn.")
    else:
        st.success("Customer is likely to stay.")

        