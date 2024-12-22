import streamlit as st
import joblib

# Load the model
model = joblib.load('logistic_regression_model.pkl')

# Streamlit app layout
st.title("Employee Attrition Predictor")
st.write("Enter employee details to predict the likelihood of attrition.")

# Collect input from users
age = st.slider("Age", 20, 60, 30)
tenure = st.slider("Tenure", 0, 40, 5)
salary = st.number_input("Salary", min_value=50000, max_value=150000, value=100000)
overtime = st.selectbox("Overtime", ["Yes", "No"])
department = st.selectbox("Department", ["Development", "QA", "Support", "Management", "DevOps"])

# Collect other necessary features...

# Preprocess inputs as needed (e.g., encoding)
# ...

# Predict attrition
if st.button("Predict"):
    prediction = model.predict([[age, tenure, salary, overtime, department]])
    st.write("Attrition Prediction:", "Yes" if prediction[0] == 1 else "No")
