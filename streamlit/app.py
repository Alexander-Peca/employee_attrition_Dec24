import streamlit as st
import joblib
import numpy as np

# Paths to the saved models
model_path = r"C:\Users\alexa\OneDrive\AlexPerez\Dokumente\2 Soros\Data Science\Projects\employee_attrition_Dec24\models\logistic_regression_model.pkl"
threshold_path = r"C:\Users\alexa\OneDrive\AlexPerez\Dokumente\2 Soros\Data Science\Projects\employee_attrition_Dec24\models\logistic_regression_threshold.pkl"

# Load the model and threshold
model = joblib.load(model_path)
threshold = joblib.load(threshold_path)

# Streamlit app layout
st.title("Employee Attrition Predictor")
st.write("Enter employee details to predict the likelihood of attrition.")

# Collect input from users
age = st.slider("Age", 20, 60, 30)
tenure = st.slider("Tenure (Years)", 0, 40, 5)
salary = st.number_input("Salary (USD)", min_value=50000, max_value=150000, value=100000, step=1000)
overtime = st.selectbox("Overtime", ["Yes", "No"])
department = st.selectbox("Department", ["Development", "QA", "Support", "Management", "DevOps"])

# Encode categorical inputs
overtime_encoded = 1 if overtime == "Yes" else 0
department_encoded = [
    1 if department == "Development" else 0,
    1 if department == "QA" else 0,
    1 if department == "Support" else 0,
    1 if department == "Management" else 0,
    1 if department == "DevOps" else 0,
]

# Combine all inputs
input_data = np.array([age, tenure, salary, overtime_encoded] + department_encoded).reshape(1, -1)

# Predict attrition
if st.button("Predict"):
    probability = model.predict_proba(input_data)[0][1]
    prediction = 1 if probability >= threshold else 0
    st.write("Attrition Prediction:", "Yes" if prediction == 1 else "No")
    st.write(f"Likelihood of Attrition: {probability:.2f}")

