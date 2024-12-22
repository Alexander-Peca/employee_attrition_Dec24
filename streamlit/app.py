import streamlit as st
import joblib
import numpy as np

# Load the saved logistic regression model and threshold
model_path = r"C:\Users\alexa\OneDrive\AlexPerez\Dokumente\2 Soros\Data Science\Projects\employee_attrition_Dec24\models\logistic_regression_model.pkl"
threshold_path = r"C:\Users\alexa\OneDrive\AlexPerez\Dokumente\2 Soros\Data Science\Projects\employee_attrition_Dec24\models\logistic_regression_threshold.pkl"

model = joblib.load(model_path)
threshold = joblib.load(threshold_path)

# Streamlit app layout
st.title("Employee Attrition Predictor")
st.write("Enter employee details to predict the likelihood of attrition.")

# Collect input from users
gender = st.selectbox("Gender", ["Male", "Female"])
age = st.slider("Age", 20, 60, 30)
tenure = st.slider("Tenure (Years)", 0, 40, 5)
salary = st.number_input("Salary (USD)", min_value=50000, max_value=150000, value=100000)
overtime = st.selectbox("Overtime", ["Yes", "No"])
department = st.selectbox("Department", ["Development", "QA", "Support", "Management", "DevOps"])

# Q12+ Questions
st.write("Answer the following questions on a scale from 1 (Strongly Disagree) to 5 (Strongly Agree):")
q1 = st.slider("Q1: I know what is expected of me at work.", 1, 5, 3)
q2 = st.slider("Q2: I have the materials and equipment I need to do my work right.", 1, 5, 3)
q3 = st.slider("Q3: At work, I have the opportunity to do what I do best every day.", 1, 5, 3)
q4 = st.slider("Q4: In the last seven days, I have received recognition or praise for doing good work.", 1, 5, 3)
q5 = st.slider("Q5: My supervisor, or someone at work, seems to care about me as a person.", 1, 5, 3)
q6 = st.slider("Q6: There is someone at work who encourages my development.", 1, 5, 3)
q7 = st.slider("Q7: At work, my opinions seem to count.", 1, 5, 3)
q8 = st.slider("Q8: The mission or purpose of my company makes me feel my job is important.", 1, 5, 3)
q9 = st.slider("Q9: My associates or fellow employees are committed to doing quality work.", 1, 5, 3)
q10 = st.slider("Q10: I have a best friend at work.", 1, 5, 3)
q11 = st.slider("Q11: In the last six months, someone at work has talked to me about my progress.", 1, 5, 3)
q12 = st.slider("Q12: This last year, I have had opportunities at work to learn and grow.", 1, 5, 3)
q13 = st.slider("Q13: At work, I am treated with respect.", 1, 5, 3)
q14 = st.slider("Q14: My organization cares about my overall wellbeing.", 1, 5, 3)
q15 = st.slider("Q15: I have received meaningful feedback in the last week.", 1, 5, 3)
q16 = st.slider("Q16: My organization always delivers on the promise we make to customers.", 1, 5, 3)

# Encode categorical variables
gender_encoded = 1 if gender == "Male" else 0  # Binary encoding for gender
overtime_encoded = 1 if overtime == "Yes" else 0
department_encoded = [0] * 5  # Initialize a zero vector for one-hot encoding
departments = ["Development", "QA", "Support", "Management", "DevOps"]
if department in departments:
    department_encoded[departments.index(department)] = 1

# Combine all features
input_data = np.array([gender_encoded, age, tenure, salary, overtime_encoded, *department_encoded,
                       q1, q2, q3, q4, q5, q6, q7, q8, q9, q10, q11, q12, q13, q14, q15, q16]).reshape(1, -1)

# Predict attrition
if st.button("Predict"):
    if input_data.shape[1] == 26:
        prob = model.predict_proba(input_data)[0][1]  # Probability of attrition (class 1)
        prediction = 1 if prob >= threshold else 0
        st.write(f"Attrition Prediction: {'Yes' if prediction == 1 else 'No'}")
        st.write(f"Probability of Attrition: {prob * 100:.2f}%")
    else:
        st.error(f"Error: Expected 26 features, but got {input_data.shape[1]}. Please verify input feature processing.")
