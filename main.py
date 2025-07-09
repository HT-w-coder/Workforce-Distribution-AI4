# main.py

import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt

st.set_page_config(page_title="Workforce Distribution AI", layout="wide")

# Load Models and Tools
leave_model = joblib.load("model.pkl")
salary_model = joblib.load("salary_predictor.pkl")
role_model = joblib.load("role_classifier.pkl")
role_encoder = joblib.load("role_encoder.pkl")
imputer = joblib.load("imputer.pkl")
df = pd.read_csv("Employee.csv")

st.title("📊 Workforce Distribution AI")
st.subheader("🔍 Predict Retention, Salary Growth & Role Classification")

# Input Fields
joining_year = st.number_input("Joining Year", min_value=2000, max_value=2025, value=2015)
payment_tier = st.selectbox("Payment Tier", [1, 2, 3])
age = st.slider("Age", 18, 60, 30)
experience = st.slider("Experience in Current Domain", 0, 20, 3)
current_salary = st.number_input("Current Salary", value=40000)
expected_salary = st.number_input("Expected Salary Next Year", value=42000)
education = st.selectbox("Education", ["High School", "Bachelors", "Masters", "PhD"])

# Education Mapping
education_map = {"High School": 0, "Bachelors": 1, "Masters": 2, "PhD": 3}
education_level = education_map[education]

# Prepare Input
annual_growth = expected_salary - current_salary

input_data = pd.DataFrame([{
    "JoiningYear": joining_year,
    "PaymentTier": payment_tier,
    "Age": age,
    "ExperienceInCurrentDomain": experience,
    "CurrentSalary": current_salary,
    "ExpectedNextYearSalary": expected_salary,
    "EducationLevel": education_level,
    "AnnualWageGrowth": annual_growth
}])

# Impute Missing Values
input_imputed = imputer.transform(input_data)

# Predictions
if st.button("Predict Retention & Role"):
    # LeaveOrNot
    leave_pred = leave_model.predict(input_imputed)[0]
    leave_result = "❌ Will Leave" if leave_pred == 1 else "✅ Will Stay"
    st.success(f"Employee Retention: **{leave_result}**")

    # Role Prediction
    role_encoded = role_model.predict(input_imputed)[0]
    role_name = role_encoder.inverse_transform([role_encoded])[0]
    st.info(f"Predicted Job Role: **{role_name}**")

    # Salary Prediction
    predicted_salary = salary_model.predict(input_imputed)[0]
    st.write(f"💰 Predicted Next Year Salary: ₹ **{int(predicted_salary):,}**")

# Salary Growth Visualization
st.subheader("📈 Average Wage Growth by Experience")
df["AnnualWageGrowth"] = df["ExpectedNextYearSalary"] - df["CurrentSalary"]
avg_growth = df.groupby("ExperienceInCurrentDomain")["AnnualWageGrowth"].mean()

fig, ax = plt.subplots()
avg_growth.plot(kind='line', marker='o', ax=ax, color='green')
ax.set_title("Avg Annual Wage Growth vs. Experience")
ax.set_xlabel("Experience (Years)")
ax.set_ylabel("Annual Wage Growth")
st.pyplot(fig)