# main.py

import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt

# Load models
leave_model = joblib.load("model.pkl")
salary_model = joblib.load("salary_predictor.pkl")
role_model = joblib.load("role_classifier.pkl")
role_encoder = joblib.load("role_encoder.pkl")

# Load dataset for visualization
df = pd.read_csv("Employee.csv")
df["AnnualWageGrowth"] = df["ExpectedNextYearSalary"] - df["CurrentSalary"]

# UI
st.set_page_config(page_title="Workforce Distribution AI", layout="centered")
st.title("📊 Workforce Distribution AI")
st.markdown("Predict **LeaveOrNot**, estimate salary growth, and classify job role.")

# Inputs
with st.form("prediction_form"):
    col1, col2 = st.columns(2)
    with col1:
        joining_year = st.number_input("Joining Year", 2000, 2025, 2015)
        payment_tier = st.selectbox("Payment Tier", [1, 2, 3])
        age = st.slider("Age", 18, 60, 30)
        education = st.selectbox("Education", ["High School", "Bachelors", "Masters", "PhD"])
    with col2:
        experience = st.slider("Experience in Current Domain", 0, 20, 3)
        current_salary = st.number_input("Current Salary", 10000, 200000, 40000)
        expected_next_year = st.number_input("Expected Salary Next Year", 10000, 300000, 50000)

    submit = st.form_submit_button("🔍 Predict")

# On submit
if submit:
    education_map = {"High School": 0, "Bachelors": 1, "Masters": 2, "PhD": 3}
    input_df = pd.DataFrame([{
        "JoiningYear": joining_year,
        "PaymentTier": payment_tier,
        "Age": age,
        "ExperienceInCurrentDomain": experience,
        "CurrentSalary": current_salary,
        "ExpectedNextYearSalary": expected_next_year,
        "EducationLevel": education_map[education],
        "AnnualWageGrowth": expected_next_year - current_salary
    }])

    # Prediction
    leave_pred = leave_model.predict(input_df)[0]
    salary_pred = salary_model.predict(input_df)[0]
    role_pred_encoded = role_model.predict(input_df)[0]
    role_decoded = role_encoder.inverse_transform([role_pred_encoded])[0]

    st.success("📌 **Results**:")
    st.markdown(f"- **Will Leave?**: {'❌ Yes' if leave_pred == 1 else '✅ No'}")
    st.markdown(f"- **Predicted Salary Next Year**: ₹{int(salary_pred):,}")
    st.markdown(f"- **Predicted Role**: {role_decoded}")

# Salary growth chart
st.subheader("📈 Annual Wage Growth by Experience")
avg_growth = df.groupby("ExperienceInCurrentDomain")["AnnualWageGrowth"].mean()
fig, ax = plt.subplots()
avg_growth.plot(kind='line', marker='o', ax=ax)
ax.set_title("Avg. Annual Wage Growth vs Experience")
ax.set_xlabel("Experience (Years)")
ax.set_ylabel("Wage Growth")
st.pyplot(fig)
