import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt

# Load models
salary_model = joblib.load("salary_predictor.pkl")
role_model = joblib.load("role_classifier.pkl")
leave_model = joblib.load("leave_model.pkl")

# Load encoders and imputers
role_encoder = joblib.load("role_encoder.pkl")
salary_imputer = joblib.load("salary_imputer.pkl")
leave_imputer = joblib.load("leave_imputer.pkl")
leave_columns = joblib.load("leave_columns.pkl")

# Load dataset
df = pd.read_csv("Employee.csv")

st.title("📊 Workforce Distribution AI Dashboard")
st.subheader("🧠 Predict Job Role, Salary & Retention")

# Form inputs
with st.form("prediction_form"):
    col1, col2 = st.columns(2)
    with col1:
        joining_year = st.number_input("Joining Year", 2000, 2025, 2015)
        payment_tier = st.selectbox("Payment Tier", [1, 2, 3])
        age = st.slider("Age", 18, 60, 30)
        city = st.selectbox("City", sorted(df["City"].dropna().unique()))
    with col2:
        experience = st.slider("Experience in Current Domain", 0, 20, 3)
        current_salary = st.number_input("Current Salary", 10000, 200000, 40000, step=1000)
        education = st.selectbox("Education", ["High School", "Bachelors", "Masters", "PhD"])

    submit = st.form_submit_button("🔍 Predict")

if submit:
    education_map = {"High School": 0, "Bachelors": 1, "Masters": 2, "PhD": 3}
    education_level = education_map[education]

    input_dict = {
        "JoiningYear": joining_year,
        "PaymentTier": payment_tier,
        "Age": age,
        "ExperienceInCurrentDomain": experience,
        "CurrentSalary": current_salary,
        "EducationLevel": education_level,
        "City": city,
    }

    # For salary & role prediction
    salary_input = pd.DataFrame([{
        "JoiningYear": joining_year,
        "PaymentTier": payment_tier,
        "Age": age,
        "ExperienceInCurrentDomain": experience,
        "CurrentSalary": current_salary,
        "EducationLevel": education_level,
    }])
    salary_input_imputed = salary_imputer.transform(salary_input)

    # Predict salary
    predicted_salary = salary_model.predict(salary_input_imputed)[0]
    salary_growth = predicted_salary - current_salary

    # Predict role
    role_encoded = role_model.predict(salary_input_imputed)[0]
    role_label = role_encoder.inverse_transform([role_encoded])[0]

    # For leave prediction (needs one-hot + impute)
    leave_input = pd.DataFrame([input_dict])
    leave_input_encoded = pd.get_dummies(leave_input, columns=["City"], drop_first=True)
    for col in leave_columns:
        if col not in leave_input_encoded.columns:
            leave_input_encoded[col] = 0  # Add missing dummy columns
    leave_input_encoded = leave_input_encoded[leave_columns]  # Order columns
    leave_input_imputed = leave_imputer.transform(leave_input_encoded)
    leave_pred = leave_model.predict(leave_input_imputed)[0]
    leave_text = "Will Leave ❌" if leave_pred == 1 else "Will Stay ✅"

    # Display results
    st.markdown(f"### 💼 Predicted Role: **{role_label}**")
    st.markdown(f"### 💰 Expected Salary Next Year: ₹ **{int(predicted_salary):,}**")
    st.markdown(f"### 📈 Wage Growth: ₹ **{int(salary_growth):,}**")
    st.markdown(f"### 🔁 Retention: **{leave_text}**")

# Visualization
st.subheader("📉 Wage Growth vs. Experience")
if "AnnualWageGrowth" not in df.columns:
    df["AnnualWageGrowth"] = df["ExpectedNextYearSalary"] - df["CurrentSalary"]

avg_growth = df.groupby("ExperienceInCurrentDomain")["AnnualWageGrowth"].mean()
fig, ax = plt.subplots()
avg_growth.plot(kind="line", marker="o", color="green", ax=ax)
ax.set_title("Average Annual Wage Growth by Experience")
ax.set_xlabel("Experience (Years)")
ax.set_ylabel("Wage Growth (₹)")
st.pyplot(fig)
