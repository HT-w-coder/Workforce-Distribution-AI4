import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt

# --- Load models and encoders ---
leave_model = joblib.load("model.pkl")
role_model = joblib.load("role_classifier.pkl")
role_encoder = joblib.load("role_encoder.pkl")

# --- Load dataset ---
df = pd.read_csv("Employee.csv")

# --- Streamlit UI ---
st.title("📊 Workforce Distribution AI")
st.subheader("🔍 Predict Retention, Job Role, and Visualize Salary Trends")

# --- Input Fields ---
joining_year = st.number_input("Joining Year", min_value=2000, max_value=2025, value=2015)
payment_tier = st.selectbox("Payment Tier", [1, 2, 3])
age = st.slider("Age", min_value=18, max_value=60, value=30)
experience = st.slider("Experience in Current Domain", 0, 20, 3)
current_salary = st.number_input("Current Salary", value=40000)
expected_next_salary = st.number_input("Expected Salary Next Year", value=45000)
education = st.selectbox("Education", ["High School", "Bachelors", "Masters", "PhD"])
city = st.selectbox("City", sorted(df["City"].unique()))

# --- Encode Education ---
education_map = {"High School": 0, "Bachelors": 1, "Masters": 2, "PhD": 3}
education_level = education_map[education]

# --- Feature Vector for Leave Prediction ---
leave_input = pd.DataFrame([{
    "JoiningYear": joining_year,
    "PaymentTier": payment_tier,
    "Age": age,
    "ExperienceInCurrentDomain": experience,
    "CurrentSalary": current_salary,
    "ExpectedNextYearSalary": expected_next_salary,
    "EducationLevel": education_level,
    "AnnualWageGrowth": expected_next_salary - current_salary
}])

# --- Feature Vector for Role Prediction ---
role_input = pd.DataFrame([{
    "JoiningYear": joining_year,
    "PaymentTier": payment_tier,
    "Age": age,
    "ExperienceInCurrentDomain": experience,
    "CurrentSalary": current_salary,
    "EducationLevel": education_level
}])

# --- Predict when button clicked ---
if st.button("Predict Outcome"):
    # Leave prediction
    leave_result = leave_model.predict(leave_input)[0]
    leave_text = "Will Leave ❌" if leave_result == 1 else "Will Stay ✅"
    st.success(f"📌 Retention Prediction: **{leave_text}**")

    # Role prediction
    role_encoded = role_model.predict(role_input)[0]
    role_decoded = role_encoder.inverse_transform([role_encoded])[0]
    st.info(f"🧑‍💼 Predicted Job Role: **{role_decoded}**")

# --- 📈 Visualizations ---
st.subheader("📈 Salary Growth Trend")

# Average salary growth by experience
df["AnnualWageGrowth"] = df["ExpectedNextYearSalary"] - df["CurrentSalary"]
avg_growth = df.groupby("ExperienceInCurrentDomain")["AnnualWageGrowth"].mean()

fig, ax = plt.subplots()
avg_growth.plot(kind='line', marker='o', color='green', ax=ax)
ax.set_title("Average Annual Wage Growth by Experience")
ax.set_xlabel("Experience (Years)")
ax.set_ylabel("Annual Wage Growth")
st.pyplot(fig)

# Optional: Show filtered data by city
st.subheader("🏙️ City-wise Overview")
city_data = df[df["City"] == city]

if not city_data.empty:
    st.write(f"Showing data for **{city}**")
    st.dataframe(city_data[["Age", "CurrentSalary", "ExpectedNextYearSalary", "Education", "JobRole"]])
else:
    st.warning("No data found for this city.")
