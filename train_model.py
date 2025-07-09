# train_model.py

import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error, accuracy_score
from sklearn.impute import SimpleImputer

# Load dataset
df = pd.read_csv("Employee.csv")

# Encode Education
education_map = {"High School": 0, "Bachelors": 1, "Masters": 2, "PhD": 3}
df["EducationLevel"] = df["Education"].map(education_map)

# Compute Annual Wage Growth
df["AnnualWageGrowth"] = df["ExpectedNextYearSalary"] - df["CurrentSalary"]

# Assign mock Job Roles based on logic
def assign_role(row):
    if row['ExperienceInCurrentDomain'] >= 7 and row['CurrentSalary'] > 70000:
        return "Senior Engineer"
    elif row['ExperienceInCurrentDomain'] >= 3:
        return "Mid-Level Developer"
    else:
        return "Junior Associate"

df["JobRole"] = df.apply(assign_role, axis=1)
role_encoder = LabelEncoder()
df["EncodedRole"] = role_encoder.fit_transform(df["JobRole"])
joblib.dump(role_encoder, "role_encoder.pkl")

# Features used for all models
features = [
    "JoiningYear", "PaymentTier", "Age", "ExperienceInCurrentDomain",
    "CurrentSalary", "ExpectedNextYearSalary", "EducationLevel", "AnnualWageGrowth"
]

# Impute missing values
imputer = SimpleImputer(strategy="mean")
df[features] = imputer.fit_transform(df[features])
joblib.dump(imputer, "imputer.pkl")

# 1. LeaveOrNot Model
X_leave = df[features]
y_leave = df["LeaveOrNot"]
X_train, X_test, y_train, y_test = train_test_split(X_leave, y_leave, test_size=0.2, random_state=42)

leave_model = RandomForestClassifier(random_state=42)
leave_model.fit(X_train, y_train)
joblib.dump(leave_model, "model.pkl")
print("✅ LeaveOrNot model saved as model.pkl")

# 2. Job Role Classifier
X_role = df[features]
y_role = df["EncodedRole"]
role_model = RandomForestClassifier(random_state=42)
role_model.fit(X_role, y_role)
joblib.dump(role_model, "role_classifier.pkl")
print("✅ Role classifier saved as role_classifier.pkl")

# 3. Salary Prediction Model
X_salary = df[features]
y_salary = df["ExpectedNextYearSalary"]
salary_model = LinearRegression()
salary_model.fit(X_salary, y_salary)
joblib.dump(salary_model, "salary_predictor.pkl")
print("✅ Salary predictor saved as salary_predictor.pkl")
