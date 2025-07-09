# train_model.py

import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error, accuracy_score

# Load dataset
df = pd.read_csv("Employee.csv")

# Encode Education
education_map = {"High School": 0, "Bachelors": 1, "Masters": 2, "PhD": 3}
df["EducationLevel"] = df["Education"].map(education_map)

# Create Annual Wage Growth
df["AnnualWageGrowth"] = df["ExpectedNextYearSalary"] - df["CurrentSalary"]

# Encode Job Role
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

# ====================
# Train LeaveOrNot Model
# ====================
features_leave = [
    "JoiningYear", "PaymentTier", "Age", "ExperienceInCurrentDomain",
    "CurrentSalary", "ExpectedNextYearSalary", "EducationLevel", "AnnualWageGrowth"
]
X_leave = df[features_leave]
y_leave = df["LeaveOrNot"]

X_train, X_test, y_train, y_test = train_test_split(X_leave, y_leave, test_size=0.2, random_state=42)

leave_model = RandomForestClassifier(random_state=42)
leave_model.fit(X_train, y_train)

joblib.dump(leave_model, "model.pkl")
print("✅ Leave model trained and saved as model.pkl")

# ====================
# Train Role Classifier
# ====================
features_role = [
    "JoiningYear", "PaymentTier", "Age", "ExperienceInCurrentDomain",
    "CurrentSalary", "EducationLevel"
]
X_role = df[features_role]
y_role = df["EncodedRole"]

X_train_r, X_test_r, y_train_r, y_test_r = train_test_split(X_role, y_role, test_size=0.2, random_state=42)

role_model = RandomForestClassifier(random_state=42)
role_model.fit(X_train_r, y_train_r)

joblib.dump(role_model, "role_classifier.pkl")
print("✅ Role classifier saved as role_classifier.pkl")

# ====================
# Salary Prediction Model (optional)
# ====================
X_salary = df[features_role]
y_salary = df["ExpectedNextYearSalary"]

salary_model = LinearRegression()
salary_model.fit(X_salary, y_salary)

joblib.dump(salary_model, "salary_predictor.pkl")
print("✅ Salary model saved as salary_predictor.pkl")
