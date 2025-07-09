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

# ===========================
# Preprocessing
# ===========================
# Encode Education
education_map = {"High School": 0, "Bachelors": 1, "Masters": 2, "PhD": 3}
df["EducationLevel"] = df["Education"].map(education_map)

# Assign JobRole
def assign_role(row):
    if row['ExperienceInCurrentDomain'] >= 7 and row['CurrentSalary'] > 70000:
        return "Senior Engineer"
    elif row['ExperienceInCurrentDomain'] >= 3:
        return "Mid-Level Developer"
    else:
        return "Junior Associate"

df["JobRole"] = df.apply(assign_role, axis=1)

# Encode JobRole
role_encoder = LabelEncoder()
df["EncodedRole"] = role_encoder.fit_transform(df["JobRole"])
joblib.dump(role_encoder, "role_encoder.pkl")

# Features used in salary and role prediction
features = ["JoiningYear", "PaymentTier", "Age", "ExperienceInCurrentDomain", "CurrentSalary", "EducationLevel"]

# ===========================
# Train Salary Predictor
# ===========================
X_salary = df[features]
y_salary = df["ExpectedNextYearSalary"]

salary_imputer = SimpleImputer(strategy='mean')
X_salary_imputed = salary_imputer.fit_transform(X_salary)
joblib.dump(salary_imputer, "salary_imputer.pkl")

X_train, X_test, y_train, y_test = train_test_split(X_salary_imputed, y_salary, test_size=0.2, random_state=42)

salary_model = LinearRegression()
salary_model.fit(X_train, y_train)
salary_pred = salary_model.predict(X_test)
print("💰 Salary RMSE:", mean_squared_error(y_test, salary_pred, squared=False))
joblib.dump(salary_model, "salary_predictor.pkl")

# ===========================
# Train Role Classifier
# ===========================
y_role = df["EncodedRole"]
X_role_imputed = salary_imputer.transform(df[features])
X_train_r, X_test_r, y_train_r, y_test_r = train_test_split(X_role_imputed, y_role, test_size=0.2, random_state=42)

role_model = RandomForestClassifier(n_estimators=100, random_state=42)
role_model.fit(X_train_r, y_train_r)
role_pred = role_model.predict(X_test_r)
print("🧑‍💼 Role Classifier Accuracy:", accuracy_score(y_test_r, role_pred))
joblib.dump(role_model, "role_classifier.pkl")

# ===========================
# Train Leave Model
# ===========================
df["AnnualWageGrowth"] = df["ExpectedNextYearSalary"] - df["CurrentSalary"]

leave_features = features + ["City"]
leave_X_raw = pd.get_dummies(df[leave_features], columns=["City"], drop_first=True)
leave_y = df["LeaveOrNot"]

leave_imputer = SimpleImputer(strategy="mean")
leave_X = leave_imputer.fit_transform(leave_X_raw)
joblib.dump(leave_imputer, "leave_imputer.pkl")
joblib.dump(leave_X_raw.columns.tolist(), "leave_columns.pkl")

X_train_l, X_test_l, y_train_l, y_test_l = train_test_split(leave_X, leave_y, test_size=0.2, random_state=42)

leave_model = RandomForestClassifier(n_estimators=100, random_state=42)
leave_model.fit(X_train_l, y_train_l)
leave_pred = leave_model.predict(X_test_l)
print("❌ Leave Model Accuracy:", accuracy_score(y_test_l, leave_pred))
joblib.dump(leave_model, "leave_model.pkl")

print("✅ All models and preprocessors saved.")
