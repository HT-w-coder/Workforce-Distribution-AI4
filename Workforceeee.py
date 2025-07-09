import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score

# Load data
df = pd.read_csv(r"C:\Users\Admin\.vscode\backend\backend\Employee.csv")

# Encode categorical features
le = LabelEncoder()
for col in ['Education', 'City', 'Gender', 'EverBenched']:
    df[col] = le.fit_transform(df[col])

# Features and target
X = df.drop('LeaveOrNot', axis=1)
y = df['LeaveOrNot']

# Split into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Evaluation
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))
