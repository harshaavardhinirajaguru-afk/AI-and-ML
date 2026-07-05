import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report

# -----------------------------
# Load Dataset
# -----------------------------
df = pd.read_csv("loan_prediction.csv")

print("First 5 Rows")
print(df.head())

print("\nDataset Information")
print(df.info())

print("\nMissing Values")
print(df.isnull().sum())

# -----------------------------
# Handle Missing Values
# -----------------------------

# Fill numerical missing values with median
num_cols = ["LoanAmount", "Loan_Amount_Term", "ApplicantIncome", "CoapplicantIncome"]

for col in num_cols:
    if col in df.columns:
        df[col] = df[col].fillna(df[col].median())

# Fill categorical missing values with mode
cat_cols = [
    "Gender",
    "Married",
    "Dependents",
    "Self_Employed",
    "Credit_History",
    "LoanAmount",
    "Loan_Amount_Term",
    "Property_Area"
]

for col in cat_cols:
    if col in df.columns:
        df[col] = df[col].fillna(df[col].mode()[0])

# -----------------------------
# Encode Categorical Variables
# -----------------------------
encoder = LabelEncoder()

for column in df.columns:
    if df[column].dtype == object:
        df[column] = encoder.fit_transform(df[column])

# -----------------------------
# Features and Target
# -----------------------------
X = df.drop(["Loan_ID", "Loan_Status"], axis=1)
y = df["Loan_Status"]

# -----------------------------
# Split Dataset
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# -----------------------------
# Train Random Forest Model
# -----------------------------
model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

model.fit(X_train, y_train)

# -----------------------------
# Predictions
# -----------------------------
y_pred = model.predict(X_test)

# -----------------------------
# Evaluation
# -----------------------------
print("\nAccuracy Score")
print(accuracy_score(y_test, y_pred))

print("\nConfusion Matrix")
print(confusion_matrix(y_test, y_pred))

print("\nClassification Report")
print(classification_report(y_test, y_pred))

# -----------------------------
# Predict New Applicant
# -----------------------------
sample = X.iloc[[0]]

prediction = model.predict(sample)

if prediction[0] == 1:
    print("\nLoan Prediction: Approved")
else:
    print("\nLoan Prediction: Not Approved")
