import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder

# -----------------------------
# Load Dataset
# -----------------------------
df = pd.read_csv("Titanic-Dataset.csv")

print("Dataset Information")
print(df.info())

print("\nStatistical Summary")
print(df.describe())

print("\nMissing Values Before Cleaning")
print(df.isnull().sum())

# -----------------------------
# Handle Missing Data
# -----------------------------
df["Age"] = df["Age"].fillna(df["Age"].median())
df["Fare"] = df["Fare"].fillna(df["Fare"].mean())
df["Embarked"] = df["Embarked"].fillna(df["Embarked"].mode()[0])

# Drop Cabin because of many missing values
df.drop(columns=["Cabin"], inplace=True)

print("\nMissing Values After Cleaning")
print(df.isnull().sum())

# -----------------------------
# Label Encoding
# -----------------------------
encoder = LabelEncoder()

df["Sex"] = encoder.fit_transform(df["Sex"])

# -----------------------------
# One Hot Encoding
# -----------------------------
df = pd.get_dummies(df, columns=["Embarked"])

print("\nFirst Five Rows")
print(df.head())

# -----------------------------
# Age Distribution
# -----------------------------
plt.figure(figsize=(8,5))
plt.hist(df["Age"], bins=20)

plt.title("Age Distribution")
plt.xlabel("Age")
plt.ylabel("Number of Passengers")

plt.show()
