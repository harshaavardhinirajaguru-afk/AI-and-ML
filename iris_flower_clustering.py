import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import LabelEncoder

# Load Dataset
df = pd.read_csv("Iris.csv")

print("Dataset Information")
print(df.info())

print("\nStatistical Summary")
print(df.describe())

# Features
X = df.iloc[:,1:5]

# Apply K-Means
kmeans = KMeans(n_clusters=3, random_state=42)
predicted_clusters = kmeans.fit_predict(X)

# Store predicted clusters
df["Predicted Cluster"] = predicted_clusters

# Encode actual labels
encoder = LabelEncoder()
df["True Label"] = encoder.fit_transform(df["Species"])

# Scatter Plot
plt.figure(figsize=(8,6))

plt.scatter(
    X["SepalLengthCm"],
    X["PetalLengthCm"],
    c=predicted_clusters
)

plt.xlabel("Sepal Length")
plt.ylabel("Petal Length")
plt.title("Iris Flower Clusters (K-Means)")

plt.show()

# Compare Results
comparison = df[["Species", "True Label", "Predicted Cluster"]]

print("\nComparison of True Labels and Predicted Clusters")
print(comparison.head(20))
