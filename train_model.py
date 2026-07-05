import os
import pickle
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    ConfusionMatrixDisplay
)

# Create plots folder

os.makedirs("plots", exist_ok=True)

# Load Dataset

df = pd.read_csv("loan_data.csv")

# Features and Target

X = df[["Age", "Income", "LoanAmount", "CreditScore"]]
y = df["Approved"]

# Train-Test Split

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)

# Train Decision Tree Model

model = DecisionTreeClassifier(
    criterion="entropy",
    max_depth=6,
    random_state=42
)

model.fit(X_train, y_train)

# Prediction

y_pred = model.predict(X_test)

# Model Evaluation

accuracy = accuracy_score(y_test, y_pred)

print("=" * 40)
print("MODEL EVALUATION")
print("=" * 40)
print(f"Accuracy : {accuracy:.2%}")
print()

print("Classification Report")
print(classification_report(y_test, y_pred))


# Confusion Matrix

ConfusionMatrixDisplay.from_predictions(y_test, y_pred)

plt.title("Confusion Matrix")
plt.tight_layout()
plt.savefig("plots/confusion_matrix.png")
plt.close()

# Feature Importance

importance = model.feature_importances_

plt.figure(figsize=(7,5))
plt.bar(X.columns, importance)

plt.title("Feature Importance")
plt.xlabel("Features")
plt.ylabel("Importance")

plt.tight_layout()
plt.savefig("plots/feature_importance.png")
plt.close()

# Class Distribution

counts = df["Approved"].value_counts()

plt.figure(figsize=(6,6))

plt.pie(
    counts,
    labels=["Rejected", "Approved"],
    autopct="%1.1f%%",
    startangle=90,
    shadow=True
)

plt.title("Loan Approval Distribution")

plt.savefig("plots/class_distribution.png")
plt.close()

# Save Model

pickle.dump(model, open("model.pkl", "wb"))

print("Model saved successfully.")
print("Plots saved inside 'plots' folder.")