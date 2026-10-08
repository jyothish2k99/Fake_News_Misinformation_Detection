import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)


# Load evaluation results
data = pd.read_csv("data/evaluation_results.csv")


# Actual and predicted labels
y_true = data["actual"]
y_pred = data["predicted"]


# Calculate metrics
accuracy = accuracy_score(y_true, y_pred)

precision = precision_score(
    y_true,
    y_pred,
    pos_label="FAKE"
)

recall = recall_score(
    y_true,
    y_pred,
    pos_label="FAKE"
)

f1 = f1_score(
    y_true,
    y_pred,
    pos_label="FAKE"
)


# Display results
print("========== MODEL EVALUATION ==========")

print(f"\nAccuracy : {accuracy:.4f}")
print(f"Precision : {precision:.4f}")
print(f"Recall : {recall:.4f}")
print(f"F1 Score : {f1:.4f}")


# Confusion matrix
cm = confusion_matrix(
    y_true,
    y_pred,
    labels=["FAKE", "REAL"]
)

print("\nConfusion Matrix:")
print(cm)


# Detailed classification report
print("\nClassification Report:")
print(
    classification_report(
        y_true,
        y_pred
    )
)