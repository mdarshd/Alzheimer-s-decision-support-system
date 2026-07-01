"""
prediction/models/performance.py

Standalone evaluation script for the clinical Random Forest model.
Loads the trained model and the clinical CSV dataset, re-creates the
train/test split used at training time, and reports accuracy, precision,
recall, F1, ROC-AUC, a classification report, a confusion matrix, and an
ROC curve for the held-out test set.

Run with: python -m prediction.models.performance
"""

import os

import joblib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_PATH = os.path.join(BASE_DIR, "data", "csv", "alzheimers_disease_data.csv")
MODEL_PATH = os.path.join(BASE_DIR, "prediction", "models", "csv_models.pkl")

DROP_COLS = ["PatientID", "DoctorInCharge"]
TARGET_COL = "Diagnosis"


def load_test_split():
    """Load the clinical CSV dataset and recreate the same train/test split used at training time."""
    data = pd.read_csv(DATA_PATH)
    print("\nDataset Information")
    print("Total samples in CSV:", len(data))

    data = data.drop(columns=DROP_COLS)

    X = data.drop(TARGET_COL, axis=1)
    y = data[TARGET_COL]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print("Training samples:", len(X_train))
    print("Testing samples :", len(X_test))

    return X_test, y_test


def evaluate(model, X_test, y_test):
    """Print classification metrics and show confusion matrix / ROC curve plots."""
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    print("\nRandom Forest Performance Metrics (Test Set)\n")
    print(f"Accuracy  : {accuracy_score(y_test, y_pred):.4f}")
    print(f"Precision : {precision_score(y_test, y_pred):.4f}")
    print(f"Recall    : {recall_score(y_test, y_pred):.4f}")
    print(f"F1-score  : {f1_score(y_test, y_pred):.4f}")
    print(f"ROC-AUC   : {roc_auc_score(y_test, y_prob):.4f}")

    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Random Forest Confusion Matrix (Test Set)")
    plt.tight_layout()
    plt.show()

    fpr, tpr, _ = roc_curve(y_test, y_prob)
    roc_auc = roc_auc_score(y_test, y_prob)
    plt.figure()
    plt.plot(fpr, tpr, label=f"ROC AUC = {roc_auc:.3f}")
    plt.plot([0, 1], [0, 1], linestyle="--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Random Forest ROC Curve (Test Set)")
    plt.legend()
    plt.tight_layout()
    plt.show()


def main():
    """Load the trained model and dataset, then run and display the evaluation."""
    model = joblib.load(MODEL_PATH)
    X_test, y_test = load_test_split()
    evaluate(model, X_test, y_test)


if __name__ == "__main__":
    main()
