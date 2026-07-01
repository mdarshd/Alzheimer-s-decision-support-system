"""
prediction/csv_prediction.py

Runs the trained Random Forest model against a single patient's clinical
questionnaire answers and returns a diagnosis, a human-readable message,
and (when risk is detected) a list of contributing symptom factors.
"""

import os

import joblib
import pandas as pd

from prediction.clinical_feature_template import FEATURE_COLUMNS

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "prediction", "models", "csv_models.pkl")

model = joblib.load(MODEL_PATH)

SYMPTOM_FIELDS = [
    "Gender",
    "BehavioralProblems",
    "Disorientation",
    "DifficultyCompletingTasks",
    "Forgetfulness",
    "FamilyHistoryAlzheimers",
    "HeadInjury",
]


def _build_model_input(user_input: dict) -> dict:
    """Reindex the raw form input against the model's expected feature columns, filling gaps with 0."""
    full_input = {feature: 0 for feature in FEATURE_COLUMNS}
    for key, value in user_input.items():
        if key in full_input:
            full_input[key] = value
    return full_input


def _collect_risk_reasons(full_input: dict) -> list:
    """Collect the human-readable list of contributing risk factors from a filled input."""
    reasons = []

    for symptom in SYMPTOM_FIELDS:
        if full_input.get(symptom) == 1:
            if symptom == "Gender":
                reasons.append("Gender-related risk factor (Female)")
            else:
                reasons.append(symptom)

    if full_input["Age"] > 60:
        reasons.append("Advanced Age")

    if full_input["MMSE"] < 24:
        reasons.append("Low MMSE Score")

    return reasons


def predict_from_clinical_form(user_input: dict) -> dict:
    """Predict Alzheimer's risk from a clinical questionnaire and return a diagnosis summary."""
    full_input = _build_model_input(user_input)

    df = pd.DataFrame([full_input], columns=FEATURE_COLUMNS)
    prediction = model.predict(df)[0]

    if prediction == 0:
        return {
            "Diagnosis": "No Alzheimer's Detected",
            "Message": "No significant clinical indicators detected.",
            "NextStep": "MRI scan not required.",
        }

    return {
        "Diagnosis": "Alzheimer's Risk Detected",
        "Message": "Clinical indicators suggest elevated Alzheimer's risk.",
        "Reasons": _collect_risk_reasons(full_input),
        "NextStep": "Proceed with MRI analysis.",
    }
