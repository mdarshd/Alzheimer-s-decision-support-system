"""
recommendation/recommendation_engine.py

Maps an MRI-derived impairment stage to a static set of lifestyle
recommendations and prevention guidance, aligned with the clinical
stage names produced in app.py.
"""

STAGE_GUIDANCE = {
    "No Impairment": {
        "Stage": "No Impairment",
        "Recommendations": [
            "Maintain a balanced and nutritious diet",
            "Engage in regular physical exercise",
            "Practice cognitive activities like reading and puzzles",
            "Maintain proper sleep habits",
        ],
        "Prevention": [
            "Avoid smoking and excessive alcohol intake",
            "Control blood pressure, cholesterol, and diabetes",
            "Stay socially active",
        ],
    },
    "Mild Impairment": {
        "Stage": "Mild Impairment",
        "Recommendations": [
            "Perform daily memory training exercises",
            "Engage in light physical activities such as walking",
            "Follow a structured daily routine",
            "Reduce stress and anxiety",
        ],
        "Prevention": [
            "Monitor memory changes regularly",
            "Improve sleep quality",
            "Consult a doctor if symptoms progress",
        ],
    },
    "Moderate Impairment": {
        "Stage": "Moderate Impairment",
        "Recommendations": [
            "Consult a neurologist regularly",
            "Participate in cognitive rehabilitation therapy",
            "Assistance may be required for daily activities",
            "Engage in guided physical and mental exercises",
        ],
        "Prevention": [
            "Adhere to prescribed medications",
            "Family supervision is recommended",
            "Ensure home safety to avoid accidents",
        ],
    },
    "Severe Impairment": {
        "Stage": "Severe Impairment",
        "Recommendations": [
            "Continuous caregiver supervision is required",
            "Occupational and behavioral therapy",
            "Ensure a safe and familiar environment",
            "Regular medical monitoring",
        ],
        "Prevention": [
            "Prevent wandering and falls",
            "Ensure strict medication compliance",
            "Provide emotional and psychological support",
        ],
    },
}

DEFAULT_GUIDANCE = {
    "Stage": "Unknown",
    "Recommendations": ["Consult a medical professional immediately"],
    "Prevention": ["Immediate medical attention required"],
}


def get_recommendation(stage_name: str) -> dict:
    """Return lifestyle recommendations and prevention guidance for a given impairment stage."""
    return STAGE_GUIDANCE.get(stage_name, DEFAULT_GUIDANCE)
