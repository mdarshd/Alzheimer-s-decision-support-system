"""
llm/llm_interface.py

Wraps the Groq chat-completions API to turn raw model outputs (clinical
risk predictions, MRI-based stage predictions) into plain-language,
clinically-cautious explanations for the end user.

Requires the GROQ_API_KEY environment variable to be set (see
.env.example in the project root).
"""

import os
import re

from dotenv import load_dotenv
from groq import Groq

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")

if not GROQ_API_KEY:
    raise RuntimeError(
        "GROQ_API_KEY is not set. Copy .env.example to .env and add your "
        "own Groq API key, or export GROQ_API_KEY in your shell."
    )

client = Groq(api_key=GROQ_API_KEY)

MODEL_NAME = "llama-3.1-8b-instant"


def clean_llm_text(text: str) -> str:
    """Strip markdown formatting and normalize whitespace/bullets in an LLM response."""
    if not text:
        return ""

    text = re.sub(r"\*\*(.*?)\*\*", r"\1", text)
    text = re.sub(r"\*(.*?)\*", r"\1", text)
    text = re.sub(r"__(.*?)__", r"\1", text)
    text = re.sub(r"`(.*?)`", r"\1", text)

    text = text.replace("\u2022", "-")
    text = text.replace("\u2013", "-")

    text = re.sub(r"\n{3,}", "\n\n", text)

    return text.strip()


def _call_llm(system_prompt: str, user_prompt: str) -> str:
    """Send a system/user prompt pair to Groq and return the cleaned response text."""
    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.3,
            max_tokens=600,
        )

        raw_text = response.choices[0].message.content
        return clean_llm_text(raw_text)

    except Exception as e:
        return f"[LLM Error] {e}"


def explain_clinical_decision(clinical_result: dict, clinical_data: dict) -> str:
    """Generate a plain-language explanation for a clinical-questionnaire prediction."""
    gender_numeric = clinical_data.get("Gender", None)

    if gender_numeric == 0:
        gender_text = "Male"
    elif gender_numeric == 1:
        gender_text = "Female"
    else:
        gender_text = "Unknown"

    system_prompt = (
        "You are a clinical decision-support AI specializing in neurology and "
        "Alzheimer's disease. Explain reasoning conservatively and clearly."
    )

    user_prompt = f"""
Clinical Inputs:
- Age: {clinical_data.get('Age')}
- Gender: {gender_text}
- BMI: {clinical_data.get('BMI')}
- MMSE Score: {clinical_data.get('MMSE')}
- Behavioral Problems: {clinical_data.get('BehavioralProblems')}
- Disorientation: {clinical_data.get('Disorientation')}
- Difficulty Completing Tasks: {clinical_data.get('DifficultyCompletingTasks')}
- Forgetfulness: {clinical_data.get('Forgetfulness')}
- Head Injury: {clinical_data.get('HeadInjury')}
- Family History of Alzheimer's: {clinical_data.get('FamilyHistoryAlzheimers')}

Model Output:
Diagnosis: {clinical_result['Diagnosis']}
Message: {clinical_result['Message']}
Explain the clinical reasoning in a clearly formatted manner using
numbered points. Follow these formatting rules strictly:

- Each clinical indicator MUST be on its own numbered line
- Leave ONE blank line between each numbered point
- Write in complete sentences with a clinical tone
- Include Gender as a demographic risk modifier where relevant

Structure the explanation exactly like this:

1. Indicator name: explanation text.

(blank line)

2. Indicator name: explanation text.

(blank line)

Continue this format for all indicators.
Avoid definitive diagnosis language.

"""

    return _call_llm(system_prompt, user_prompt)


def explain_mri_stage(stage: str, clinical_data: dict) -> str:
    """Generate a plain-language explanation for an MRI-based stage prediction."""
    system_prompt = (
        "You are an AI neurologist assistant explaining MRI-based Alzheimer's staging "
        "in a cautious, clinical manner."
    )

    user_prompt = f"""
MRI Stage: {stage}

Clinical Context:
{clinical_data}

Explain:
1. Neurological meaning of this stage
2. Expected cognitive impact
3. Justification for recommendations
No treatment prescriptions.
"""

    return _call_llm(system_prompt, user_prompt)
