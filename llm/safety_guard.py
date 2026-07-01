"""
llm/safety_guard.py

Simple post-processing filter that catches overly definitive medical
language (e.g. a confirmed diagnosis or guaranteed cure) in LLM-generated
explanations and replaces it with a safer disclaimer.

Note: this filter is defined but not currently called from
llm/llm_interface.py. Wire safety_filter(...) around the return value of
_call_llm(...) if you want it enforced on every generated explanation.
"""

FORBIDDEN_PHRASES = [
    "you have alzheimer",
    "confirmed diagnosis",
    "guaranteed cure",
    "no chance of recovery",
]


def safety_filter(text: str) -> str:
    """Return a safe disclaimer if text contains overly definitive medical claims, else text unchanged."""
    lowered = text.lower()

    for phrase in FORBIDDEN_PHRASES:
        if phrase in lowered:
            return (
                "This explanation has been adjusted to avoid definitive "
                "medical claims. Please consult a medical professional."
            )

    return text
