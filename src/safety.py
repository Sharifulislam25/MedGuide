"""
safety.py

Two responsibilities, both deliberately simple and rule-based rather
than model-based (see Sections 18 and 19 of the project spec -- this is
explicitly NOT meant to be a symptom checker or emergency-prediction
model, just a conservative first filter):

1. Emergency detection: catch questions that describe a possible acute
   medical emergency, and return urgent guidance instead of letting a
   normal RAG-based answer proceed.

2. Diagnosis-language guard: scan any generated response text for
   phrasing that reads as a definitive diagnosis ("you have diabetes").
   This is a last-resort safety net in case a future change to
   response.py or medical_rules.py accidentally introduces language
   like that -- the templates in response.py are already written to
   avoid it, but this catches it regardless of where it came from.
"""

import re
from typing import Optional


# --- Emergency detection ---

# Deliberately narrow, literal phrases describing symptoms commonly
# associated with a medical emergency. This list exists only to catch
# a request before it gets a calm, textbook-style answer when what it
# actually needs is "get help right now" -- it is NOT trying to
# diagnose or assess likelihood of an emergency.
EMERGENCY_PHRASES = [
    "chest pain", "crushing chest pain", "can't breathe", "cannot breathe",
    "difficulty breathing", "trouble breathing", "shortness of breath",
    "heart attack", "having a heart attack", "stroke", "face drooping",
    "slurred speech", "sudden numbness", "sudden weakness",
    "severe bleeding", "won't stop bleeding", "unconscious", "unresponsive",
    "overdose", "took too many pills", "severe allergic reaction",
    "anaphylaxis", "throat closing", "suicidal", "want to kill myself",
    "want to die", "end my life", "seizure",
]

EMERGENCY_RESPONSE = (
    "**This sounds like it could be a medical emergency.**\n\n"
    "MedGuide is an educational tool and cannot help in an emergency. "
    "Please contact emergency services immediately:\n\n"
    "- In the US, call 911 (or your local emergency number).\n"
    "- If this involves thoughts of suicide or self-harm, you can also "
    "call or text 988 (Suicide & Crisis Lifeline, US) at any time.\n"
    "- If someone is unconscious, not breathing, or bleeding heavily, "
    "call emergency services now rather than waiting.\n\n"
    "Do not rely on this app for emergency guidance."
)


def check_for_emergency(text: str) -> bool:
    """
    Check whether a piece of text (typically the user's question)
    contains a phrase associated with a possible medical emergency.

    Deliberately simple substring matching -- conservative and easy to
    audit, per the project spec's instruction not to build a prediction
    model for this.
    """
    lowered = text.lower()
    return any(phrase in lowered for phrase in EMERGENCY_PHRASES)


def get_emergency_response() -> str:
    """Return the fixed emergency guidance message."""
    return EMERGENCY_RESPONSE


# --- Diagnosis-language guard ---

# Phrase patterns that read as a definitive diagnosis rather than
# cautious, educational language (Section 18: "Avoid: You have
# diabetes"). Checked against every generated response as a safety net.
DIAGNOSTIC_PATTERNS = [
    r"\byou have\b",
    r"\byou('| a)re diabetic\b",
    r"\byou('| a)re anemic\b",
    r"\bthis (means|confirms) you have\b",
    r"\byou definitely have\b",
    r"\bthis is (a diagnosis of|diabetes|cancer)\b",
]

SAFE_FALLBACK_MESSAGE = (
    "I can't provide a specific interpretation of this result safely. "
    "Please review this result with a qualified healthcare professional, "
    "who can consider your full medical history and other factors."
)


def contains_diagnostic_language(text: str) -> Optional[str]:
    """
    Check generated response text for phrasing that reads as a
    definitive diagnosis.

    Returns
    -------
    Optional[str]
        The matched pattern (for debugging/logging) if diagnostic
        language was found, else None.
    """
    lowered = text.lower()
    for pattern in DIAGNOSTIC_PATTERNS:
        if re.search(pattern, lowered):
            return pattern
    return None


def enforce_safe_language(text: str) -> str:
    """
    Last line of defense before showing a generated answer: if it's
    somehow found to contain definitive-diagnosis language, discard it
    and return a safe fallback instead of displaying something that
    violates the medical safety layer.
    """
    if contains_diagnostic_language(text):
        return SAFE_FALLBACK_MESSAGE
    return text
