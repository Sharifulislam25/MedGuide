"""
medical_rules.py

Defines what MedGuide knows about a handful of common lab tests: what
units they're usually reported in, what a "normal" range typically
looks like, and which knowledge-base file has more detail. Also
contains simple pattern-matching logic to pull an actual reported
value for a test out of extracted document text.

This is deliberately a small, explicit set of rules for common tests --
not a general medical inference system, and not exhaustive. See
Section 18 of the project spec (Medical Safety Layer) for why staying
narrow and explicit here matters.
"""

import re
from dataclasses import dataclass
from typing import List, Optional


@dataclass
class TestRule:
    """Everything needed to recognize and interpret one lab test."""
    key: str                 # internal id, e.g. "glucose"
    display_name: str        # human-readable, e.g. "Fasting Glucose"
    aliases: List[str]       # phrases to look for in text (lowercase)
    unit: str
    normal_low: float
    normal_high: float
    knowledge_source: str    # which knowledge/*.txt file has more detail


# Ranges here match the same "commonly used, can vary by lab" ranges
# described in the corresponding knowledge/*.txt file from Phase 10.
TEST_RULES: List[TestRule] = [
    TestRule("glucose", "Fasting Glucose",
             ["fasting glucose", "blood glucose", "glucose"],
             "mg/dL", 70, 99, "glucose.txt"),
    TestRule("hemoglobin", "Hemoglobin",
             ["hemoglobin", "hgb"],
             "g/dL", 12.0, 17.5, "hemoglobin.txt"),
    TestRule("total_cholesterol", "Total Cholesterol",
             ["total cholesterol"],
             "mg/dL", 0, 200, "cholesterol.txt"),
    TestRule("ldl", "LDL Cholesterol",
             ["ldl cholesterol", "ldl"],
             "mg/dL", 0, 100, "cholesterol.txt"),
    TestRule("hdl", "HDL Cholesterol",
             ["hdl cholesterol", "hdl"],
             "mg/dL", 40, 999, "cholesterol.txt"),
    TestRule("alt", "ALT",
             ["alt"],
             "U/L", 7, 56, "liver_function.txt"),
    TestRule("ast", "AST",
             ["ast"],
             "U/L", 8, 48, "liver_function.txt"),
    TestRule("creatinine", "Creatinine",
             ["creatinine"],
             "mg/dL", 0.6, 1.3, "kidney_function.txt"),
    TestRule("tsh", "TSH",
             ["tsh"],
             "mIU/L", 0.4, 4.0, "thyroid.txt"),
]


def find_test_value(text: str, rule: TestRule) -> Optional[float]:
    """
    Look for one test's reported numeric value in a piece of text, e.g.
    find 145 in "Fasting Glucose     145 mg/dL".

    Tries each of the rule's aliases in turn, matching "<alias> ... <number>"
    with up to 15 characters of separator in between (spaces, colons,
    stray punctuation) -- OCR'd text especially tends to be messy here.

    Returns
    -------
    Optional[float]
        The first matching number found, or None if the test name isn't
        mentioned or no number follows it closely enough to be confident.
    """
    lowered = text.lower()
    for alias in rule.aliases:
        pattern = re.escape(alias) + r"[\s:]{1,15}(\d+\.?\d*)"
        match = re.search(pattern, lowered)
        if match:
            try:
                return float(match.group(1))
            except ValueError:
                continue
    return None


def interpret_value(rule: TestRule, value: float) -> str:
    """
    Compare a value against a rule's normal range.

    Returns
    -------
    str
        One of "above", "below", or "within" -- deliberately just a
        position relative to the range, not a diagnostic label.
    """
    if value > rule.normal_high:
        return "above"
    elif value < rule.normal_low:
        return "below"
    else:
        return "within"


def find_matching_rule(text: str) -> Optional[TestRule]:
    """Check whether a piece of text (e.g. a user's question) mentions a known test."""
    lowered = text.lower()
    for rule in TEST_RULES:
        for alias in rule.aliases:
            if alias in lowered:
                return rule
    return None
