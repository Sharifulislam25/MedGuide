"""
response.py

The V1 response engine: turns a RagContext (Phase 11-13's retrieved
chunks) into an actual written answer, using plain string templates and
the rules in medical_rules.py. No LLM is involved -- see Section 17 of
the project spec for why (a template can't hallucinate a fake
reference range or invent a diagnosis; a small local LLM might).

Overall approach for a question about a specific, known test:
    1. Check whether the question mentions a test we have a rule for.
    2. If so, look for that test's actual reported value across the
       retrieved user-document chunks.
    3. Compare it to the normal range and describe it in cautious,
       non-diagnostic language ("above/below/within a commonly used
       range" -- never "you have X").
    4. Always attach the medical safety disclaimer.

If no known test is recognized, or a test is recognized but no value
was found in the user's documents, falls back to a generic templated
summary of whatever was retrieved -- still with the disclaimer.
"""

from typing import List
from src.rag import RagContext
from src.medical_rules import find_test_value, interpret_value, find_matching_rule

SAFETY_DISCLAIMER = (
    "This is educational information, not a diagnosis. A single result "
    "should not be interpreted on its own -- please discuss it with a "
    "qualified healthcare professional."
)

_POSITION_PHRASES = {
    "above": "above the commonly used range",
    "below": "below the commonly used range",
    "within": "within the commonly used range",
}


def generate_response(context: RagContext) -> str:
    """
    Build a final answer string from a RagContext.

    Parameters
    ----------
    context : RagContext
        The combined retrieval results for one question (Phase 13).

    Returns
    -------
    str
        A complete, formatted answer, always ending with the safety
        disclaimer.
    """
    if context.is_empty():
        return (
            "I couldn't find anything relevant in your uploaded documents "
            "or the medical knowledge base to answer that. Try uploading a "
            "relevant document first, or rephrasing your question.\n\n"
            + SAFETY_DISCLAIMER
        )

    rule = find_matching_rule(context.query)

    if rule:
        # Look across ALL retrieved user-document chunks (not just the
        # top-ranked one) -- the actual value might be in the second
        # or third chunk instead.
        found_value = None
        for chunk in context.user_document_chunks:
            value = find_test_value(chunk.text, rule)
            if value is not None:
                found_value = value
                break

        if found_value is not None:
            position = interpret_value(rule, found_value)
            range_text = f"{rule.normal_low}-{rule.normal_high} {rule.unit}"

            answer = (
                f"Your {rule.display_name} result is {found_value} {rule.unit}, "
                f"which is {_POSITION_PHRASES[position]} of {range_text}.\n\n"
            )

            if position != "within":
                answer += (
                    f"Results {position} this range can be associated with "
                    "several possible causes -- see the retrieved medical "
                    "knowledge below for more detail. A single result like "
                    "this does not, by itself, confirm a diagnosis.\n\n"
                )
            else:
                answer += (
                    "This result falls within the commonly used reference "
                    "range, though ranges can vary somewhat by lab and by "
                    "individual circumstances.\n\n"
                )

            return answer + SAFETY_DISCLAIMER

    # Fallback: no recognized test, or a test was recognized but its
    # value wasn't found anywhere in the user's documents.
    summary_parts = []
    if context.user_document_chunks:
        top = context.user_document_chunks[0]
        summary_parts.append(
            f"From your document ({top.source}, page {top.page}): {top.text[:300]}"
        )
    if context.medical_knowledge_chunks:
        top = context.medical_knowledge_chunks[0]
        summary_parts.append(
            f"General medical background ({top.source}): {top.text[:300]}"
        )

    if summary_parts:
        answer = "\n\n".join(summary_parts)
    else:
        answer = (
            "I found some potentially related information, but couldn't "
            "identify a specific value or a clear answer to your question."
        )

    return answer + "\n\n" + SAFETY_DISCLAIMER


def get_sources(context: RagContext) -> List[str]:
    """
    Build a simple, deduplicated list of human-readable source strings
    for display, e.g. "blood_report.pdf — Page 2" or
    "Medical Knowledge — Glucose". Multiple chunks from the same
    source/page collapse into one entry.
    """
    sources = []
    for chunk in context.user_document_chunks:
        sources.append(f"{chunk.source} — Page {chunk.page}")
    for chunk in context.medical_knowledge_chunks:
        topic_name = chunk.source.replace(".txt", "").replace("_", " ").title()
        sources.append(f"Medical Knowledge — {topic_name}")

    # Deduplicate while preserving order (dict keys keep insertion order).
    return list(dict.fromkeys(sources))
