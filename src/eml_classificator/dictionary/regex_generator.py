"""
Safe regex pattern generation from surface forms.

Implements regex generation as specified in brainstorming v2 section 5.3.
"""

import re
from typing import List

import structlog


logger = structlog.get_logger(__name__)


def generate_safe_regex(surface_forms: List[str], case_sensitive: bool = False) -> str:
    """
    Generate safe regex pattern from list of surface forms.

    As specified in brainstorming v2 section 5.3 (lines 1629-1653).

    Escapes special regex characters and uses word boundaries for Italian text.

    Args:
        surface_forms: List of terms (e.g., ["fattura", "Fattura", "FATTURA"])
        case_sensitive: Whether to match case-sensitively (default: False)

    Returns:
        Regex pattern string with proper escaping and word boundaries

    Raises:
        ValueError: If pattern fails to compile

    Examples:
        >>> pattern = generate_safe_regex(["fattura", "Fattura"], case_sensitive=False)
        >>> pattern
        '(?i)\\\\b(fattura|Fattura)\\\\b'

        >>> regex = re.compile(pattern)
        >>> regex.findall("Invio fattura n. 12345")
        ['fattura']
    """
    if not surface_forms:
        logger.warning("generate_safe_regex_called_with_empty_forms")
        return r"\b(?!)\b"  # Pattern that never matches

    # Escape special regex characters
    escaped = [re.escape(sf) for sf in surface_forms]

    # Word boundary for Italian (includes apostrophe)
    pattern = r"\b(" + "|".join(escaped) + r")\b"

    # Case insensitive flag
    if not case_sensitive:
        pattern = "(?i)" + pattern

    # Validate pattern compiles
    try:
        re.compile(pattern)
    except re.error as e:
        logger.error(
            "regex_compilation_failed", pattern=pattern, error=str(e), surface_forms=surface_forms
        )
        raise ValueError(f"Invalid regex pattern: {pattern}, error: {e}")

    logger.debug(
        "regex_pattern_generated",
        pattern=pattern,
        surface_forms_count=len(surface_forms),
        case_sensitive=case_sensitive,
    )

    return pattern


def test_regex_pattern(pattern: str, test_texts: List[str]) -> List[List[str]]:
    """
    Test regex pattern against sample texts.

    Utility function for validating regex patterns before deployment.

    Args:
        pattern: Regex pattern string to test
        test_texts: List of texts to test against

    Returns:
        List of match lists for each text

    Examples:
        >>> pattern = generate_safe_regex(["fattura"])
        >>> test_regex_pattern(pattern, ["Invio fattura", "No match here"])
        [['fattura'], []]
    """
    try:
        regex = re.compile(pattern)
    except re.error as e:
        logger.error("test_regex_invalid_pattern", pattern=pattern, error=str(e))
        return [[] for _ in test_texts]

    results = []
    for text in test_texts:
        matches = regex.findall(text)
        results.append(matches)

    logger.debug(
        "regex_pattern_tested",
        pattern=pattern,
        test_texts_count=len(test_texts),
        total_matches=sum(len(m) for m in results),
    )

    return results
