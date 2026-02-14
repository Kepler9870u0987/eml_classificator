"""
RegEx-based entity extraction.

Extracts entities using regex patterns from dictionary entries
as specified in brainstorming v2 section 5.4.
"""

import re
from typing import Dict, List

import structlog

from .entity import Entity


logger = structlog.get_logger(__name__)


def extract_entities_regex(
    text: str, regex_lexicon: Dict[str, List[dict]], label_id: str
) -> List[Entity]:
    """
    Extract entities using RegEx from dictionary for label.

    As specified in brainstorming v2 section 5.4 (lines 1692-1718).

    Args:
        text: Email body canonicalized
        regex_lexicon: Dictionary mapping label_id to list of entries.
            Each entry contains: {"lemma": ..., "regex_pattern": ..., ...}
        label_id: Label to extract entities for

    Returns:
        List of Entity objects with source='regex' and confidence=0.95

    Examples:
        >>> text = "Buongiorno, invio fattura n. 12345"
        >>> lexicon = {
        ...     "FATTURAZIONE": [
        ...         {"lemma": "fattura", "regex_pattern": r"(?i)\\b(fattura)\\b"}
        ...     ]
        ... }
        >>> entities = extract_entities_regex(text, lexicon, "FATTURAZIONE")
        >>> len(entities)
        1
        >>> entities[0].text
        'fattura'
    """
    entities = []
    entries = regex_lexicon.get(label_id, [])

    logger.debug(
        "regex_extraction_start", label_id=label_id, entries_count=len(entries), text_length=len(text)
    )

    for entry in entries:
        pattern = entry.get("regex_pattern")
        lemma = entry.get("lemma")

        if not pattern or not lemma:
            logger.warning("regex_entry_missing_fields", entry=entry, label_id=label_id)
            continue

        try:
            regex = re.compile(pattern, re.IGNORECASE)
        except Exception as e:
            logger.error(
                "invalid_regex_pattern",
                pattern=pattern,
                lemma=lemma,
                label_id=label_id,
                error=str(e),
            )
            continue

        for match in regex.finditer(text):
            entity = Entity(
                text=match.group(0),
                label=lemma,
                start=match.start(),
                end=match.end(),
                source="regex",
                confidence=0.95,  # High confidence for regex matches
            )
            entities.append(entity)

    logger.debug("regex_extraction_complete", label_id=label_id, entities_count=len(entities))

    return entities
