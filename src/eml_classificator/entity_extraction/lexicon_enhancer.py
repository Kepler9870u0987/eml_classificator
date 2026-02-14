"""
Lexicon-based entity enhancement (gazetteer).

Enhances NER entities with matches from ner_lexicon
as specified in brainstorming v2 section 5.4.
"""

from typing import Dict, List

import structlog

from .entity import Entity


logger = structlog.get_logger(__name__)


def enhance_ner_with_lexicon(
    ner_entities: List[Entity], ner_lexicon: Dict[str, List[dict]], label_id: str, text: str
) -> List[Entity]:
    """
    Enhance NER entities with matches from ner_lexicon (gazetteer).

    As specified in brainstorming v2 section 5.4 (lines 1736-1769).

    Performs substring search with word boundary checking for each surface form
    in the lexicon, adding lexicon-based entities to the original NER entities.

    Args:
        ner_entities: Entities from NER extraction
        ner_lexicon: Dictionary mapping label_id to list of entries.
            Each entry contains: {"lemma": ..., "surface_forms": [...]}
        label_id: Label to enhance for
        text: Email body canonicalized

    Returns:
        Enhanced list of entities (original NER + lexicon matches)

    Examples:
        >>> ner_entities = []  # Assume empty NER results
        >>> lexicon = {
        ...     "FATTURAZIONE": [
        ...         {"lemma": "fattura", "surface_forms": ["fattura", "Fattura"]}
        ...     ]
        ... }
        >>> text = "Invio fattura n. 12345"
        >>> enhanced = enhance_ner_with_lexicon(ner_entities, lexicon, "FATTURAZIONE", text)
        >>> len(enhanced)
        1
        >>> enhanced[0].source
        'lexicon'
    """
    enhanced = list(ner_entities)

    entries = ner_lexicon.get(label_id, [])

    logger.debug(
        "lexicon_enhancement_start",
        label_id=label_id,
        ner_entities_count=len(ner_entities),
        lexicon_entries_count=len(entries),
    )

    for entry in entries:
        lemma = entry.get("lemma")
        surface_forms = entry.get("surface_forms", [lemma] if lemma else [])

        if not lemma or not surface_forms:
            logger.warning("lexicon_entry_missing_fields", entry=entry, label_id=label_id)
            continue

        for sf in surface_forms:
            # Simple substring search (case insensitive)
            lower_text = text.lower()
            lower_sf = sf.lower()
            start = 0

            while True:
                pos = lower_text.find(lower_sf, start)
                if pos == -1:
                    break

                # Check word boundary (simplified)
                # Check if character before match is alphanumeric
                is_start_boundary = pos == 0 or not lower_text[pos - 1].isalnum()

                # Check if character after match is alphanumeric
                end_pos = pos + len(lower_sf)
                is_end_boundary = end_pos == len(lower_text) or not lower_text[end_pos].isalnum()

                if is_start_boundary and is_end_boundary:
                    entity = Entity(
                        text=text[pos : pos + len(sf)],
                        label=lemma,
                        start=pos,
                        end=pos + len(sf),
                        source="lexicon",
                        confidence=0.85,  # Mid confidence for lexicon matches
                    )
                    enhanced.append(entity)

                start = pos + 1

    logger.debug(
        "lexicon_enhancement_complete",
        label_id=label_id,
        original_count=len(ner_entities),
        enhanced_count=len(enhanced),
        added_count=len(enhanced) - len(ner_entities),
    )

    return enhanced
