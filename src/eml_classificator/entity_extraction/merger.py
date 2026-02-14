"""
Deterministic entity merging.

Merges overlapping entities with deterministic priority rules
as specified in brainstorming v2 section 5.4.
"""

from typing import List

import structlog

from .entity import Entity


logger = structlog.get_logger(__name__)


def merge_entities_deterministic(entities: List[Entity]) -> List[Entity]:
    """
    Merge overlapping entities with deterministic rules.

    As specified in brainstorming v2 section 5.4 (lines 1770-1814).

    Priority rules (in order):
    1. RegEx > Lexicon > NER (by source priority)
    2. If same source, longest span wins
    3. If same length, higher confidence wins

    Args:
        entities: List of entities to merge

    Returns:
        Merged list of non-overlapping entities, sorted by position

    Examples:
        >>> e1 = Entity("fattura", "fattura", 10, 17, "regex", 0.95)
        >>> e2 = Entity("fattura", "NOUN", 10, 17, "ner", 0.75)
        >>> merged = merge_entities_deterministic([e1, e2])
        >>> len(merged)
        1
        >>> merged[0].source
        'regex'
    """
    if not entities:
        return []

    # Sort by priority
    source_priority = {"regex": 0, "lexicon": 1, "ner": 2}

    # Sort primarily by start position, then by:
    # - end position (descending - prefer longer spans)
    # - source priority (ascending - prefer regex)
    # - confidence (descending - prefer higher confidence)
    entities_sorted = sorted(
        entities,
        key=lambda e: (e.start, -e.end, source_priority[e.source], -e.confidence),
    )

    merged = []
    for entity in entities_sorted:
        # Check overlap with entities already in merged
        has_overlap = False

        for existing in merged:
            if entity.overlaps(existing):
                has_overlap = True

                # Decide which to keep based on priority rules
                should_replace = False

                # Priority 1: source type
                entity_priority = source_priority[entity.source]
                existing_priority = source_priority[existing.source]

                if entity_priority < existing_priority:
                    # Entity has higher source priority (regex > lexicon > ner)
                    should_replace = True
                elif entity_priority == existing_priority:
                    # Same source, check length
                    entity_len = entity.length()
                    existing_len = existing.length()

                    if entity_len > existing_len:
                        # Entity is longer
                        should_replace = True
                    elif entity_len == existing_len:
                        # Same length, check confidence
                        if entity.confidence > existing.confidence:
                            should_replace = True

                if should_replace:
                    merged.remove(existing)
                    merged.append(entity)
                    logger.debug(
                        "entity_replaced",
                        replaced=str(existing),
                        new=str(entity),
                        reason=f"priority: {entity.source} > {existing.source}"
                        if entity_priority < existing_priority
                        else f"length: {entity.length()} > {existing.length()}"
                        if entity.length() != existing.length()
                        else f"confidence: {entity.confidence} > {existing.confidence}",
                    )

                break

        if not has_overlap:
            merged.append(entity)

    # Final sort by position
    merged.sort(key=lambda e: e.start)

    logger.debug(
        "entity_merge_complete",
        original_count=len(entities),
        merged_count=len(merged),
        removed_count=len(entities) - len(merged),
    )

    return merged
