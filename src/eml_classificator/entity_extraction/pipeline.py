"""
Complete entity extraction pipeline.

Orchestrates RegEx + NER + Lexicon + Merge
as specified in brainstorming v2 section 5.4.
"""

from typing import Dict, List

import structlog

from .entity import Entity
from .lexicon_enhancer import enhance_ner_with_lexicon
from .merger import merge_entities_deterministic
from .ner_extractor import extract_entities_ner
from .regex_matcher import extract_entities_regex


logger = structlog.get_logger(__name__)


def extract_all_entities(
    text: str,
    label_id: str,
    regex_lexicon: Dict[str, List[dict]],
    ner_lexicon: Dict[str, List[dict]],
) -> List[Entity]:
    """
    Complete extraction pipeline: RegEx + NER + Lexicon + Merge.

    As specified in brainstorming v2 section 5.4 (lines 1815-1838).

    Pipeline stages:
    1. RegEx extraction (high precision) - from regex_lexicon
    2. NER generic extraction - spaCy Italian NER
    3. Enhance NER with lexicon (gazetteer) - from ner_lexicon
    4. Combine all entities
    5. Merge with deterministic rules (regex > lexicon > NER)

    Args:
        text: Email body canonicalized
        label_id: Label to extract entities for
        regex_lexicon: RegEx dictionary by label
            Format: {label_id: [{"lemma": ..., "regex_pattern": ...}]}
        ner_lexicon: NER dictionary by label
            Format: {label_id: [{"lemma": ..., "surface_forms": [...]}]}

    Returns:
        Merged list of non-overlapping entities, sorted by position

    Examples:
        >>> text = "Buongiorno, invio fattura n. 12345 per servizio gennaio."
        >>> regex_lex = {"FATTURAZIONE": [{"lemma": "fattura", "regex_pattern": r"(?i)\\b(fattura)\\b"}]}
        >>> ner_lex = {"FATTURAZIONE": [{"lemma": "fattura", "surface_forms": ["fattura"]}]}
        >>> entities = extract_all_entities(text, "FATTURAZIONE", regex_lex, ner_lex)
        >>> len(entities) > 0
        True
        >>> entities[0].source  # Regex has priority
        'regex'
    """
    logger.info(
        "entity_extraction_pipeline_start",
        label_id=label_id,
        text_length=len(text),
        regex_lexicon_entries=len(regex_lexicon.get(label_id, [])),
        ner_lexicon_entries=len(ner_lexicon.get(label_id, [])),
    )

    # Stage 1: RegEx extraction (high precision)
    regex_entities = extract_entities_regex(text, regex_lexicon, label_id)

    # Stage 2: NER generic extraction
    ner_entities = extract_entities_ner(text)

    # Stage 3: Enhance NER with lexicon (gazetteer)
    enhanced_entities = enhance_ner_with_lexicon(ner_entities, ner_lexicon, label_id, text)

    # Stage 4: Combine all entities
    all_entities = regex_entities + enhanced_entities

    # Stage 5: Merge with deterministic rules
    merged = merge_entities_deterministic(all_entities)

    logger.info(
        "entity_extraction_pipeline_complete",
        label_id=label_id,
        regex_count=len(regex_entities),
        ner_count=len(ner_entities),
        enhanced_count=len(enhanced_entities),
        total_before_merge=len(all_entities),
        merged_count=len(merged),
    )

    return merged
