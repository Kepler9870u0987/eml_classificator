"""
Entity extraction pipeline (RegEx + NER + Lexicon merge).

Implements entity extraction as specified in brainstorming v2 section 5.4.

Public API:
    - Entity: Dataclass for extracted entities
    - extract_all_entities: Complete pipeline (recommended)
    - extract_entities_regex: RegEx-only extraction
    - extract_entities_ner: NER-only extraction
    - enhance_ner_with_lexicon: Lexicon enhancement
    - merge_entities_deterministic: Entity merging

Example usage:
    >>> from eml_classificator.entity_extraction import extract_all_entities, Entity
    >>>
    >>> text = "Invio fattura n. 12345"
    >>> regex_lexicon = {
    ...     "FATTURAZIONE": [
    ...         {"lemma": "fattura", "regex_pattern": r"(?i)\\b(fattura)\\b"}
    ...     ]
    ... }
    >>> ner_lexicon = {
    ...     "FATTURAZIONE": [
    ...         {"lemma": "fattura", "surface_forms": ["fattura", "Fattura"]}
    ...     ]
    ... }
    >>>
    >>> entities = extract_all_entities(text, "FATTURAZIONE", regex_lexicon, ner_lexicon)
    >>> for entity in entities:
    ...     print(f"{entity.text} [{entity.label}] @ {entity.start}-{entity.end} ({entity.source})")
    fattura [fattura] @ 6-13 (regex)
"""

from .entity import Entity
from .lexicon_enhancer import enhance_ner_with_lexicon
from .merger import merge_entities_deterministic
from .ner_extractor import extract_entities_ner, get_ner_model
from .pipeline import extract_all_entities
from .regex_matcher import extract_entities_regex

__all__ = [
    # Main API
    "Entity",
    "extract_all_entities",
    # Component functions (for advanced usage)
    "extract_entities_regex",
    "extract_entities_ner",
    "enhance_ner_with_lexicon",
    "merge_entities_deterministic",
    "get_ner_model",
]
