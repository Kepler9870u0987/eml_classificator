"""
NER-based entity extraction using spaCy.

Extracts entities using spaCy Italian NER model
as specified in brainstorming v2 section 5.4.
"""

from typing import List

import spacy
import structlog

from .entity import Entity


logger = structlog.get_logger(__name__)


# Global NER model (loaded once)
_nlp_model = None


def get_ner_model():
    """
    Load spaCy NER model (singleton).

    Returns:
        Loaded spaCy model

    Raises:
        OSError: If model is not installed
    """
    global _nlp_model

    if _nlp_model is None:
        from eml_classificator.config import settings

        try:
            _nlp_model = spacy.load(settings.spacy_model_name)
            logger.info("ner_model_loaded", model=settings.spacy_model_name)
        except OSError as e:
            logger.error(
                "ner_model_load_failed",
                model=settings.spacy_model_name,
                error=str(e),
                hint="Run: python -m spacy download it_core_news_lg",
            )
            raise

    return _nlp_model


def extract_entities_ner(text: str) -> List[Entity]:
    """
    Extract entities using spaCy NER.

    As specified in brainstorming v2 section 5.4 (lines 1719-1735).

    Args:
        text: Email body canonicalized

    Returns:
        List of Entity objects with source='ner' and confidence=0.75

    Examples:
        >>> text = "Mario Rossi lavora a Roma per Acme S.p.A."
        >>> entities = extract_entities_ner(text)
        >>> any(e.label == "PER" for e in entities)  # Person entity
        True
    """
    nlp = get_ner_model()
    doc = nlp(text)

    entities = []
    for ent in doc.ents:
        entity = Entity(
            text=ent.text,
            label=ent.label_,  # spaCy NER label (PER, ORG, LOC, etc.)
            start=ent.start_char,
            end=ent.end_char,
            source="ner",
            confidence=0.75,  # NER generally less precise than regex
        )
        entities.append(entity)

    logger.debug("ner_extraction_complete", entities_count=len(entities), text_length=len(text))

    return entities
