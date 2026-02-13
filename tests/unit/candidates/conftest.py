"""
Shared fixtures for Phase 2 candidate extraction unit tests.

Provides mocked models and sample data for deterministic testing.
"""

import pytest
from unittest.mock import MagicMock, Mock
from typing import List
from eml_classificator.models.candidates import KeywordCandidate


@pytest.fixture
def mock_spacy_model():
    """Mock spaCy model for fast testing without model download."""
    mock_nlp = MagicMock()
    mock_doc = MagicMock()

    # Mock lemmatization behavior
    def process_text(text: str):
        # Simple mock lemmatization (for testing purposes)
        mock_doc.text = text
        mock_tokens = []

        # Simple lemma mapping for common Italian words
        lemma_map = {
            "cliente": "cliente",
            "clienti": "cliente",
            "fattura": "fattura",
            "fatture": "fattura",
            "urgente": "urgente",
            "urgent": "urgente",
            "problema": "problema",
            "problemi": "problema",
            "ordine": "ordine",
            "ordini": "ordine",
            "pagamento": "pagamento",
            "pagamenti": "pagamento",
            "città": "città",
            "dell'ordine": "dell'ordine",
            "12345": "12345",
        }

        for token_text in text.lower().split():
            mock_token = MagicMock()
            mock_token.text = token_text
            mock_token.lemma_ = lemma_map.get(token_text, token_text)
            mock_tokens.append(mock_token)

        mock_doc.__iter__ = lambda self: iter(mock_tokens)
        return mock_doc

    mock_nlp.side_effect = process_text
    return mock_nlp


@pytest.fixture
def mock_keybert_model():
    """Mock KeyBERT model for deterministic testing."""
    mock_model = MagicMock()

    def mock_extract_keywords(text, **kwargs):
        """Return deterministic keyword scores based on text content."""
        # Simple scoring logic for testing
        keywords = []
        if "fattura" in text.lower():
            keywords.append(("fattura", 0.85))
        if "cliente" in text.lower():
            keywords.append(("cliente", 0.75))
        if "urgente" in text.lower():
            keywords.append(("urgente", 0.90))
        if "problema" in text.lower():
            keywords.append(("problema", 0.70))
        if "pagamento" in text.lower():
            keywords.append(("pagamento", 0.65))

        # Return top_n results
        top_n = kwargs.get("top_n", 50)
        return keywords[:top_n]

    mock_model.extract_keywords = Mock(side_effect=mock_extract_keywords)
    return mock_model


@pytest.fixture
def sample_italian_texts():
    """Sample Italian texts for testing."""
    return {
        "simple": "Il cliente ha un problema",
        "with_apostrophe": "dell'ordine dell'utente",
        "with_accents": "città però caffè",
        "mixed_case": "URGENTE Fattura",
        "with_numbers": "Fattura 12345 del 2025",
        "with_punctuation": "Salve, come va?",
        "email_signature": "Cordiali saluti, Distinti saluti",
        "long_text": " ".join(["Il cliente ha un problema con la fattura"] * 100),
    }


@pytest.fixture
def sample_keyword_candidates() -> List[KeywordCandidate]:
    """Sample KeywordCandidate objects for testing."""
    return [
        KeywordCandidate(
            candidate_id="abc123def456",
            source="subject",
            term="fattura urgente",
            count=1,
            lemma="fattura urgente",
            ngram_size=2,
            embedding_score=0.85,
            composite_score=0.70,
        ),
        KeywordCandidate(
            candidate_id="def456ghi789",
            source="body",
            term="cliente",
            count=3,
            lemma="cliente",
            ngram_size=1,
            embedding_score=0.75,
            composite_score=0.65,
        ),
        KeywordCandidate(
            candidate_id="ghi789jkl012",
            source="body",
            term="problema",
            count=2,
            lemma="problema",
            ngram_size=1,
            embedding_score=0.70,
            composite_score=0.60,
        ),
    ]


@pytest.fixture
def sample_stopwords():
    """Sample Italian stopwords for testing."""
    return {
        # Articles
        "il", "la", "i", "le", "lo", "gli", "un", "una", "uno",
        # Prepositions
        "di", "a", "da", "in", "con", "su", "per", "tra", "fra",
        # Email-specific
        "cordiali", "saluti", "distinti", "grazie", "buongiorno", "buonasera",
        # Common verbs
        "è", "sono", "ha", "hanno", "essere", "avere",
    }


@pytest.fixture
def sample_blacklist_patterns():
    """Sample blacklist regex patterns for testing."""
    return [
        r"^re:",
        r"^fwd:",
        r"^r:",
        r"^fw:",
        r"^\d+$",  # Isolated numbers
        r"^\d{1,2}/\d{1,2}/\d{2,4}$",  # Dates
    ]


@pytest.fixture
def fixed_timestamp():
    """Fixed timestamp for deterministic testing."""
    return "2025-02-13T10:00:00Z"


@pytest.fixture
def sample_email_document_dict():
    """Sample EmailDocument as dict for API testing."""
    return {
        "message_id": "<test@example.com>",
        "subject": "Problema con la fattura",
        "from_email": "mario.rossi@example.com",
        "to_email": ["support@company.com"],
        "date": "2025-02-13T09:00:00Z",
        "body_text": "Buongiorno, ho un problema urgente con la fattura numero 12345. Grazie.",
        "body_html": None,
        "has_attachments": False,
        "attachment_count": 0,
        "canonical_text": "Buongiorno, ho un problema urgente con la fattura numero 12345.",
        "parser_version": "email-parser-1.0.0",
        "ingestion_timestamp": "2025-02-13T09:05:00Z",
    }


@pytest.fixture
def sample_email_document():
    """Sample EmailDocument instance for testing candidate extraction."""
    from datetime import datetime
    from eml_classificator.models.email_document import (
        EmailDocument,
        EmailHeaders,
    )
    from eml_classificator.models.pipeline_version import PipelineVersion

    return EmailDocument(
        document_id="test-doc-id-12345",
        headers=EmailHeaders(
            from_address="mario.rossi@example.com",
            to_addresses=["support@company.com"],
            cc_addresses=[],
            subject="Problema con la fattura",
            date=datetime(2025, 2, 13, 9, 0, 0),
            message_id="<test@example.com>",
            in_reply_to=None,
            references=[],
            extra_headers={},
        ),
        body_text="Buongiorno, ho un problema urgente con la fattura numero 12345. Grazie.",
        body_html=None,
        canonical_text="Buongiorno, ho un problema urgente con la fattura numero 12345.",
        attachments=[],
        removed_sections=[],
        pii_redacted=False,
        pii_redaction_log=[],
        pipeline_version=PipelineVersion(
            dictionary_version=1,
            model_version="test-model-1.0",
            parser_version="eml-parser-1.0.0",
            stoplist_version="stopwords-it-2025.1",
            ner_model_version="it_core_news_lg-3.8.0",
            schema_version="json-schema-v2.0",
            tool_calling_version="test-tool-calling-1.0",
            canonicalization_version="canon-1.0.0",
            pii_redaction_version="pii-redact-1.0.0",
            pii_redaction_level="standard",
        ),
        ingestion_timestamp=datetime(2025, 2, 13, 9, 5, 0),
        processing_time_ms=50.0,
    )
