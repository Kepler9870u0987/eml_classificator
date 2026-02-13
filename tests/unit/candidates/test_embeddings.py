"""
Unit tests for embeddings and composite scoring (Phase 2 - Candidate Generation).

Tests coverage:
- enrich_candidates_with_embeddings: KeyBERT integration
- calculate_composite_scores: Composite score formula
- Model loading and singleton pattern
- Error handling (KeyBERT failures)
- Score range validation

Testing principles:
1. Mock KeyBERT for deterministic tests
2. Test composite scoring formula
3. Test error handling
4. Validate score ranges
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
import numpy as np
from datetime import datetime

from eml_classificator.candidates.embeddings import (
    enrich_candidates_with_embeddings,
    calculate_composite_scores,
    get_keybert_model,
    EMBEDDING_MODEL_VERSION,
)
from eml_classificator.models.candidates import KeywordCandidate
from eml_classificator.models.email_document import EmailDocument, EmailHeaders
from eml_classificator.models.pipeline_version import PipelineVersion


def create_simple_document(subject: str = "", body: str = "") -> EmailDocument:
    """Helper to create EmailDocument for testing."""
    return EmailDocument(
        document_id="test-doc-id",
        headers=EmailHeaders(
            from_address="test@example.com",
            to_addresses=["support@example.com"],
            cc_addresses=[],
            subject=subject,
            date=datetime(2025, 2, 13, 10, 0, 0),
            message_id="<test@example.com>",
            in_reply_to=None,
            references=[],
            extra_headers={},
        ),
        body_text=body,
        body_html=None,
        canonical_text=body,
        attachments=[],
        removed_sections=[],
        pii_redacted=False,
        pii_redaction_log=[],
        pipeline_version=PipelineVersion(
            dictionary_version=1,
            model_version="test-1.0",
            parser_version="parser-1.0",
            stoplist_version="stop-1.0",
            ner_model_version="ner-1.0",
            schema_version="schema-1.0",
            tool_calling_version="tool-1.0",
            canonicalization_version="canon-1.0",
            pii_redaction_version="pii-1.0",
            pii_redaction_level="standard",
        ),
        ingestion_timestamp=datetime(2025, 2, 13, 10, 5, 0),
        processing_time_ms=50.0,
        raw_size_bytes=1024,
        encoding_detected="utf-8",
    )


def create_candidate(term: str, source: str = "body", count: int = 1, lemma: str = None) -> KeywordCandidate:
    """Helper to create KeywordCandidate for testing."""
    if lemma is None:
        lemma = term.lower()

    return KeywordCandidate(
        candidate_id=f"test-{term}",
        source=source,
        term=term,
        count=count,
        lemma=lemma,
        ngram_size=len(term.split()),
        embedding_score=0.0,
        composite_score=0.0,
    )


# ============================================================================
# Test Class: EnrichCandidatesWithEmbeddings
# ============================================================================


@pytest.mark.unit
class TestEnrichCandidatesWithEmbeddings:
    """Tests for enrich_candidates_with_embeddings function."""

    @patch("eml_classificator.candidates.embeddings.get_keybert_model")
    def test_basic_enrichment(self, mock_get_keybert):
        """Test that candidates get embedding_score assigned."""
        # Mock KeyBERT
        mock_model = Mock()
        mock_model.extract_keywords.return_value = [
            ("fattura", 0.85),
            ("cliente", 0.75),
        ]
        mock_get_keybert.return_value = mock_model

        doc = create_simple_document(
            subject="Fattura urgente", body="Il cliente ha un problema"
        )
        candidates = [
            create_candidate("fattura"),
            create_candidate("cliente"),
            create_candidate("urgente"),
        ]

        enriched = enrich_candidates_with_embeddings(doc, candidates)

        # Check embedding scores assigned
        assert enriched[0].embedding_score > 0.0
        assert enriched[1].embedding_score > 0.0

    @patch("eml_classificator.candidates.embeddings.get_keybert_model")
    def test_top_candidates_score_higher(self, mock_get_keybert):
        """Test that relevant keywords get higher scores."""
        mock_model = Mock()
        mock_model.extract_keywords.return_value = [
            ("urgente", 0.90),  # High relevance
            ("cliente", 0.50),  # Lower relevance
        ]
        mock_get_keybert.return_value = mock_model

        doc = create_simple_document(body="Cliente urgente")
        candidates = [
            create_candidate("urgente"),
            create_candidate("cliente"),
        ]

        enriched = enrich_candidates_with_embeddings(doc, candidates)

        urgente_score = next(c.embedding_score for c in enriched if c.term == "urgente")
        cliente_score = next(c.embedding_score for c in enriched if c.term == "cliente")

        assert urgente_score > cliente_score

    @patch("eml_classificator.candidates.embeddings.get_keybert_model")
    def test_italian_text_support(self, mock_get_keybert):
        """Test Italian text processing."""
        mock_model = Mock()
        mock_model.extract_keywords.return_value = [
            ("fattura", 0.85),
            ("problema", 0.70),
        ]
        mock_get_keybert.return_value = mock_model

        doc = create_simple_document(
            body="Il cliente ha un problema con la fattura"
        )
        candidates = [
            create_candidate("fattura"),
            create_candidate("problema"),
        ]

        enriched = enrich_candidates_with_embeddings(doc, candidates)

        assert all(c.embedding_score > 0.0 for c in enriched)

    @patch("eml_classificator.candidates.embeddings.get_keybert_model")
    def test_empty_text(self, mock_get_keybert):
        """Test with empty subject and body."""
        doc = create_simple_document(subject="", body="")
        candidates = [create_candidate("test")]

        enriched = enrich_candidates_with_embeddings(doc, candidates)

        # Should return candidates unchanged (no enrichment)
        assert all(c.embedding_score == 0.0 for c in enriched)
        mock_get_keybert.assert_not_called()

    @patch("eml_classificator.candidates.embeddings.get_keybert_model")
    def test_keybert_failure_handling(self, mock_get_keybert):
        """Test graceful handling of KeyBERT exception."""
        mock_model = Mock()
        mock_model.extract_keywords.side_effect = Exception("KeyBERT error")
        mock_get_keybert.return_value = mock_model

        doc = create_simple_document(body="Test text")
        candidates = [create_candidate("test")]

        # Should not raise, should return candidates with 0.0 scores
        enriched = enrich_candidates_with_embeddings(doc, candidates)

        assert len(enriched) == 1
        assert enriched[0].embedding_score == 0.0

    @patch("eml_classificator.candidates.embeddings.get_keybert_model")
    def test_candidate_not_in_keybert_results(self, mock_get_keybert):
        """Test candidate not found in KeyBERT results gets 0.0 score."""
        mock_model = Mock()
        mock_model.extract_keywords.return_value = [
            ("fattura", 0.85),
        ]
        mock_get_keybert.return_value = mock_model

        doc = create_simple_document(body="Fattura")
        candidates = [
            create_candidate("fattura"),  # Will match
            create_candidate("nonexistent"),  # Won't match
        ]

        enriched = enrich_candidates_with_embeddings(doc, candidates)

        nonexistent_score = next(
            c.embedding_score for c in enriched if c.term == "nonexistent"
        )
        assert nonexistent_score == 0.0

    @patch("eml_classificator.candidates.embeddings.get_keybert_model")
    def test_lemma_matching(self, mock_get_keybert):
        """Test that KeyBERT scores match by lemma if term doesn't match."""
        mock_model = Mock()
        # KeyBERT returns lemma form
        mock_model.extract_keywords.return_value = [
            ("cliente", 0.75),
        ]
        mock_get_keybert.return_value = mock_model

        doc = create_simple_document(body="Clienti")
        candidates = [
            create_candidate("clienti", lemma="cliente"),
        ]

        enriched = enrich_candidates_with_embeddings(doc, candidates)

        # Should match by lemma
        assert enriched[0].embedding_score == 0.75

    @patch("eml_classificator.candidates.embeddings.get_keybert_model")
    def test_top_n_limit_respected(self, mock_get_keybert):
        """Test that top_n parameter is passed to KeyBERT."""
        mock_model = Mock()
        mock_model.extract_keywords.return_value = []
        mock_get_keybert.return_value = mock_model

        doc = create_simple_document(body="Test text")
        candidates = [create_candidate("test")]

        enrich_candidates_with_embeddings(doc, candidates, top_n=20)

        # Check that KeyBERT was called with top_n=20
        call_kwargs = mock_model.extract_keywords.call_args[1]
        assert call_kwargs["top_n"] == 20


# ============================================================================
# Test Class: CalculateCompositeScores
# ============================================================================


@pytest.mark.unit
class TestCalculateCompositeScores:
    """Tests for calculate_composite_scores function."""

    def test_score_calculation_formula(self):
        """Test composite score formula: 0.3*count + 0.5*embedding + 0.2*subject_bonus."""
        candidates = [
            KeywordCandidate(
                candidate_id="test-1",
                source="body",
                term="test",
                count=2,
                lemma="test",
                ngram_size=1,
                embedding_score=0.8,
                composite_score=0.0,
            )
        ]

        scored = calculate_composite_scores(candidates)

        # Manual calculation:
        # count_norm = log1p(2) / 5.0 = 1.0986 / 5.0 = 0.2197
        # composite = 0.3 * 0.2197 + 0.5 * 0.8 + 0.2 * 0.0
        #           = 0.0659 + 0.4 + 0.0 = 0.4659

        assert scored[0].composite_score > 0.0
        assert 0.4 < scored[0].composite_score < 0.5

    def test_count_component(self):
        """Test count component normalization with log1p."""
        cand = create_candidate("test", count=10)
        cand.embedding_score = 0.0

        scored = calculate_composite_scores([cand])

        # Count component should be: log1p(10) / 5.0 = 2.398 / 5.0 = 0.4796
        # With weight 0.3: 0.3 * 0.4796 = 0.1439
        assert scored[0].composite_score > 0.1
        assert scored[0].composite_score < 0.2

    def test_subject_bonus_applied(self):
        """Test subject bonus: source='subject' gets +0.2 weight."""
        candidates = [
            KeywordCandidate(
                candidate_id="test-1",
                source="subject",
                term="test",
                count=2,
                lemma="test",
                ngram_size=1,
                embedding_score=0.8,
                composite_score=0.0,
            ),
            KeywordCandidate(
                candidate_id="test-2",
                source="body",
                term="test",
                count=2,
                lemma="test",
                ngram_size=1,
                embedding_score=0.8,
                composite_score=0.0,
            ),
        ]

        scored = calculate_composite_scores(candidates)

        subject_score = scored[0].composite_score
        body_score = scored[1].composite_score

        # Subject should have higher score due to bonus
        assert subject_score > body_score
        assert abs(subject_score - body_score - 0.2) < 0.01

    def test_embedding_dominant(self):
        """Test that high embedding score contributes most to composite."""
        candidates = [
            create_candidate("high_emb", count=1),
            create_candidate("low_emb", count=1),
        ]
        candidates[0].embedding_score = 0.9  # High
        candidates[1].embedding_score = 0.1  # Low

        scored = calculate_composite_scores(candidates)

        high_score = scored[0].composite_score
        low_score = scored[1].composite_score

        assert high_score > low_score

    def test_zero_embedding_handling(self):
        """Test candidates with embedding_score=0.0 still get score from count."""
        cand = create_candidate("test", count=5)
        cand.embedding_score = 0.0

        scored = calculate_composite_scores([cand])

        # Should have score from count component only
        assert scored[0].composite_score > 0.0

    def test_custom_weights(self):
        """Test that custom weights are applied correctly."""
        custom_weights = {
            "count": 0.5,
            "embedding": 0.3,
            "subject_bonus": 0.2,
        }

        cand = create_candidate("test", count=2)
        cand.embedding_score = 0.8

        scored = calculate_composite_scores([cand], weights=custom_weights)

        # With custom weights, count has more weight
        # count_norm = log1p(2) / 5.0 = 0.2197
        # composite = 0.5 * 0.2197 + 0.3 * 0.8 + 0.0
        #           = 0.1099 + 0.24 = 0.3499
        assert 0.3 < scored[0].composite_score < 0.4

    def test_score_range(self):
        """Test that composite scores are in reasonable range (typically 0-1)."""
        candidates = [
            create_candidate("test1", count=1),
            create_candidate("test2", count=10),
            create_candidate("test3", source="subject", count=5),
        ]
        for c in candidates:
            c.embedding_score = 0.5

        scored = calculate_composite_scores(candidates)

        for cand in scored:
            # Scores should typically be in 0-1 range
            assert 0.0 <= cand.composite_score <= 2.0  # Allow some overhead


# ============================================================================
# Test: Model Loading and Version
# ============================================================================


@pytest.mark.unit
def test_embedding_model_version_defined():
    """Test that EMBEDDING_MODEL_VERSION is defined."""
    assert EMBEDDING_MODEL_VERSION is not None
    assert isinstance(EMBEDDING_MODEL_VERSION, str)
    assert "multilingual" in EMBEDDING_MODEL_VERSION.lower()
