"""
Unit tests for candidate data models (Phase 2 - Candidate Generation).

Tests coverage:
- KeywordCandidate: Field validation, ranges, types
- CandidateExtractionResult: Nested model validation, metadata
- Pydantic validation behavior
- Model serialization (dict, JSON)

Testing principles:
1. Test Pydantic field validation
2. Test field constraints (ge, le, Literal)
3. Test model serialization
4. Test required vs optional fields
"""

import pytest
from pydantic import ValidationError
import json

from eml_classificator.models.candidates import (
    KeywordCandidate,
    CandidateExtractionResult,
    CandidateSource,
)


# ============================================================================
# Test Class: KeywordCandidate Model
# ============================================================================


@pytest.mark.unit
class TestKeywordCandidateModel:
    """Tests for KeywordCandidate Pydantic model."""

    def test_valid_candidate_creation(self):
        """Test creating valid KeywordCandidate with all fields."""
        cand = KeywordCandidate(
            candidate_id="abc123def456",
            source="subject",
            term="fattura urgente",
            count=3,
            lemma="fattura urgente",
            ngram_size=2,
            embedding_score=0.85,
            composite_score=0.70,
        )

        assert cand.candidate_id == "abc123def456"
        assert cand.source == "subject"
        assert cand.term == "fattura urgente"
        assert cand.count == 3
        assert cand.lemma == "fattura urgente"
        assert cand.ngram_size == 2
        assert cand.embedding_score == 0.85
        assert cand.composite_score == 0.70

    def test_candidate_id_validation(self):
        """Test candidate_id accepts string (12-char hex in practice)."""
        cand = KeywordCandidate(
            candidate_id="short",  # Pydantic doesn't enforce length, just string
            source="body",
            term="test",
            count=1,
            lemma="test",
            ngram_size=1,
        )
        assert isinstance(cand.candidate_id, str)

    def test_source_validation(self):
        """Test source must be 'subject' or 'body'."""
        # Valid sources
        cand_subject = KeywordCandidate(
            candidate_id="test",
            source="subject",
            term="test",
            count=1,
            lemma="test",
            ngram_size=1,
        )
        assert cand_subject.source == "subject"

        cand_body = KeywordCandidate(
            candidate_id="test",
            source="body",
            term="test",
            count=1,
            lemma="test",
            ngram_size=1,
        )
        assert cand_body.source == "body"

        # Invalid source should raise ValidationError
        with pytest.raises(ValidationError):
            KeywordCandidate(
                candidate_id="test",
                source="unknown",  # Not 'subject' or 'body'
                term="test",
                count=1,
                lemma="test",
                ngram_size=1,
            )

    def test_count_validation(self):
        """Test count must be positive integer ≥ 1."""
        # Valid count
        cand = KeywordCandidate(
            candidate_id="test",
            source="body",
            term="test",
            count=1,
            lemma="test",
            ngram_size=1,
        )
        assert cand.count == 1

        # Invalid count (< 1)
        with pytest.raises(ValidationError):
            KeywordCandidate(
                candidate_id="test",
                source="body",
                term="test",
                count=0,  # Must be >= 1
                lemma="test",
                ngram_size=1,
            )

    def test_ngram_size_validation(self):
        """Test ngram_size must be 1, 2, or 3."""
        # Valid sizes
        for size in [1, 2, 3]:
            cand = KeywordCandidate(
                candidate_id="test",
                source="body",
                term="test",
                count=1,
                lemma="test",
                ngram_size=size,
            )
            assert cand.ngram_size == size

        # Invalid size (> 3)
        with pytest.raises(ValidationError):
            KeywordCandidate(
                candidate_id="test",
                source="body",
                term="test",
                count=1,
                lemma="test",
                ngram_size=4,  # Max is 3
            )

    def test_embedding_score_range(self):
        """Test embedding_score must be in range 0.0-1.0."""
        # Valid scores
        for score in [0.0, 0.5, 1.0]:
            cand = KeywordCandidate(
                candidate_id="test",
                source="body",
                term="test",
                count=1,
                lemma="test",
                ngram_size=1,
                embedding_score=score,
            )
            assert cand.embedding_score == score

        # Invalid score (> 1.0)
        with pytest.raises(ValidationError):
            KeywordCandidate(
                candidate_id="test",
                source="body",
                term="test",
                count=1,
                lemma="test",
                ngram_size=1,
                embedding_score=1.5,  # Must be <= 1.0
            )

    def test_optional_fields(self):
        """Test that embedding_score and composite_score have defaults."""
        cand = KeywordCandidate(
            candidate_id="test",
            source="body",
            term="test",
            count=1,
            lemma="test",
            ngram_size=1,
            # Omit embedding_score and composite_score
        )

        assert cand.embedding_score == 0.0  # Default
        assert cand.composite_score == 0.0  # Default

    def test_model_serialization(self):
        """Test model serialization to dict and JSON."""
        cand = KeywordCandidate(
            candidate_id="abc123",
            source="subject",
            term="fattura",
            count=2,
            lemma="fattura",
            ngram_size=1,
            embedding_score=0.85,
            composite_score=0.70,
        )

        # to_dict equivalent
        cand_dict = cand.model_dump()
        assert cand_dict["candidate_id"] == "abc123"
        assert cand_dict["source"] == "subject"
        assert cand_dict["term"] == "fattura"

        # to JSON
        cand_json = cand.model_dump_json()
        assert isinstance(cand_json, str)

        # Parse back
        parsed = json.loads(cand_json)
        assert parsed["candidate_id"] == "abc123"


# ============================================================================
# Test Class: CandidateExtractionResult Model
# ============================================================================


@pytest.mark.unit
class TestCandidateExtractionResultModel:
    """Tests for CandidateExtractionResult Pydantic model."""

    def test_valid_result_creation(self):
        """Test creating valid CandidateExtractionResult."""
        result = CandidateExtractionResult(
            document_id="doc-12345",
            raw_candidates=[],
            filtered_candidates=[],
            enriched_candidates=[],
            tokenizer_version="tokenizer-1.0.0",
            stoplist_version="stopwords-it-2025.1",
            embedding_model_version="paraphrase-multilingual-mpnet-base-v2",
            ner_model_version="it_core_news_lg-3.8.0",
            total_tokens_subject=5,
            total_tokens_body=20,
            candidates_before_filter=10,
            candidates_after_filter=8,
            processing_time_ms=150.0,
        )

        assert result.document_id == "doc-12345"
        assert result.tokenizer_version == "tokenizer-1.0.0"
        assert result.total_tokens_subject == 5
        assert result.processing_time_ms == 150.0

    def test_candidate_lists_validation(self):
        """Test candidate lists are List[KeywordCandidate]."""
        cand1 = KeywordCandidate(
            candidate_id="test1",
            source="body",
            term="test",
            count=1,
            lemma="test",
            ngram_size=1,
        )

        result = CandidateExtractionResult(
            document_id="doc-123",
            raw_candidates=[cand1],
            filtered_candidates=[cand1],
            enriched_candidates=[cand1],
            tokenizer_version="v1",
            stoplist_version="v1",
            embedding_model_version="v1",
            ner_model_version="v1",
            total_tokens_subject=0,
            total_tokens_body=10,
            candidates_before_filter=1,
            candidates_after_filter=1,
            processing_time_ms=50.0,
        )

        assert len(result.raw_candidates) == 1
        assert isinstance(result.raw_candidates[0], KeywordCandidate)

    def test_version_fields_required(self):
        """Test that version fields are required (not optional)."""
        # All version fields present - valid
        result = CandidateExtractionResult(
            document_id="doc-123",
            tokenizer_version="v1",
            stoplist_version="v1",
            embedding_model_version="v1",
            ner_model_version="v1",
            total_tokens_subject=0,
            total_tokens_body=0,
            candidates_before_filter=0,
            candidates_after_filter=0,
            processing_time_ms=10.0,
        )
        assert result.tokenizer_version == "v1"

        # Missing version field should raise ValidationError
        with pytest.raises(ValidationError):
            CandidateExtractionResult(
                document_id="doc-123",
                # Missing tokenizer_version
                stoplist_version="v1",
                embedding_model_version="v1",
                ner_model_version="v1",
                total_tokens_subject=0,
                total_tokens_body=0,
                candidates_before_filter=0,
                candidates_after_filter=0,
                processing_time_ms=10.0,
            )

    def test_count_fields_non_negative(self):
        """Test count fields must be >= 0."""
        # Valid counts
        result = CandidateExtractionResult(
            document_id="doc-123",
            tokenizer_version="v1",
            stoplist_version="v1",
            embedding_model_version="v1",
            ner_model_version="v1",
            total_tokens_subject=0,
            total_tokens_body=0,
            candidates_before_filter=0,
            candidates_after_filter=0,
            processing_time_ms=0.0,
        )
        assert result.total_tokens_subject == 0

        # Negative count should raise ValidationError
        with pytest.raises(ValidationError):
            CandidateExtractionResult(
                document_id="doc-123",
                tokenizer_version="v1",
                stoplist_version="v1",
                embedding_model_version="v1",
                ner_model_version="v1",
                total_tokens_subject=-1,  # Must be >= 0
                total_tokens_body=0,
                candidates_before_filter=0,
                candidates_after_filter=0,
                processing_time_ms=0.0,
            )

    def test_processing_time_non_negative(self):
        """Test processing_time_ms must be >= 0.0."""
        # Valid time
        result = CandidateExtractionResult(
            document_id="doc-123",
            tokenizer_version="v1",
            stoplist_version="v1",
            embedding_model_version="v1",
            ner_model_version="v1",
            total_tokens_subject=0,
            total_tokens_body=0,
            candidates_before_filter=0,
            candidates_after_filter=0,
            processing_time_ms=100.5,
        )
        assert result.processing_time_ms == 100.5

        # Negative time should raise ValidationError
        with pytest.raises(ValidationError):
            CandidateExtractionResult(
                document_id="doc-123",
                tokenizer_version="v1",
                stoplist_version="v1",
                embedding_model_version="v1",
                ner_model_version="v1",
                total_tokens_subject=0,
                total_tokens_body=0,
                candidates_before_filter=0,
                candidates_after_filter=0,
                processing_time_ms=-10.0,  # Must be >= 0.0
            )

    def test_candidate_progression(self):
        """Test that typically len(raw) >= len(filtered) >= len(enriched)."""
        # Create sample candidates
        candidates = [
            KeywordCandidate(
                candidate_id=f"test{i}",
                source="body",
                term=f"term{i}",
                count=1,
                lemma=f"term{i}",
                ngram_size=1,
            )
            for i in range(3)
        ]

        result = CandidateExtractionResult(
            document_id="doc-123",
            raw_candidates=candidates[:3],  # 3 candidates
            filtered_candidates=candidates[:2],  # 2 candidates (some filtered)
            enriched_candidates=candidates[:2],  # 2 candidates (enriched)
            tokenizer_version="v1",
            stoplist_version="v1",
            embedding_model_version="v1",
            ner_model_version="v1",
            total_tokens_subject=0,
            total_tokens_body=10,
            candidates_before_filter=3,
            candidates_after_filter=2,
            processing_time_ms=50.0,
        )

        assert len(result.raw_candidates) >= len(result.filtered_candidates)
        assert len(result.filtered_candidates) >= len(result.enriched_candidates)

    def test_model_serialization(self):
        """Test CandidateExtractionResult serialization with nested candidates."""
        cand = KeywordCandidate(
            candidate_id="test1",
            source="body",
            term="fattura",
            count=2,
            lemma="fattura",
            ngram_size=1,
        )

        result = CandidateExtractionResult(
            document_id="doc-123",
            raw_candidates=[cand],
            filtered_candidates=[cand],
            enriched_candidates=[cand],
            tokenizer_version="v1",
            stoplist_version="v1",
            embedding_model_version="v1",
            ner_model_version="v1",
total_tokens_subject=0,
            total_tokens_body=10,
            candidates_before_filter=1,
            candidates_after_filter=1,
            processing_time_ms=50.0,
        )

        # to dict
        result_dict = result.model_dump()
        assert result_dict["document_id"] == "doc-123"
        assert len(result_dict["raw_candidates"]) == 1
        assert result_dict["raw_candidates"][0]["term"] == "fattura"

        # to JSON
        result_json = result.model_dump_json()
        assert isinstance(result_json, str)

        parsed = json.loads(result_json)
        assert parsed["document_id"] == "doc-123"
        assert len(parsed["raw_candidates"]) == 1
