"""
Unit tests for candidate builder (Phase 2 - Candidate Generation).

Tests coverage:
- build_candidates: Extract candidates from EmailDocument with various scenarios
- Source tracking (subject vs body)
- Term counting and deduplication
- Lemmatization integration
- Deterministic sorting
- Min term length filtering
- Max n-gram handling
- Edge cases

Testing principles:
1. Test with EmailDocument fixtures
2. Verify deterministic behavior
3. Test Italian text handling
4. Cover edge cases (empty documents, long text, special chars)
"""

import pytest
from datetime import datetime
from eml_classificator.candidates.candidate_builder import build_candidates, MIN_TERM_LENGTH
from eml_classificator.models.email_document import EmailDocument, EmailHeaders
from eml_classificator.models.pipeline_version import PipelineVersion


def create_simple_document(subject: str = "", body: str = "") -> EmailDocument:
    """Helper to create EmailDocument for testing."""
    return EmailDocument(
        document_id="test-doc-id",
        headers=EmailHeaders(
            from_address="test@example.com",
            to_addresses=["support@company.com"],
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
        raw_size_bytes=1024,  # Added missing field
        encoding_detected="utf-8",  # Added missing field
    )


@pytest.mark.unit
@pytest.mark.slow
class TestBuildCandidates:
    """Tests for build_candidates function."""

    def test_simple_subject_only(self):
        """Test candidate extraction from subject only."""
        doc = create_simple_document(subject="Fattura urgente", body="")
        candidates = build_candidates(doc)

        # Check that we have candidates
        assert len(candidates) > 0

        # Check source field
        source_values = [c.source for c in candidates]
        assert all(s == "subject" for s in source_values)

        # Check terms exist
        terms = [c.term for c in candidates]
        assert "fattura" in terms
        assert "urgente" in terms

    def test_simple_body_only(self):
        """Test candidate extraction from body only."""
        doc = create_simple_document(subject="", body="Il cliente ha un problema")
        candidates = build_candidates(doc)

        # Check that we have candidates
        assert len(candidates) > 0

        # All should be from body
        assert all(c.source == "body" for c in candidates)

        # Check terms
        terms = [c.term for c in candidates]
        assert "cliente" in terms
        assert "problema" in terms

    def test_subject_and_body_combined(self):
        """Test candidates from both subject and body with proper source tracking."""
        doc = create_simple_document(
            subject="Fattura urgente", body="Il cliente ha un problema"
        )
        candidates = build_candidates(doc)

        # Check we have candidates from both sources
        sources = {c.source for c in candidates}
        assert "subject" in sources
        assert "body" in sources

        # Subject candidates should appear first (deterministic sorting)
        subject_candidates = [c for c in candidates if c.source == "subject"]
        body_candidates = [c for c in candidates if c.source == "body"]
        assert len(subject_candidates) > 0
        assert len(body_candidates) > 0

        # First candidate should be from subject (due to sorting)
        assert candidates[0].source == "subject"

    def test_empty_subject_and_body(self):
        """Test empty email returns empty candidate list."""
        doc = create_simple_document(subject="", body="")
        candidates = build_candidates(doc)
        assert candidates == []

    def test_only_subject_empty(self):
        """Test with empty subject but populated body."""
        doc = create_simple_document(subject="", body="Il cliente ha un problema")
        candidates = build_candidates(doc)

        # All candidates from body
        assert len(candidates) > 0
        assert all(c.source == "body" for c in candidates)

    def test_only_body_empty(self):
        """Test with populated subject but empty body."""
        doc = create_simple_document(subject="Fattura urgente", body="")
        candidates = build_candidates(doc)

        # All candidates from subject
        assert len(candidates) > 0
        assert all(c.source == "subject" for c in candidates)

    def test_italian_apostrophes(self):
        """Test Italian text with apostrophes (dell'ordine)."""
        doc = create_simple_document(
            subject="", body="dell'ordine è già stato processato"
        )
        candidates = build_candidates(doc)

        terms = [c.term for c in candidates]
        assert "dell'ordine" in terms or any("dell'ordine" in t for t in terms)

    def test_term_counting(self):
        """Test that repeated terms are counted correctly."""
        doc = create_simple_document(
            subject="", body="fattura fattura fattura pagamento"
        )
        candidates = build_candidates(doc)

        # Find fattura candidate
        fattura_candidates = [c for c in candidates if c.term == "fattura"]
        assert len(fattura_candidates) == 1
        assert fattura_candidates[0].count == 3

        # Find pagamento candidate
        pagamento_candidates = [c for c in candidates if c.term == "pagamento"]
        assert len(pagamento_candidates) == 1
        assert pagamento_candidates[0].count == 1

    def test_source_tracking(self):
        """Test that source field correctly marks subject vs body."""
        doc = create_simple_document(subject="urgente", body="problema")
        candidates = build_candidates(doc)

        urgente_cand = [c for c in candidates if c.term == "urgente"]
        assert len(urgente_cand) == 1
        assert urgente_cand[0].source == "subject"

        problema_cand = [c for c in candidates if c.term == "problema"]
        assert len(problema_cand) == 1
        assert problema_cand[0].source == "body"

    def test_lemmatization_applied(self):
        """Test that lemmatization is applied to candidates."""
        doc = create_simple_document(subject="", body="clienti problemi")
        candidates = build_candidates(doc)

        # Check that lemmas are populated (even if same as term)
        for cand in candidates:
            assert cand.lemma is not None
            assert len(cand.lemma) > 0

    def test_min_term_length_filter(self):
        """Test minimum term length filtering."""
        doc = create_simple_document(subject="", body="a il un problema")
        candidates = build_candidates(doc, min_term_length=MIN_TERM_LENGTH)

        terms = [c.term for c in candidates]
        # Short words should be filtered
        assert "a" not in terms
        assert "il" not in terms
        assert "un" not in terms
        # Long word should remain
        assert "problema" in terms

    def test_max_ngram_respected(self):
        """Test that max_ngram parameter is respected."""
        doc = create_simple_document(subject="", body="il cliente ha un problema urgente")
        candidates = build_candidates(doc, max_ngram=2)

        # Should have unigrams and bigrams only
        ngram_sizes = {c.ngram_size for c in candidates}
        assert 1 in ngram_sizes
        assert 2 in ngram_sizes
        assert 3 not in ngram_sizes  # No trigrams

    def test_deterministic_sorting(self):
        """Test that same input produces same sorted order."""
        doc = create_simple_document(
            subject="Fattura urgente", body="Il cliente ha un problema"
        )

        candidates1 = build_candidates(doc)
        candidates2 = build_candidates(doc)

        # Same order
        assert len(candidates1) == len(candidates2)
        for c1, c2 in zip(candidates1, candidates2):
            assert c1.term == c2.term
            assert c1.source == c2.source
            assert c1.count == c2.count

    def test_stable_candidate_ids(self):
        """Test that same source + term produces same candidate_id across runs."""
        doc = create_simple_document(subject="Fattura urgente", body="")

        candidates1 = build_candidates(doc)
        candidates2 = build_candidates(doc)

        # Find matching candidates
        for c1 in candidates1:
            matching = [c2 for c2 in candidates2 if c2.term == c1.term and c2.source == c1.source]
            assert len(matching) == 1
            assert c1.candidate_id == matching[0].candidate_id

    def test_metadata_populated(self):
        """Test that all candidate fields are populated."""
        doc = create_simple_document(subject="Fattura urgente", body="")
        candidates = build_candidates(doc)

        for cand in candidates:
            assert cand.candidate_id is not None
            assert len(cand.candidate_id) == 12
            assert cand.source in ["subject", "body"]
            assert cand.term is not None
            assert cand.count >= 1
            assert cand.lemma is not None
            assert cand.ngram_size in [1, 2, 3]
            assert cand.embedding_score == 0.0  # Not enriched yet
            assert cand.composite_score == 0.0  # Not calculated yet

    # Edge Cases

    def test_very_long_text(self):
        """Test handling of very long text."""
        long_body = " ".join(["Il cliente ha un problema con la fattura"] * 500)
        doc = create_simple_document(subject="", body=long_body)

        candidates = build_candidates(doc)

        # Should handle without crashing
        assert len(candidates) > 0

        # High-frequency terms should have high counts
        cliente_cands = [c for c in candidates if c.term == "cliente"]
        if cliente_cands:
            assert cliente_cands[0].count >= 100

    def test_special_characters(self):
        """Test handling of special characters (€, $, #)."""
        doc = create_simple_document(subject="", body="€100 $50 #urgente")
        candidates = build_candidates(doc)

        # Should extract numeric parts
        terms = [c.term for c in candidates]
        assert "100" in terms or "50" in terms

    def test_numbers_in_text(self):
        """Test that numbers are extracted as candidates."""
        doc = create_simple_document(subject="Fattura 12345", body="")
        candidates = build_candidates(doc)

        terms = [c.term for c in candidates]
        assert "12345" in terms

    def test_mixed_languages(self):
        """Test mixed Italian and English text."""
        doc = create_simple_document(subject="", body="Il cliente urgent problem")
        candidates = build_candidates(doc)

        # Should tokenize both languages
        terms = [c.term for c in candidates]
        assert "cliente" in terms
        assert "urgent" in terms
        assert "problem" in terms

    def test_repeated_punctuation(self):
        """Test handling of repeated punctuation."""
        doc = create_simple_document(subject="Urgente!!!", body="")
        candidates = build_candidates(doc)

        terms = [c.term for c in candidates]
        assert "urgente" in terms
        # Punctuation should be stripped
        assert "!" not in terms
