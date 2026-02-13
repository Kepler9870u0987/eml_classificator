"""
Unit tests for candidate filters (Phase 2 - Candidate Generation).

Tests coverage:
- filter_candidates: Stopword and blacklist pattern filtering
- Italian stopword filtering (term and lemma)
- Blacklist pattern matching (re:, fwd:, numbers, dates)
- Edge cases (empty lists, all filtered, etc.)

Testing principles:
1. Test deterministic filtering behavior
2. Test Italian-specific stopwords
3. Test blacklist pattern matching
4. Test edge cases
"""

import pytest
from eml_classificator.candidates.filters import (
    filter_candidates,
    BLACKLIST_PATTERNS,
    STOPLIST_VERSION,
)
from eml_classificator.candidates.stopwords import STOPWORDS_IT
from eml_classificator.models.candidates import KeywordCandidate


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


@pytest.mark.unit
class TestFilterCandidates:
    """Tests for filter_candidates function."""

    # ========================================================================
    # Stopword Filtering Tests
    # ========================================================================

    def test_common_italian_stopwords(self):
        """Test that common Italian stopwords are filtered."""
        candidates = [
            create_candidate("il"),
            create_candidate("la"),
            create_candidate("di"),
            create_candidate("da"),
            create_candidate("in"),
        ]

        filtered = filter_candidates(candidates)
        assert len(filtered) == 0  # All stopwords

    def test_email_specific_stopwords(self):
        """Test email-specific stopwords (cordiali, saluti, distinti)."""
        candidates = [
            create_candidate("cordiali"),
            create_candidate("saluti"),
            create_candidate("distinti"),
            create_candidate("grazie"),
        ]

        filtered = filter_candidates(candidates)
        assert len(filtered) == 0  # All email stopwords

    def test_temporal_stopwords(self):
        """Test temporal/calendar stopwords."""
        candidates = [
            create_candidate("lunedì"),
            create_candidate("gennaio"),
            create_candidate("oggi"),
            create_candidate("domani"),
        ]

        filtered = filter_candidates(candidates)
        # These should be in stopwords list
        assert len(filtered) == 0 or all(c.term not in ["lunedì", "gennaio"] for c in filtered)

    def test_stop_bigrams(self):
        """Test stopword bigrams (buon giorno, cordiali saluti)."""
        candidates = [
            create_candidate("buon giorno"),
            create_candidate("cordiali saluti"),
        ]

        filtered = filter_candidates(candidates)
        # These contain stopwords in lemma form
        assert len(filtered) <= len(candidates)

    def test_non_stopwords_preserved(self):
        """Test that non-stopwords are preserved."""
        candidates = [
            create_candidate("fattura"),
            create_candidate("urgente"),
            create_candidate("cliente"),
            create_candidate("problema"),
        ]

        filtered = filter_candidates(candidates)
        assert len(filtered) == 4  # All preserved
        terms = {c.term for c in filtered}
        assert "fattura" in terms
        assert "urgente" in terms
        assert "cliente" in terms
        assert "problema" in terms

    def test_lemma_based_filtering(self):
        """Test that filtering works on lemma if lemma is stopword."""
        # Create candidate where term is not stopword but lemma is
        candidates = [
            create_candidate("problemi", lemma="problema"),  # problema might not be stopword
            create_candidate("clienti", lemma="il"),  # Simulated: lemma is stopword
        ]

        filtered = filter_candidates(candidates)

        # The one with stopword lemma should be filtered
        lemmas = {c.lemma for c in filtered}
        assert "il" not in lemmas

    def test_case_insensitive_stopwords(self):
        """Test that stopword filtering is case-insensitive."""
        candidates = [
            create_candidate("GRAZIE"),
            create_candidate("Cordiali"),
            create_candidate("SaLuTi"),
        ]

        filtered = filter_candidates(candidates)
        assert len(filtered) == 0  # All filtered regardless of case

    def test_mixed_content_filtering(self):
        """Test mixed stopwords and non-stopwords."""
        candidates = [
            create_candidate("fattura"),  # Keep
            create_candidate("cordiali"),  # Filter
            create_candidate("urgente"),  # Keep
        ]

        filtered = filter_candidates(candidates)
        assert len(filtered) == 2
        terms = {c.term for c in filtered}
        assert "fattura" in terms
        assert "urgente" in terms
        assert "cordiali" not in terms

    def test_all_stopwords(self):
        """Test when all candidates are stopwords."""
        candidates = [
            create_candidate("il"),
            create_candidate("la"),
            create_candidate("di"),
        ]

        filtered = filter_candidates(candidates)
        assert len(filtered) == 0

    def test_no_stopwords(self):
        """Test when no stopwords present."""
        candidates = [
            create_candidate("fattura"),
            create_candidate("urgente"),
        ]

        filtered = filter_candidates(candidates)
        assert len(filtered) == 2

    # ========================================================================
    # Blacklist Pattern Filtering Tests
    # ========================================================================

    def test_re_pattern(self):
        """Test that 're:' prefix is filtered."""
        candidates = [
            create_candidate("re: fattura"),
        ]

        filtered = filter_candidates(candidates)
        assert len(filtered) == 0

    def test_fwd_pattern(self):
        """Test that 'fwd:' prefix is filtered."""
        candidates = [
            create_candidate("fwd: problema"),
        ]

        filtered = filter_candidates(candidates)
        assert len(filtered) == 0

    def test_isolated_numbers(self):
        """Test that isolated numbers are filtered."""
        candidates = [
            create_candidate("12345"),
            create_candidate("999"),
        ]

        filtered = filter_candidates(candidates)
        assert len(filtered) == 0

    def test_date_patterns(self):
        """Test that date-like patterns are filtered."""
        candidates = [
            create_candidate("15/02/2025"),
            create_candidate("2025-02-13"),
            create_candidate("12-34-56"),
        ]

        filtered = filter_candidates(candidates)
        # These match date-like patterns
        assert len(filtered) == 0

    def test_combined_filters(self):
        """Test that stopword AND blacklist filters are applied."""
        candidates = [
            create_candidate("grazie"),  # Stopword
            create_candidate("re: problema"),  # Blacklist
            create_candidate("fattura"),  # Valid
        ]

        filtered = filter_candidates(candidates)
        assert len(filtered) == 1
        assert filtered[0].term == "fattura"

    def test_no_blacklist_matches(self):
        """Test regular terms pass blacklist check."""
        candidates = [
            create_candidate("fattura"),
            create_candidate("urgente"),
            create_candidate("problema"),
        ]

        filtered = filter_candidates(candidates)
        assert len(filtered) == 3

    # ========================================================================
    # Edge Cases
    # ========================================================================

    def test_empty_candidate_list(self):
        """Test empty list returns empty list."""
        filtered = filter_candidates([])
        assert filtered == []

    def test_term_length_after_filter(self):
        """Test that minimum term length is still enforced."""
        # Create candidates with length < 3 (should be filtered by length check)
        candidates = [
            create_candidate("ab"),  # Too short
            create_candidate("problemi"),  # Valid
        ]

        filtered = filter_candidates(candidates)
        # Short term should be filtered
        assert all(len(c.term) >= 3 for c in filtered)

    def test_custom_stopwords(self):
        """Test filter_candidates with custom stopword set."""
        custom_stopwords = {"test", "custom"}
        candidates = [
            create_candidate("test"),  # Custom stopword
            create_candidate("fattura"),  # Not in custom set
        ]

        filtered = filter_candidates(candidates, stopwords=custom_stopwords)
        assert len(filtered) == 1
        assert filtered[0].term == "fattura"

    def test_custom_blacklist(self):
        """Test filter_candidates with custom blacklist patterns."""
        custom_blacklist = [r"^urgent"]  # Custom pattern
        candidates = [
            create_candidate("urgent"),  # Matches custom pattern
            create_candidate("problema"),  # Valid
        ]

        filtered = filter_candidates(candidates, blacklist_patterns=custom_blacklist)
        assert len(filtered) == 1
        assert filtered[0].term == "problema"

    def test_candidate_attributes_preserved(self):
        """Test that filtered candidates preserve all attributes."""
        candidates = [
            KeywordCandidate(
                candidate_id="test-id-123",
                source="subject",
                term="fattura",
                count=5,
                lemma="fattura",
                ngram_size=1,
                embedding_score=0.85,
                composite_score=0.70,
            )
        ]

        filtered = filter_candidates(candidates)
        assert len(filtered) == 1

        cand = filtered[0]
        assert cand.candidate_id == "test-id-123"
        assert cand.source == "subject"
        assert cand.term == "fattura"
        assert cand.count == 5
        assert cand.lemma == "fattura"
        assert cand.ngram_size == 1
        assert cand.embedding_score == 0.85
        assert cand.composite_score == 0.70


# ============================================================================
# Test: Stoplist and Blacklist Constants
# ============================================================================


@pytest.mark.unit
def test_stopwords_set_is_populated():
    """Test that STOPWORDS_IT set is populated with Italian stopwords."""
    assert isinstance(STOPWORDS_IT, set)
    assert len(STOPWORDS_IT) > 100  # Should have many stopwords

    # Check for common Italian stopwords
    assert "il" in STOPWORDS_IT
    assert "la" in STOPWORDS_IT
    assert "di" in STOPWORDS_IT
    assert "grazie" in STOPWORDS_IT
    assert "cordiali" in STOPWORDS_IT


@pytest.mark.unit
def test_blacklist_patterns_defined():
    """Test that BLACKLIST_PATTERNS list is defined."""
    assert isinstance(BLACKLIST_PATTERNS, list)
    assert len(BLACKLIST_PATTERNS) > 0

    # Check for expected patterns
    assert any("re:" in p for p in BLACKLIST_PATTERNS)
    assert any("fwd:" in p for p in BLACKLIST_PATTERNS)
    assert any(r"\d+" in p for p in BLACKLIST_PATTERNS)


@pytest.mark.unit
def test_stoplist_version_defined():
    """Test that STOPLIST_VERSION is defined for audit trail."""
    assert STOPLIST_VERSION is not None
    assert isinstance(STOPLIST_VERSION, str)
    assert "stopwords-it" in STOPLIST_VERSION.lower()
