"""
Unit tests for Italian tokenizer (Phase 2 - Candidate Generation).

Tests coverage:
- tokenize_it: Italian text tokenization with apostrophes,accents, numbers, edge cases
- lemmatize_term: spaCy-based lemmatization for Italian
- ngrams: N-gram generation (unigrams, bigrams, trigrams)
- stable_id: Deterministic ID generation
- get_spacy_model: Model loading and singleton pattern

Testing principles:
1. Determinism: Same input → same output across runs
2. Italian support: Apostrophes (dell'ordine), accents (città, però)
3. Edge cases: Empty input, special characters, long text
"""

import pytest
from eml_classificator.candidates.tokenizer import (
    tokenize_it,
    lemmatize_term,
    ngrams,
    stable_id,
    get_spacy_model,
    TOKENIZER_VERSION,
    NER_MODEL_VERSION,
)


# ============================================================================
# Test Class: TokenizeIt
# ============================================================================


@pytest.mark.unit
class TestTokenizeIt:
    """Tests for tokenize_it function."""

    def test_simple_italian_text(self):
        """Test tokenization of simple Italian sentence."""
        text = "Il cliente ha un problema"
        result = tokenize_it(text)
        assert result == ["il", "cliente", "ha", "un", "problema"]

    def test_apostrophes(self):
        """Test that Italian apostrophes are preserved (dell'ordine)."""
        text = "dell'ordine"
        result = tokenize_it(text)
        assert result == ["dell'ordine"]
        assert "dell" not in result  # Not split
        assert "'ordine" not in result

    def test_multiple_apostrophes(self):
        """Test multiple words with apostrophes."""
        text = "dell'ordine dell'utente l'azienda"
        result = tokenize_it(text)
        assert "dell'ordine" in result
        assert "dell'utente" in result
        assert "l'azienda" in result

    def test_accented_characters(self):
        """Test Italian accented characters (àèéìòù) are preserved."""
        text = "città però caffè"
        result = tokenize_it(text)
        assert "città" in result
        assert "però" in result
        assert "caffè" in result

    def test_mixed_case_lowercased(self):
        """Test that mixed case text is lowercased."""
        text = "URGENTE Fattura"
        result = tokenize_it(text)
        assert result == ["urgente", "fattura"]

    def test_numbers_preserved(self):
        """Test that numbers are preserved as tokens."""
        text = "Fattura 12345 del 2025"
        result = tokenize_it(text)
        assert "fattura" in result
        assert "12345" in result
        assert "del" in result
        assert "2025" in result

    def test_punctuation_removed(self):
        """Test that punctuation is removed, only words/numbers remain."""
        text = "Salve, come va?"
        result = tokenize_it(text)
        assert result == ["salve", "come", "va"]
        assert "," not in result
        assert "?" not in result

    def test_empty_string(self):
        """Test empty string returns empty list."""
        result = tokenize_it("")
        assert result == []

    def test_only_punctuation(self):
        """Test string with only punctuation returns empty list."""
        result = tokenize_it("!!!")
        assert result == []

    def test_multiple_spaces(self):
        """Test that multiple spaces are handled correctly."""
        text = "Il   cliente   "
        result = tokenize_it(text)
        assert result == ["il", "cliente"]

    def test_determinism(self):
        """Test that same input produces same output (determinism)."""
        text = "Il cliente ha un problema"
        result1 = tokenize_it(text)
        result2 = tokenize_it(text)
        assert result1 == result2


# ============================================================================
# Test Class: LemmatizeTerm
# ============================================================================


@pytest.mark.unit
@pytest.mark.slow
class TestLemmatizeTerm:
    """Tests for lemmatize_term function (requires spaCy model)."""

    def test_italian_nouns_plural_to_singular(self):
        """Test Italian noun lemmatization: plural → singular."""
        assert lemmatize_term("fatture") == "fattura"
        assert lemmatize_term("clienti") == "cliente"
        assert lemmatize_term("problemi") == "problema"
        assert lemmatize_term("ordini") == "ordine"

    def test_italian_adjectives(self):
        """Test Italian adjective lemmatization."""
        assert lemmatize_term("urgenti") == "urgente"
        assert lemmatize_term("importanti") == "importante"

    def test_italian_verbs(self):
        """Test Italian verb lemmatization to infinitive."""
        # Note: spaCy lemmatization might vary; these are common cases
        result = lemmatize_term("ordinare")
        assert "ordina" in result or "ordinare" in result

    def test_accented_lemmas_preserved(self):
        """Test that accented characters are preserved in lemmas."""
        result = lemmatize_term("città")
        assert "città" in result or "città" == result  # Should preserve accent

    def test_apostrophes_in_lemma(self):
        """Test lemmatization of terms with apostrophes."""
        result = lemmatize_term("dell'ordine")
        assert result is not None
        assert len(result) > 0

    def test_numbers_unchanged(self):
        """Test that numbers are not lemmatized (returned as-is)."""
        result = lemmatize_term("12345")
        assert "12345" in result

    def test_determinism(self):
        """Test that same term lemmatized twice produces same result."""
        term = "clienti"
        result1 = lemmatize_term(term)
        result2 = lemmatize_term(term)
        assert result1 == result2

    def test_batch_lemmatization(self):
        """Test lemmatization of multiple terms."""
        terms = ["fatture", "clienti", "problemi"]
        lemmas = [lemmatize_term(t) for t in terms]
        assert len(lemmas) == 3
        assert all(len(l) > 0 for l in lemmas)


# ============================================================================
# Test Class: Ngrams
# ============================================================================


@pytest.mark.unit
class TestNgrams:
    """Tests for ngrams function."""

    def test_unigrams(self):
        """Test unigram generation (n=1)."""
        tokens = ["il", "cliente", "ha"]
        result = ngrams(tokens, 1)
        assert result == ["il", "cliente", "ha"]

    def test_bigrams(self):
        """Test bigram generation (n=2)."""
        tokens = ["il", "cliente", "ha"]
        result = ngrams(tokens, 2)
        assert result == ["il cliente", "cliente ha"]

    def test_trigrams(self):
        """Test trigram generation (n=3)."""
        tokens = ["il", "cliente", "ha", "un"]
        result = ngrams(tokens, 3)
        assert result == ["il cliente ha", "cliente ha un"]

    def test_single_token_no_bigrams(self):
        """Test that single token produces no bigrams or trigrams."""
        tokens = ["ciao"]
        assert ngrams(tokens, 1) == ["ciao"]
        assert ngrams(tokens, 2) == []
        assert ngrams(tokens, 3) == []

    def test_two_tokens_no_trigrams(self):
        """Test that two tokens produce bigram but no trigrams."""
        tokens = ["ciao", "marco"]
        assert ngrams(tokens, 2) == ["ciao marco"]
        assert ngrams(tokens, 3) == []

    def test_empty_list(self):
        """Test empty token list returns empty n-grams."""
        assert ngrams([], 1) == []
        assert ngrams([], 2) == []
        assert ngrams([], 3) == []

    def test_determinism(self):
        """Test that same tokens produce same n-grams."""
        tokens = ["il", "cliente", "ha"]
        result1 = ngrams(tokens, 2)
        result2 = ngrams(tokens, 2)
        assert result1 == result2


# ============================================================================
# Test Class: StableId
# ============================================================================


@pytest.mark.unit
class TestStableId:
    """Tests for stable_id function (deterministic ID generation)."""

    def test_basic_id_generation(self):
        """Test that stable_id generates 12-char hex string."""
        result = stable_id("subject", "fattura")
        assert isinstance(result, str)
        assert len(result) == 12
        assert all(c in "0123456789abcdef" for c in result)

    def test_different_inputs_different_ids(self):
        """Test that different inputs produce different IDs."""
        id1 = stable_id("subject", "fattura")
        id2 = stable_id("body", "pagamento")
        assert id1 != id2

    def test_same_inputs_same_id(self):
        """Test determinism: same inputs produce same ID."""
        id1 = stable_id("subject", "fattura")
        id2 = stable_id("subject", "fattura")
        assert id1 == id2

    def test_order_matters(self):
        """Test that part order affects ID."""
        id1 = stable_id("A", "B")
        id2 = stable_id("B", "A")
        assert id1 != id2

    def test_utf8_handling(self):
        """Test UTF-8 characters (Italian accents) in ID generation."""
        result = stable_id("subject", "città")
        assert isinstance(result, str)
        assert len(result) == 12

    def test_empty_parts(self):
        """Test that empty parts don't crash (produce valid ID)."""
        result = stable_id("", "")
        assert isinstance(result, str)
        assert len(result) == 12


# ============================================================================
# Test Class: SpacyModel
# ============================================================================


@pytest.mark.unit
@pytest.mark.slow
class TestSpacyModel:
    """Tests for spaCy model loading and singleton pattern."""

    def test_model_loads_successfully(self):
        """Test that get_spacy_model returns Language object."""
        nlp = get_spacy_model()
        assert nlp is not None
        assert hasattr(nlp, "__call__")  # Can process text

    def test_singleton_pattern(self):
        """Test that multiple calls return same model instance."""
        nlp1 = get_spacy_model()
        nlp2 = get_spacy_model()
        assert nlp1 is nlp2  # Same object in memory

    def test_italian_model_components(self):
        """Test that model has expected Italian pipeline components."""
        nlp = get_spacy_model()
        # Check for common spaCy components
        assert "tagger" in nlp.pipe_names or "morphologizer" in nlp.pipe_names
        assert nlp.lang == "it"  # Italian model


# ============================================================================
# Test: Version Constants
# ============================================================================


@pytest.mark.unit
def test_version_constants_defined():
    """Test that version constants are defined for audit trail."""
    assert TOKENIZER_VERSION is not None
    assert isinstance(TOKENIZER_VERSION, str)
    assert len(TOKENIZER_VERSION) > 0

    assert NER_MODEL_VERSION is not None
    assert isinstance(NER_MODEL_VERSION, str)
    assert "it_core_news" in NER_MODEL_VERSION.lower()
