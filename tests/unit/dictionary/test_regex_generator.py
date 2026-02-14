"""
Unit tests for regex_generator module.

Tests safe regex pattern generation from surface forms
as specified in brainstorming v2 section 5.3.
"""

import re

import pytest

from eml_classificator.dictionary.regex_generator import generate_safe_regex, validate_regex_pattern


class TestGenerateSafeRegex:
    """Test generate_safe_regex function."""

    def test_basic_pattern_generation(self):
        """Test basic regex pattern generation."""
        pattern = generate_safe_regex(["fattura"])

        assert "fattura" in pattern
        assert r"\b" in pattern  # Word boundary
        assert "(?i)" in pattern  # Case insensitive

    def test_multiple_surface_forms(self):
        """Test pattern with multiple surface forms."""
        pattern = generate_safe_regex(["fattura", "Fattura", "FATTURA"])

        # All forms should be in pattern
        assert "fattura" in pattern or "Fattura" in pattern
        assert "|" in pattern  # OR operator

        # Pattern should match all variants
        regex = re.compile(pattern)
        assert regex.search("invio fattura")
        assert regex.search("invio Fattura")
        assert regex.search("invio FATTURA")

    def test_case_sensitive_mode(self):
        """Test case sensitive pattern generation."""
        pattern = generate_safe_regex(["Fattura"], case_sensitive=True)

        assert "(?i)" not in pattern  # No case insensitive flag

        regex = re.compile(pattern)
        assert regex.search("invio Fattura")
        assert not regex.search("invio fattura")  # Should not match lowercase

    def test_case_insensitive_mode(self):
        """Test case insensitive pattern generation (default)."""
        pattern = generate_safe_regex(["fattura"], case_sensitive=False)

        assert "(?i)" in pattern

        regex = re.compile(pattern)
        assert regex.search("invio fattura")
        assert regex.search("invio Fattura")
        assert regex.search("invio FATTURA")

    def test_special_characters_escaped(self):
        """Test that special regex characters are escaped."""
        pattern = generate_safe_regex(["test.com", "test+", "test*", "test?"])

        # Pattern should compile without errors
        regex = re.compile(pattern)

        # Should match literal characters, not regex operators
        assert regex.search("test.com")
        assert not regex.search("test_com")  # . should not match any char

    def test_word_boundaries(self):
        """Test word boundary matching."""
        pattern = generate_safe_regex(["test"])
        regex = re.compile(pattern)

        # Should match word "test"
        assert regex.search("a test here")
        assert regex.search("test here")
        assert regex.search("here test")

        # Should NOT match partial words
        assert not regex.search("testing")
        assert not regex.search("attest")
        assert not regex.search("contest")

    def test_empty_surface_forms(self):
        """Test behavior with empty surface forms list."""
        pattern = generate_safe_regex([])

        # Should return pattern that never matches
        regex = re.compile(pattern)
        assert not regex.search("any text here")

    def test_unicode_characters(self):
        """Test pattern with Unicode characters."""
        pattern = generate_safe_regex(["città", "perché"])
        regex = re.compile(pattern)

        assert regex.search("la città di Roma")
        assert regex.search("perché è importante")

    def test_apostrophe_handling(self):
        """Test Italian words with apostrophes."""
        pattern = generate_safe_regex(["l'email", "un'altra"])
        regex = re.compile(pattern)

        assert regex.search("inviare l'email")
        assert regex.search("un'altra volta")

    def test_pattern_validation(self):
        """Test that invalid patterns raise errors."""
        # generate_safe_regex should escape, so it shouldn't create invalid patterns
        # But we can test with malformed input
        pattern = generate_safe_regex(["valid"])
        # Should not raise
        re.compile(pattern)

    def test_multiple_matches_in_text(self):
        """Test pattern matching multiple occurrences."""
        pattern = generate_safe_regex(["test"])
        regex = re.compile(pattern)

        text = "test one, test two, test three"
        matches = regex.findall(text)

        assert len(matches) == 3
        assert all(m == "test" for m in matches)


class TestValidateRegexPattern:
    """Test validate_regex_pattern utility function."""

    def test_basic_testing(self):
        """Test basic pattern testing functionality."""
        pattern = generate_safe_regex(["fattura"])
        test_texts = ["invio fattura", "no match", "altra fattura qui"]

        results = validate_regex_pattern(pattern, test_texts)

        assert len(results) == 3
        assert len(results[0]) == 1  # One match in first text
        assert len(results[1]) == 0  # No match in second text
        assert len(results[2]) == 1  # One match in third text

    def test_multiple_matches_per_text(self):
        """Test pattern with multiple matches per text."""
        pattern = generate_safe_regex(["test"])
        test_texts = ["test one test two"]

        results = validate_regex_pattern(pattern, test_texts)

        assert len(results[0]) == 2

    def test_invalid_pattern(self):
        """Test behavior with invalid pattern."""
        # Create invalid pattern manually (not through generate_safe_regex)
        invalid_pattern = r"(?P<unclosed"

        results = validate_regex_pattern(invalid_pattern, ["any text"])

        # Should return empty results, not crash
        assert results == [[]]

    def test_empty_test_texts(self):
        """Test with empty test texts list."""
        pattern = generate_safe_regex(["test"])
        results = validate_regex_pattern(pattern, [])

        assert results == []


class TestRegexIntegration:
    """Integration tests for regex generation and matching."""

    def test_invoice_number_pattern(self):
        """Test pattern for invoice-related terms."""
        pattern = generate_safe_regex(["fattura", "Fattura", "FATTURA"])
        regex = re.compile(pattern)

        test_cases = [
            ("Invio fattura n. 12345", True),
            ("Fattura elettronica", True),
            ("FATTURA PROFORMA", True),
            ("fatturazione", False),  # Should not match (longer word)
            ("infatturato", False),  # Should not match (partial)
        ]

        for text, should_match in test_cases:
            result = bool(regex.search(text))
            assert (
                result == should_match
            ), f"Pattern match failed for '{text}': expected {should_match}, got {result}"

    def test_multiple_terms_pattern(self):
        """Test pattern with multiple different terms."""
        terms = ["problema", "errore", "guasto", "assistenza"]
        pattern = generate_safe_regex(terms)
        regex = re.compile(pattern)

        assert regex.search("Ho un problema")
        assert regex.search("Errore nel sistema")
        assert regex.search("Guasto dell'impianto")
        assert regex.search("Richiedo assistenza")

    def test_email_domain_pattern(self):
        """Test pattern for email domain matching."""
        pattern = generate_safe_regex(["@company.com", "@client.it"])
        regex = re.compile(pattern)

        assert regex.search("user@company.com")
        assert regex.search("test@client.it")
        assert not regex.search("user@other.com")
