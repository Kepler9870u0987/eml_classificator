"""
Unit tests for text normalization module (text_normalizer.py).

Tests cover:
- Complete canonicalization pipeline
- Whitespace normalization
- URL removal
- Email signature detection and removal (Italian patterns)
- Reply chain detection and removal
- Excessive newline handling
"""

import pytest

from eml_classificator.canonicalization.text_normalizer import (
    canonicalize_text,
    normalize_whitespace,
    remove_email_signatures,
    remove_excessive_newlines,
    remove_reply_chains,
    remove_urls,
)


class TestCanonicalizeText:
    """Tests for complete canonicalize_text() pipeline."""

    @pytest.mark.unit
    def test_canonicalize_text_complete_pipeline(self):
        """Test full canonicalization pipeline end-to-end."""
        text = """Hello,   this   is   a test.


Multiple    blank   lines above.

Cordiali saluti,
Mario Rossi
"""
        result = canonicalize_text(text)

        # Should be lowercased
        assert result.islower()
        # Should not contain signature
        assert "cordiali saluti" not in result
        assert "mario rossi" not in result
        # Should have normalized whitespace
        assert "  " not in result
        # Should be stripped
        assert not result.startswith(" ")
        assert not result.endswith(" ")

    @pytest.mark.unit
    def test_canonicalize_text_with_signature_removal_disabled(self):
        """Test canonicalization with signature removal disabled."""
        text = "Test\n\nCordiali saluti,\nMario"
        result = canonicalize_text(text, remove_signatures=False)

        # Signature should still be present (lowercased)
        assert "cordiali saluti" in result

    @pytest.mark.unit
    def test_canonicalize_text_empty_string(self):
        """Test handling of empty string."""
        assert canonicalize_text("") == ""
        assert canonicalize_text("   ") == ""


class TestNormalizeWhitespace:
    """Tests for normalize_whitespace() function."""

    @pytest.mark.unit
    def test_normalize_whitespace_multiple_spaces(self):
        """Test collapsing multiple spaces to single space."""
        text = "Hello    world    test"
        result = normalize_whitespace(text)

        assert result == "Hello world test"
        assert "  " not in result

    @pytest.mark.unit
    def test_normalize_whitespace_line_endings(self):
        """Test normalizing \\r\\n to \\n."""
        text = "Line 1\r\nLine 2\rLine 3\nLine 4"
        result = normalize_whitespace(text)

        assert "\r" not in result
        assert result == "Line 1\nLine 2\nLine 3\nLine 4"

    @pytest.mark.unit
    def test_normalize_whitespace_trailing(self):
        """Test removing trailing whitespace from lines."""
        text = "Line 1   \nLine 2\t\t\nLine 3"
        result = normalize_whitespace(text)

        assert result == "Line 1\nLine 2\nLine 3"

    @pytest.mark.unit
    def test_normalize_whitespace_excessive_newlines(self):
        """Test collapsing multiple blank lines to max 2."""
        text = "Line 1\n\n\n\n\nLine 2"
        result = normalize_whitespace(text)

        assert result == "Line 1\n\nLine 2"

    @pytest.mark.unit
    def test_normalize_whitespace_preserves_indentation(self):
        """Test that line start indentation is preserved."""
        text = "  Indented line\n    More indented"
        result = normalize_whitespace(text)

        # Should still have indentation at start
        assert result.startswith("  ")


class TestRemoveUrls:
    """Tests for remove_urls() function."""

    @pytest.mark.unit
    def test_remove_urls_http(self):
        """Test replacing http:// URLs."""
        text = "Visit http://example.com for more info"
        result = remove_urls(text)

        assert "http://example.com" not in result
        assert "[URL]" in result

    @pytest.mark.unit
    def test_remove_urls_https(self):
        """Test replacing https:// URLs."""
        text = "Secure site: https://secure.example.com/path"
        result = remove_urls(text)

        assert "https://secure.example.com" not in result
        assert "[URL]" in result

    @pytest.mark.unit
    def test_remove_urls_multiple(self):
        """Test replacing multiple URLs."""
        text = "Links: http://one.com and https://two.com/path"
        result = remove_urls(text)

        assert result.count("[URL]") == 2
        assert "http://" not in result
        assert "https://" not in result

    @pytest.mark.unit
    def test_remove_urls_custom_replacement(self):
        """Test custom placeholder for URLs."""
        text = "Visit https://example.com"
        result = remove_urls(text, replacement="<link>")

        assert "<link>" in result
        assert "[URL]" not in result


class TestRemoveEmailSignatures:
    """Tests for remove_email_signatures() function."""

    @pytest.mark.unit
    def test_remove_email_signatures_standard_delimiter(self):
        """Test detection of standard '--' signature delimiter."""
        text = """Email body content.

--
John Doe
CEO"""
        result, detected = remove_email_signatures(text)

        assert detected is True
        assert "Email body content" in result
        assert "John Doe" not in result
        assert "CEO" not in result

    @pytest.mark.unit
    def test_remove_email_signatures_italian_cordiali_saluti(self):
        """Test detection of 'Cordiali saluti' Italian signature."""
        text = """Buongiorno,

vorrei informazioni.

Cordiali saluti,
Mario Rossi"""
        result, detected = remove_email_signatures(text)

        assert detected is True
        assert "vorrei informazioni" in result
        assert "Cordiali saluti" not in result
        assert "Mario Rossi" not in result

    @pytest.mark.unit
    def test_remove_email_signatures_italian_distinti_saluti(self):
        """Test detection of 'Distinti saluti' Italian signature."""
        text = """Gentile team,

richiedo assistenza.

Distinti saluti,
Anna Bianchi"""
        result, detected = remove_email_signatures(text)

        assert detected is True
        assert "richiedo assistenza" in result
        assert "Distinti saluti" not in result
        assert "Anna Bianchi" not in result

    @pytest.mark.unit
    def test_remove_email_signatures_sent_from_iphone(self):
        """Test detection of 'Sent from my iPhone' signature."""
        text = """Quick reply!

Sent from my iPhone"""
        result, detected = remove_email_signatures(text)

        assert detected is True
        assert "Quick reply" in result
        assert "Sent from my iPhone" not in result

    @pytest.mark.unit
    def test_remove_email_signatures_inviato_da(self):
        """Test detection of 'Inviato da' Italian mobile signature."""
        text = """Risposta veloce!

Inviato da iPhone"""
        result, detected = remove_email_signatures(text)

        assert detected is True
        assert "Risposta veloce" in result
        assert "Inviato da" not in result

    @pytest.mark.unit
    def test_remove_email_signatures_none_found(self):
        """Test return (text, False) when no signature detected."""
        text = """Just a simple email body.
No signature here."""
        result, detected = remove_email_signatures(text)

        assert detected is False
        assert result == text

    @pytest.mark.unit
    def test_remove_email_signatures_underscore_delimiter(self):
        """Test detection of '___' alternative delimiter."""
        text = """Email content

___
Signature below"""
        result, detected = remove_email_signatures(text)

        assert detected is True
        assert "Email content" in result
        assert "Signature below" not in result

    @pytest.mark.unit
    def test_remove_email_signatures_case_insensitive(self):
        """Test that signature detection is case-insensitive."""
        text = """Body

CORDIALI SALUTI,
Marco"""
        result, detected = remove_email_signatures(text)

        assert detected is True
        assert "Marco" not in result


class TestRemoveReplyChains:
    """Tests for remove_reply_chains() function."""

    @pytest.mark.unit
    def test_remove_reply_chains_quoted_lines(self):
        """Test removal of '>' quoted lines."""
        text = """My reply to your email.

> This is the original message
> that was quoted.
> Multiple lines."""
        result, detected = remove_reply_chains(text)

        assert detected is True
        assert "My reply to your email" in result
        assert "> This is the original" not in result

    @pytest.mark.unit
    def test_remove_reply_chains_on_wrote(self):
        """Test removal of 'On ... wrote:' blocks."""
        text = """Thanks for your email!

On Feb 12, 2026, at 10:00 AM, john@example.com wrote:
> Previous message content"""
        result, detected = remove_reply_chains(text)

        assert detected is True
        assert "Thanks for your email" in result
        assert "On Feb 12" not in result

    @pytest.mark.unit
    def test_remove_reply_chains_italian_ha_scritto(self):
        """Test removal of 'Il ... ha scritto:' Italian pattern."""
        text = """Grazie per la risposta!

Il giorno 12 feb 2026, alle ore 10:00, supporto@example.com ha scritto:
> Messaggio precedente"""
        result, detected = remove_reply_chains(text)

        assert detected is True
        assert "Grazie per la risposta" in result
        assert "Il giorno" not in result

    @pytest.mark.unit
    def test_remove_reply_chains_original_message(self):
        """Test removal of '---Original Message---' blocks."""
        text = """See message below.

-----Original Message-----
From: sender@example.com
Content here"""
        result, detected = remove_reply_chains(text)

        assert detected is True
        assert "See message below" in result
        assert "Original Message" not in result

    @pytest.mark.unit
    def test_remove_reply_chains_italian_original_message(self):
        """Test removal of Italian 'Messaggio originale' blocks."""
        text = """Vedi sotto.

-----Messaggio originale-----
Da: mittente@example.com
Contenuto"""
        result, detected = remove_reply_chains(text)

        assert detected is True
        assert "Vedi sotto" in result
        assert "Messaggio originale" not in result

    @pytest.mark.unit
    def test_remove_reply_chains_none_found(self):
        """Test return (text, False) when no replies detected."""
        text = """Simple email with no quotes or replies."""
        result, detected = remove_reply_chains(text)

        assert detected is False
        assert result == text

    @pytest.mark.unit
    def test_remove_reply_chains_pipe_quotes(self):
        """Test removal of '|' alternative quote style."""
        text = """My response.

| Original message
| quoted with pipes"""
        result, detected = remove_reply_chains(text)

        assert detected is True
        assert "My response" in result


class TestRemoveExcessiveNewlines:
    """Tests for remove_excessive_newlines() function."""

    @pytest.mark.unit
    def test_remove_excessive_newlines_default(self):
        """Test limiting to 2 consecutive newlines by default."""
        text = "Line 1\n\n\n\n\n\nLine 2"
        result = remove_excessive_newlines(text)

        assert result == "Line 1\n\nLine 2"

    @pytest.mark.unit
    def test_remove_excessive_newlines_custom_max(self):
        """Test custom maximum consecutive newlines."""
        text = "Line 1\n\n\n\nLine 2"
        result = remove_excessive_newlines(text, max_consecutive=1)

        assert result == "Line 1\nLine 2"

    @pytest.mark.unit
    def test_remove_excessive_newlines_already_ok(self):
        """Test that already-ok text is unchanged."""
        text = "Line 1\n\nLine 2"
        result = remove_excessive_newlines(text)

        assert result == text
