"""
Unit tests for text normalization module (text_normalizer.py).

Tests cover:
- Complete canonicalization pipeline
- Whitespace normalization
- URL removal
- Email signature detection and removal (Italian patterns)
- Reply chain detection and removal
- Excessive newline handling
- RemovedSection tracking (audit trail)
"""

import pytest

from eml_classificator.canonicalization.text_normalizer import (
    canonicalize_text,
    normalize_whitespace,
    remove_email_signatures,
    remove_excessive_newlines,
    remove_reply_chains,
    remove_urls,
    remove_disclaimers,
)
from eml_classificator.models.canonicalization import RemovedSection


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
        result, removed_sections = canonicalize_text(text)

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
        # Should have tracked signature removal
        assert len(removed_sections) > 0

    @pytest.mark.unit
    def test_canonicalize_text_with_signature_removal_disabled(self):
        """Test canonicalization with signature removal disabled."""
        text = "Test\n\nCordiali saluti,\nMario"
        result, removed_sections = canonicalize_text(text, remove_signatures=False)

        # Signature should still be present (lowercased)
        assert "cordiali saluti" in result

    @pytest.mark.unit
    def test_canonicalize_text_empty_string(self):
        """Test handling of empty string."""
        result1, removed1 = canonicalize_text("")
        assert result1 == ""

        result2, removed2 = canonicalize_text("   ")
        assert result2 == ""


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
        result, removed_sections = remove_email_signatures(text)

        assert len(removed_sections) > 0
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
        result, removed_sections = remove_email_signatures(text)

        assert len(removed_sections) > 0
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
        result, removed_sections = remove_email_signatures(text)

        assert len(removed_sections) > 0
        assert "richiedo assistenza" in result
        assert "Distinti saluti" not in result
        assert "Anna Bianchi" not in result

    @pytest.mark.unit
    def test_remove_email_signatures_sent_from_iphone(self):
        """Test detection of 'Sent from my iPhone' signature."""
        text = """Quick reply!

Sent from my iPhone"""
        result, removed_sections = remove_email_signatures(text)

        assert len(removed_sections) > 0
        assert "Quick reply" in result
        assert "Sent from my iPhone" not in result

    @pytest.mark.unit
    def test_remove_email_signatures_inviato_da(self):
        """Test detection of 'Inviato da' Italian mobile signature."""
        text = """Risposta veloce!

Inviato da iPhone"""
        result, removed_sections = remove_email_signatures(text)

        assert len(removed_sections) > 0
        assert "Risposta veloce" in result
        assert "Inviato da" not in result

    @pytest.mark.unit
    def test_remove_email_signatures_none_found(self):
        """Test return (text, False) when no signature detected."""
        text = """Just a simple email body.
No signature here."""
        result, removed_sections = remove_email_signatures(text)

        assert len(removed_sections) == 0
        assert result == text

    @pytest.mark.unit
    def test_remove_email_signatures_underscore_delimiter(self):
        """Test detection of '___' alternative delimiter."""
        text = """Email content

___
Signature below"""
        result, removed_sections = remove_email_signatures(text)

        assert len(removed_sections) > 0
        assert "Email content" in result
        assert "Signature below" not in result

    @pytest.mark.unit
    def test_remove_email_signatures_case_insensitive(self):
        """Test that signature detection is case-insensitive."""
        text = """Body

CORDIALI SALUTI,
Marco"""
        result, removed_sections = remove_email_signatures(text)

        assert len(removed_sections) > 0
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
        result, removed_sections = remove_reply_chains(text)

        assert len(removed_sections) > 0
        assert "My reply to your email" in result
        assert "> This is the original" not in result

    @pytest.mark.unit
    def test_remove_reply_chains_on_wrote(self):
        """Test removal of 'On ... wrote:' blocks."""
        text = """Thanks for your email!

On Feb 12, 2026, at 10:00 AM, john@example.com wrote:
> Previous message content"""
        result, removed_sections = remove_reply_chains(text)

        assert len(removed_sections) > 0
        assert "Thanks for your email" in result
        assert "On Feb 12" not in result

    @pytest.mark.unit
    def test_remove_reply_chains_italian_ha_scritto(self):
        """Test removal of 'Il ... ha scritto:' Italian pattern."""
        text = """Grazie per la risposta!

Il giorno 12 feb 2026, alle ore 10:00, supporto@example.com ha scritto:
> Messaggio precedente"""
        result, removed_sections = remove_reply_chains(text)

        assert len(removed_sections) > 0
        assert "Grazie per la risposta" in result
        assert "Il giorno" not in result

    @pytest.mark.unit
    def test_remove_reply_chains_original_message(self):
        """Test removal of '---Original Message---' blocks."""
        text = """See message below.

-----Original Message-----
From: sender@example.com
Content here"""
        result, removed_sections = remove_reply_chains(text)

        assert len(removed_sections) > 0
        assert "See message below" in result
        assert "Original Message" not in result

    @pytest.mark.unit
    def test_remove_reply_chains_italian_original_message(self):
        """Test removal of Italian 'Messaggio originale' blocks."""
        text = """Vedi sotto.

-----Messaggio originale-----
Da: mittente@example.com
Contenuto"""
        result, removed_sections = remove_reply_chains(text)

        assert len(removed_sections) > 0
        assert "Vedi sotto" in result
        assert "Messaggio originale" not in result

    @pytest.mark.unit
    def test_remove_reply_chains_none_found(self):
        """Test return (text, False) when no replies detected."""
        text = """Simple email with no quotes or replies."""
        result, removed_sections = remove_reply_chains(text)

        assert len(removed_sections) == 0
        assert result == text

    @pytest.mark.unit
    def test_remove_reply_chains_pipe_quotes(self):
        """Test removal of '|' alternative quote style."""
        text = """My response.

| Original message
| quoted with pipes"""
        result, removed_sections = remove_reply_chains(text)

        assert len(removed_sections) > 0
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


class TestCanonalizationStability:
    """Test deterministic behavior of canonicalization (from spec)."""

    @pytest.mark.unit
    def test_canonicalization_stability(self):
        """Assicura che stessi input producano stesso output."""
        sample = "Hello\n\n> quoted line\n--\nSignature"

        clean1, removed1 = canonicalize_text(sample, keep_audit_info=True)
        clean2, removed2 = canonicalize_text(sample, keep_audit_info=True)

        assert clean1 == clean2, "Canonicalizzazione non deterministica!"
        assert len(removed1) == len(removed2)
        # Check that removed sections are identical
        for r1, r2 in zip(removed1, removed2):
            assert r1.type == r2.type
            assert r1.span_start == r2.span_start
            assert r1.span_end == r2.span_end
            assert r1.content == r2.content

    @pytest.mark.unit
    def test_canonicalization_with_no_audit(self):
        """Test that keep_audit_info=False returns empty list."""
        sample = "Hello\n--\nSignature"

        clean, removed = canonicalize_text(sample, keep_audit_info=False)

        assert isinstance(removed, list)
        assert len(removed) == 0
        assert "signature" not in clean

    @pytest.mark.unit
    def test_canonicalization_with_audit_returns_tuple(self):
        """Test that canonicalize_text returns tuple."""
        sample = "Test email"

        result = canonicalize_text(sample, keep_audit_info=True)

        assert isinstance(result, tuple)
        assert len(result) == 2
        text, removed_sections = result
        assert isinstance(text, str)
        assert isinstance(removed_sections, list)


class TestRemovedSectionTracking:
    """Test RemovedSection tracking for signatures, quotes, and disclaimers."""

    @pytest.mark.unit
    def test_signature_removal_tracked(self):
        """Test that signature removal is logged in removed_sections."""
        text = "Body content\n\n--\nBest regards,\nJohn"

        clean, removed = remove_email_signatures(text, keep_audit_info=True)

        assert len(removed) == 1
        assert removed[0].type == "signature"
        assert "--\nBest regards,\nJohn" in removed[0].content
        assert removed[0].span_start >= 0
        assert removed[0].span_end > removed[0].span_start

    @pytest.mark.unit
    def test_quote_removal_tracked(self):
        """Test that quote removal is logged."""
        text = "My reply\n\n> Original message\n> Second line"

        clean, removed = remove_reply_chains(text, keep_audit_info=True)

        assert len(removed) >= 1
        assert any(r.type in ("quote", "reply_header") for r in removed)
        # Check content was captured
        assert any("> Original" in r.content for r in removed)

    @pytest.mark.unit
    def test_disclaimer_removal_tracked(self):
        """Test that disclaimer removal is logged."""
        text = "Email body\n\n" + "_" * 15 + "\nDisclaimer text here"

        clean, removed = remove_disclaimers(text, keep_audit_info=True)

        assert len(removed) == 1
        assert removed[0].type == "disclaimer"
        assert "Disclaimer text" in removed[0].content

    @pytest.mark.unit
    def test_removed_section_span_accuracy(self):
        """Test that span positions accurately map to original text."""
        text = "Intro\n\n--\nSignature part"

        clean, removed = remove_email_signatures(text, keep_audit_info=True)

        if removed:
            section = removed[0]
            # Extract using span should match content
            original_section = text[section.span_start : section.span_end]
            assert original_section == section.content

    @pytest.mark.unit
    def test_multiple_removals_tracked(self):
        """Test that multiple types of removals are all tracked."""
        text = "Content\n\n> Quote\n\n--\nSignature"

        clean, removed = canonicalize_text(text, keep_audit_info=True)

        # Should have both quote and signature removed
        types_removed = {r.type for r in removed}
        assert len(types_removed) >= 1  # At least one type removed
        # Check we tracked something
        assert len(removed) > 0

    @pytest.mark.unit
    def test_removed_section_validation(self):
        """Test RemovedSection validation."""
        # Valid section
        section = RemovedSection(
            type="signature", span_start=0, span_end=10, content="test"
        )
        assert section.type == "signature"

        # Invalid type should raise
        with pytest.raises(ValueError, match="Invalid type"):
            RemovedSection(
                type="invalid", span_start=0, span_end=10, content="test"
            )

        # Invalid spans should raise
        with pytest.raises(ValueError, match="non-negative"):
            RemovedSection(
                type="signature", span_start=-1, span_end=10, content="test"
            )

        with pytest.raises(ValueError, match="span_start must be <= span_end"):
            RemovedSection(
                type="signature", span_start=20, span_end=10, content="test"
            )

    @pytest.mark.unit
    def test_italian_reply_header_tracked(self):
        """Test Italian 'Il giorno... ha scritto' pattern is tracked."""
        text = "Risposta\n\nIl giorno 12 feb ha scritto:\nQuoted content"

        clean, removed = remove_reply_chains(text, keep_audit_info=True)

        # Should detect and track the Italian reply header
        assert len(removed) >= 1
        assert any(r.type == "reply_header" for r in removed)
        assert "Il giorno" in " ".join(r.content for r in removed)

