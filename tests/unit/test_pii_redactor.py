"""
Unit tests for PII redaction module (redactor.py).

Tests cover:
- Different redaction levels (none, minimal, standard, aggressive)
- Email address redaction
- Italian phone number redaction
- Audit log generation
- Hashing with salt
- Version and level tracking in audit log (GDPR compliance)
"""

import pytest

from eml_classificator.pii.redactor import PIIRedactor, RedactionLevel, PII_REDACTION_VERSION


class TestRedactionLevels:
    """Tests for different PII redaction levels."""

    @pytest.mark.unit
    def test_redaction_level_none(self):
        """Test that no redaction occurs when level=none."""
        text = "Contact me at john@example.com or call 340 1234567"
        redactor = PIIRedactor(level=RedactionLevel.NONE, salt="test-salt")
        result, log = redactor.redact_text(text)

        assert result == text  # Unchanged
        assert len(log) == 0  # No redactions logged

    @pytest.mark.unit
    def test_redaction_level_minimal_emails_only(self):
        """Test that only emails are redacted at minimal level."""
        text = "Contact me at john@example.com or call 340 1234567"
        redactor = PIIRedactor(level=RedactionLevel.MINIMAL, salt="test-salt")
        result, log = redactor.redact_text(text)

        assert "john@example.com" not in result
        assert "[EMAIL_1]" in result
        assert "340 1234567" in result  # Phone not redacted
        assert len(log) == 1
        assert log[0]["type"] == "email"

    @pytest.mark.unit
    def test_redaction_level_standard_emails_phones(self):
        """Test that emails and phones are redacted at standard level."""
        text = "Contact me at john@example.com or call 340 1234567"
        redactor = PIIRedactor(level=RedactionLevel.STANDARD, salt="test-salt")
        result, log = redactor.redact_text(text)

        assert "john@example.com" not in result
        assert "[EMAIL_1]" in result
        assert "340 1234567" not in result
        assert "[PHONE_1]" in result
        assert len(log) == 2

    @pytest.mark.unit
    def test_redaction_level_aggressive_emails_phones(self):
        """Test aggressive level (currently same as standard)."""
        text = "Contact john@example.com or 340 1234567"
        redactor = PIIRedactor(level=RedactionLevel.AGGRESSIVE, salt="test-salt")
        result, log = redactor.redact_text(text)

        # Aggressive currently redacts emails + phones (NER not implemented)
        assert "john@example.com" not in result
        assert "340 1234567" not in result
        assert len(log) == 2


class TestRedactEmails:
    """Tests for email address redaction."""

    @pytest.mark.unit
    def test_redact_emails_single(self):
        """Test redacting single email address."""
        text = "My email is john.doe@example.com"
        redactor = PIIRedactor(level=RedactionLevel.STANDARD, salt="test-salt")
        result, log = redactor.redact_emails(text)

        assert "john.doe@example.com" not in result
        assert "[EMAIL_1]" in result
        assert len(log) == 1

    @pytest.mark.unit
    def test_redact_emails_multiple(self):
        """Test redacting multiple emails with sequential placeholders."""
        text = "Contact john@example.com or jane@example.com"
        redactor = PIIRedactor(level=RedactionLevel.STANDARD, salt="test-salt")
        result, log = redactor.redact_emails(text)

        assert "john@example.com" not in result
        assert "jane@example.com" not in result
        assert "[EMAIL_1]" in result
        assert "[EMAIL_2]" in result
        assert len(log) == 2

    @pytest.mark.unit
    def test_redact_emails_none_found(self):
        """Test that empty log is returned when no emails found."""
        text = "No emails in this text"
        redactor = PIIRedactor(level=RedactionLevel.STANDARD, salt="test-salt")
        result, log = redactor.redact_emails(text)

        assert result == text
        assert len(log) == 0

    @pytest.mark.unit
    def test_redact_emails_various_formats(self):
        """Test redacting emails in various valid formats."""
        text = """Emails:
- simple@example.com
- with.dots@example.com
- with+plus@example.com
- numbers123@example.com
- subdomain@mail.example.com
"""
        redactor = PIIRedactor(level=RedactionLevel.STANDARD, salt="test-salt")
        result, log = redactor.redact_emails(text)

        assert len(log) == 5
        assert "[EMAIL_1]" in result
        assert "[EMAIL_5]" in result
        assert "@" not in result or result.count("@") == 0  # All @ should be gone


class TestRedactEmailsAuditLog:
    """Tests for email redaction audit log."""

    @pytest.mark.unit
    def test_redact_emails_audit_log(self):
        """Test that audit log contains required fields."""
        text = "Email: test@example.com"
        redactor = PIIRedactor(level=RedactionLevel.STANDARD, salt="test-salt")
        result, log = redactor.redact_emails(text)

        assert len(log) == 1
        entry = log[0]
        assert entry["type"] == "email"
        assert "position" in entry
        assert isinstance(entry["position"], list)
        assert len(entry["position"]) == 2
        assert "original_hash" in entry
        assert "placeholder" in entry
        assert entry["placeholder"] == "[EMAIL_1]"

    @pytest.mark.unit
    def test_redact_emails_audit_log_version(self):
        """Test that audit log includes pii_redaction_version."""
        text = "Email: test@example.com"
        redactor = PIIRedactor(level=RedactionLevel.STANDARD, salt="test-salt")
        result, log = redactor.redact_emails(text)

        assert len(log) == 1
        assert log[0]["pii_redaction_version"] == PII_REDACTION_VERSION

    @pytest.mark.unit
    def test_redact_emails_audit_log_level(self):
        """Test that audit log includes pii_redaction_level."""
        text = "Email: test@example.com"
        redactor = PIIRedactor(level=RedactionLevel.MINIMAL, salt="test-salt")
        result, log = redactor.redact_emails(text)

        assert len(log) == 1
        assert log[0]["pii_redaction_level"] == "minimal"

    @pytest.mark.unit
    def test_redact_emails_audit_log_position(self):
        """Test that position tracking is correct."""
        text = "Contact test@example.com for info"
        redactor = PIIRedactor(level=RedactionLevel.STANDARD, salt="test-salt")
        result, log = redactor.redact_emails(text)

        # Position should point to the original location
        assert len(log) == 1
        start, end = log[0]["position"]
        assert text[start:end] == "test@example.com"


class TestRedactPhoneNumbers:
    """Tests for Italian phone number redaction."""

    @pytest.mark.unit
    def test_redact_phone_italian_mobile(self):
        """Test redacting Italian mobile format (3XX XXXXXXX)."""
        text = "Call me at 340 1234567"
        redactor = PIIRedactor(level=RedactionLevel.STANDARD, salt="test-salt")
        result, log = redactor.redact_phone_numbers(text)

        assert "340 1234567" not in result
        assert "[PHONE_1]" in result
        assert len(log) == 1

    @pytest.mark.unit
    def test_redact_phone_italian_landline(self):
        """Test redacting Italian landline format (0XX XXXXXXX)."""
        text = "Office: 02 12345678"
        redactor = PIIRedactor(level=RedactionLevel.STANDARD, salt="test-salt")
        result, log = redactor.redact_phone_numbers(text)

        assert "02 12345678" not in result
        assert "[PHONE_1]" in result

    @pytest.mark.unit
    def test_redact_phone_with_country_code(self):
        """Test redacting phone with +39 country code."""
        text = "International: +39 06 12345678"
        redactor = PIIRedactor(level=RedactionLevel.STANDARD, salt="test-salt")
        result, log = redactor.redact_phone_numbers(text)

        assert "+39 06 12345678" not in result or "06 12345678" not in result
        assert "[PHONE_1]" in result

    @pytest.mark.unit
    def test_redact_phone_multiple(self):
        """Test redacting multiple phone numbers with sequential placeholders."""
        text = "Mobile: 340 1234567, Office: 02 87654321"
        redactor = PIIRedactor(level=RedactionLevel.STANDARD, salt="test-salt")
        result, log = redactor.redact_phone_numbers(text)

        assert len(log) == 2
        assert "[PHONE_1]" in result
        assert "[PHONE_2]" in result

    @pytest.mark.unit
    def test_redact_phone_with_dash(self):
        """Test redacting phone with dash separator."""
        text = "Call 02-12345678"
        redactor = PIIRedactor(level=RedactionLevel.STANDARD, salt="test-salt")
        result, log = redactor.redact_phone_numbers(text)

        assert "02-12345678" not in result
        assert "[PHONE_1]" in result

    @pytest.mark.unit
    def test_redact_phone_no_space(self):
        """Test redacting phone without spaces."""
        text = "Mobile: 3401234567"
        redactor = PIIRedactor(level=RedactionLevel.STANDARD, salt="test-salt")
        result, log = redactor.redact_phone_numbers(text)

        # Pattern should match mobile without spaces
        assert len(log) >= 1 or "3401234567" not in result


class TestRedactPhoneAuditLog:
    """Tests for phone redaction audit log."""

    @pytest.mark.unit
    def test_redact_phone_audit_log(self):
        """Test that phone audit log has correct format."""
        text = "Call 340 1234567"
        redactor = PIIRedactor(level=RedactionLevel.STANDARD, salt="test-salt")
        result, log = redactor.redact_phone_numbers(text)

        assert len(log) == 1
        entry = log[0]
        assert entry["type"] == "phone"
        assert "position" in entry
        assert "original_hash" in entry
        assert "placeholder" in entry
        assert entry["placeholder"] == "[PHONE_1]"
        assert entry["pii_redaction_version"] == PII_REDACTION_VERSION
        assert entry["pii_redaction_level"] == "standard"


class TestRedactTextMixed:
    """Tests for redacting mixed PII types."""

    @pytest.mark.unit
    def test_redact_text_mixed_pii(self):
        """Test redacting text with both emails and phone numbers."""
        text = "Contact john@example.com or call 340 1234567"
        redactor = PIIRedactor(level=RedactionLevel.STANDARD, salt="test-salt")
        result, log = redactor.redact_text(text)

        # Both should be redacted
        assert "john@example.com" not in result
        assert "340 1234567" not in result
        assert "[EMAIL_1]" in result
        assert "[PHONE_1]" in result
        assert len(log) == 2
        # Check log entries
        email_entries = [e for e in log if e["type"] == "email"]
        phone_entries = [e for e in log if e["type"] == "phone"]
        assert len(email_entries) == 1
        assert len(phone_entries) == 1

    @pytest.mark.unit
    def test_redact_text_multiple_pii_types(self):
        """Test redacting text with multiple emails and phones."""
        text = """
        Primary: john@example.com, 340 1234567
        Secondary: jane@example.com, 02 87654321
        """
        redactor = PIIRedactor(level=RedactionLevel.STANDARD, salt="test-salt")
        result, log = redactor.redact_text(text)

        assert len(log) == 4
        assert "[EMAIL_1]" in result
        assert "[EMAIL_2]" in result
        assert "[PHONE_1]" in result
        assert "[PHONE_2]" in result


class TestHashPII:
    """Tests for PII hashing."""

    @pytest.mark.unit
    def test_hash_pii_with_salt(self):
        """Test that hashing uses salt correctly."""
        redactor = PIIRedactor(level=RedactionLevel.STANDARD, salt="test-salt")
        email = "test@example.com"

        hash1 = redactor._hash_pii(email)
        assert hash1 is not None
        assert len(hash1) > 0

    @pytest.mark.unit
    def test_hash_pii_deterministic(self):
        """Test that same value + salt produces same hash."""
        redactor = PIIRedactor(level=RedactionLevel.STANDARD, salt="test-salt")
        email = "test@example.com"

        hash1 = redactor._hash_pii(email)
        hash2 = redactor._hash_pii(email)

        assert hash1 == hash2

    @pytest.mark.unit
    def test_hash_pii_different_salt(self):
        """Test that different salt produces different hash."""
        email = "test@example.com"
        redactor1 = PIIRedactor(level=RedactionLevel.STANDARD, salt="salt1")
        redactor2 = PIIRedactor(level=RedactionLevel.STANDARD, salt="salt2")

        hash1 = redactor1._hash_pii(email)
        hash2 = redactor2._hash_pii(email)

        assert hash1 != hash2

    @pytest.mark.unit
    def test_hash_pii_different_values(self):
        """Test that different values produce different hashes."""
        redactor = PIIRedactor(level=RedactionLevel.STANDARD, salt="test-salt")

        hash1 = redactor._hash_pii("email1@example.com")
        hash2 = redactor._hash_pii("email2@example.com")

        assert hash1 != hash2


class TestGDPRCompliance:
    """Tests for GDPR compliance features."""

    @pytest.mark.unit
    def test_audit_log_includes_version_and_level(self):
        """Test that all audit log entries include version and level."""
        text = "Contact john@example.com or 340 1234567"
        redactor = PIIRedactor(level=RedactionLevel.STANDARD, salt="test-salt")
        result, log = redactor.redact_text(text)

        # Every entry must have version and level
        for entry in log:
            assert "pii_redaction_version" in entry
            assert entry["pii_redaction_version"] == PII_REDACTION_VERSION
            assert "pii_redaction_level" in entry
            assert entry["pii_redaction_level"] == "standard"

    @pytest.mark.unit
    def test_audit_log_hashes_are_salted(self):
        """Test that hashes in audit log are salted."""
        text = "Email: test@example.com"
        redactor1 = PIIRedactor(level=RedactionLevel.STANDARD, salt="salt1")
        redactor2 = PIIRedactor(level=RedactionLevel.STANDARD, salt="salt2")

        _, log1 = redactor1.redact_text(text)
        _, log2 = redactor2.redact_text(text)

        # Same email, different salts = different hashes
        assert log1[0]["original_hash"] != log2[0]["original_hash"]
