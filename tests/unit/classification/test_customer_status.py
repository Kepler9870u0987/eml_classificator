"""
Unit tests for customer status determination with mock CRM.

Tests email/domain matching, text signal detection, and deterministic overrides.
"""

import pytest
from unittest.mock import Mock, patch

from eml_classificator.classification.customer_status import (
    MockCRMDatabase,
    detect_text_signals,
    compute_customer_status,
    override_customer_status_with_crm,
    EXISTING_CUSTOMER_SIGNALS,
    NEW_CUSTOMER_SIGNALS
)
from eml_classificator.classification.schemas import (
    CustomerStatus,
    CustomerStatusValue
)


class TestMockCRMDatabase:
    """Test mock CRM database functionality."""

    def test_crm_empty_database(self):
        """Test CRM with empty database."""
        crm = MockCRMDatabase(
            known_emails=[],
            known_domains=[]
        )

        match_type, confidence, source = crm.lookup_email("test@example.com")

        assert match_type == "no_match"
        assert confidence == 0.0

    def test_crm_exact_email_match(self):
        """Test exact email match in CRM."""
        crm = MockCRMDatabase(
            known_emails=["john@company.com", "jane@company.com"],
            known_domains=[]
        )

        match_type, confidence, source = crm.lookup_email("john@company.com")

        assert match_type == "exact"
        assert confidence == 1.0
        assert source == "crm_exact_match"

    def test_crm_domain_match(self):
        """Test domain match in CRM."""
        crm = MockCRMDatabase(
            known_emails=[],
            known_domains=["company.com", "cliente.it"]
        )

        match_type, confidence, source = crm.lookup_email("newuser@company.com")

        assert match_type == "domain"
        assert confidence == 0.9
        assert source == "crm_domain_match"

    def test_crm_exact_takes_precedence_over_domain(self):
        """Test exact match takes precedence over domain match."""
        crm = MockCRMDatabase(
            known_emails=["john@company.com"],
            known_domains=["company.com"]
        )

        match_type, confidence, source = crm.lookup_email("john@company.com")

        assert match_type == "exact"
        assert confidence == 1.0

    def test_crm_case_insensitive(self):
        """Test CRM lookup is case-insensitive."""
        crm = MockCRMDatabase(
            known_emails=["John@Company.COM"],
            known_domains=[]
        )

        match_type, confidence, source = crm.lookup_email("john@company.com")

        assert match_type == "exact"
        assert confidence == 1.0

    def test_crm_normalize_whitespace(self):
        """Test CRM normalizes whitespace in emails."""
        crm = MockCRMDatabase(
            known_emails=["  john@company.com  "],
            known_domains=[]
        )

        match_type, confidence, source = crm.lookup_email("john@company.com")

        assert match_type == "exact"
        assert confidence == 1.0

    def test_crm_invalid_email_format(self):
        """Test CRM handles invalid email formats gracefully."""
        crm = MockCRMDatabase(
            known_emails=["john@company.com"],
            known_domains=[]
        )

        match_type, confidence, source = crm.lookup_email("not-an-email")

        assert match_type == "no_match"

    def test_crm_empty_email(self):
        """Test CRM handles empty email string."""
        crm = MockCRMDatabase(
            known_emails=["john@company.com"],
            known_domains=[]
        )

        match_type, confidence, source = crm.lookup_email("")

        assert match_type == "no_match"


class TestTextSignalDetection:
    """Test text signal detection for customer status."""

    def test_detect_existing_customer_signal(self):
        """Test detection of existing customer signals."""
        text = "Buongiorno, sono vostro cliente dal 2020 e ho un problema."

        result = detect_text_signals(text)

        assert result is not None
        status_value, matched_signals = result
        assert status_value == "existing"
        assert len(matched_signals) > 0

    def test_detect_existing_customer_signal_contract(self):
        """Test detection of existing contract signal."""
        text = "Ho già un contratto attivo con voi e vorrei modificarlo."

        result = detect_text_signals(text)

        assert result is not None
        status_value, matched_signals = result
        assert status_value == "existing"

    def test_detect_new_customer_signal(self):
        """Test detection of new customer signals."""
        text = "Salve, sono un nuovo cliente e vorrei informazioni."

        result = detect_text_signals(text)

        assert result is not None
        status_value, matched_signals = result
        assert status_value == "new"
        assert len(matched_signals) > 0

    def test_detect_new_customer_interested(self):
        """Test detection of potential new customer signal."""
        text = "Sono interessato ai vostri servizi e vorrei maggiori informazioni."

        result = detect_text_signals(text)

        assert result is not None
        status_value, matched_signals = result
        assert status_value == "new"

    def test_no_signals_detected(self):
        """Test when no customer status signals are present."""
        text = "Buongiorno, ho una domanda generale."

        result = detect_text_signals(text)

        assert result is None

    def test_conflicting_signals_existing_wins(self):
        """Test when both signals present, existing takes precedence."""
        text = "Sono vostro cliente dal 2020. Ora sono interessato a nuovi servizi."

        result = detect_text_signals(text)

        assert result is not None
        status_value, _ = result
        # Existing should take precedence (comes first in detection order)
        assert status_value == "existing"

    def test_empty_text(self):
        """Test text signal detection with empty text."""
        result = detect_text_signals("")

        assert result is None

    def test_case_insensitive_detection(self):
        """Test text signals are detected case-insensitively."""
        text = "SONO VOSTRO CLIENTE DAL 2020"

        result = detect_text_signals(text)

        assert result is not None
        status_value, _ = result
        assert status_value == "existing"


class TestComputeCustomerStatus:
    """Test main customer status computation function."""

    def test_compute_status_crm_exact_match(self):
        """Test CRM exact match gives highest priority."""
        crm = MockCRMDatabase(
            known_emails=["john@company.com"],
            known_domains=[]
        )

        status = compute_customer_status(
            from_email="john@company.com",
            text_body="Sono un nuovo cliente",  # Contradictory text ignored
            crm_db=crm
        )

        assert status.value == CustomerStatusValue.EXISTING
        assert status.confidence == 1.0
        assert status.source == "crm_exact_match"

    def test_compute_status_crm_domain_match(self):
        """Test CRM domain match."""
        crm = MockCRMDatabase(
            known_emails=[],
            known_domains=["company.com"]
        )

        status = compute_customer_status(
            from_email="newperson@company.com",
            text_body="",
            crm_db=crm
        )

        assert status.value == CustomerStatusValue.EXISTING
        assert status.confidence == 0.9
        assert status.source == "crm_domain_match"

    def test_compute_status_text_signal_existing(self):
        """Test text signal detection when no CRM match."""
        crm = MockCRMDatabase(known_emails=[], known_domains=[])

        status = compute_customer_status(
            from_email="unknown@example.com",
            text_body="Sono vostro cliente dal 2020.",
            crm_db=crm
        )

        assert status.value == CustomerStatusValue.EXISTING
        assert status.confidence == 0.5
        assert status.source == "text_signal"

    def test_compute_status_text_signal_new(self):
        """Test text signal for new customer."""
        crm = MockCRMDatabase(known_emails=[], known_domains=[])

        status = compute_customer_status(
            from_email="prospect@example.com",
            text_body="Sono un nuovo cliente interessato ai servizi.",
            crm_db=crm
        )

        assert status.value == CustomerStatusValue.NEW
        assert status.confidence == 0.6
        assert status.source == "text_signal"

    def test_compute_status_default_new(self):
        """Test default to NEW when no signals."""
        crm = MockCRMDatabase(known_emails=[], known_domains=[])

        status = compute_customer_status(
            from_email="unknown@example.com",
            text_body="Buongiorno, ho una domanda.",
            crm_db=crm
        )

        assert status.value == CustomerStatusValue.NEW
        assert status.confidence == 0.8
        assert status.source == "no_crm_no_signal"

    def test_compute_status_no_crm_provided(self):
        """Test fallback when no CRM database provided."""
        status = compute_customer_status(
            from_email="test@example.com",
            text_body="",
            crm_db=None
        )

        assert status.value == CustomerStatusValue.NEW
        assert status.confidence == 0.8
        assert status.source == "no_crm_no_signal"


class TestOverrideCustomerStatus:
    """Test override function that replaces LLM guess."""

    def test_override_with_crm_exact_match(self):
        """Test LLM guess is overridden by CRM exact match."""
        llm_status = CustomerStatus(
            value=CustomerStatusValue.UNKNOWN,
            confidence=0.5,
            source="llm_guess"
        )

        crm = MockCRMDatabase(
            known_emails=["john@company.com"],
            known_domains=[]
        )

        final_status = override_customer_status_with_crm(
            llm_customer_status=llm_status,
            from_email="john@company.com",
            text_body="",
            crm_db=crm
        )

        assert final_status.value == CustomerStatusValue.EXISTING
        assert final_status.confidence == 1.0
        assert final_status.source == "crm_exact_match"
        # LLM guess completely ignored

    def test_override_with_text_signal(self):
        """Test LLM guess is overridden by text signal."""
        llm_status = CustomerStatus(
            value=CustomerStatusValue.UNKNOWN,
            confidence=0.3,
            source="llm_guess"
        )

        crm = MockCRMDatabase(known_emails=[], known_domains=[])

        final_status = override_customer_status_with_crm(
            llm_customer_status=llm_status,
            from_email="test@example.com",
            text_body="Sono vostro cliente dal 2019.",
            crm_db=crm
        )

        assert final_status.value == CustomerStatusValue.EXISTING
        assert final_status.confidence == 0.5
        assert final_status.source == "text_signal"

    def test_override_default_no_signals(self):
        """Test default behavior when no deterministic signals."""
        llm_status = CustomerStatus(
            value=CustomerStatusValue.UNKNOWN,
            confidence=0.4,
            source="llm_guess"
        )

        crm = MockCRMDatabase(known_emails=[], known_domains=[])

        final_status = override_customer_status_with_crm(
            llm_customer_status=llm_status,
            from_email="test@example.com",
            text_body="Buongiorno.",
            crm_db=crm
        )

        assert final_status.value == CustomerStatusValue.NEW
        assert final_status.source == "no_crm_no_signal"


@pytest.mark.skip(reason="load_known_customers_from_file not implemented yet")
class TestLoadKnownCustomers:
    """Test loading known customers from file."""

    def test_load_from_file_success(self, tmp_path):
        """Test loading customer emails from file."""
        # Create temporary file
        customers_file = tmp_path / "customers.txt"
        customers_file.write_text(
            "john@company.com\n"
            "jane@cliente.it\n"
            "# This is a comment\n"
            "\n"  # Empty line
            "  bob@example.com  \n"  # Whitespace
        )

        emails = load_known_customers_from_file(str(customers_file))

        assert len(emails) == 3
        assert "john@company.com" in emails
        assert "jane@cliente.it" in emails
        assert "bob@example.com" in emails

    def test_load_from_file_not_found(self):
        """Test loading from non-existent file returns empty list."""
        emails = load_known_customers_from_file("/nonexistent/file.txt")

        assert emails == []

    def test_load_from_file_empty(self, tmp_path):
        """Test loading from empty file."""
        customers_file = tmp_path / "empty.txt"
        customers_file.write_text("")

        emails = load_known_customers_from_file(str(customers_file))

        assert emails == []

    def test_load_from_file_comments_only(self, tmp_path):
        """Test loading from file with only comments."""
        customers_file = tmp_path / "comments.txt"
        customers_file.write_text(
            "# Comment 1\n"
            "# Comment 2\n"
            "\n"
        )

        emails = load_known_customers_from_file(str(customers_file))

        assert emails == []


class TestSignalPatterns:
    """Test specific signal patterns."""

    def test_existing_signal_patterns(self):
        """Test each existing customer signal pattern."""
        test_cases = [
            "Ho già un contratto con voi",
            "Abbiamo già un account attivo",
            "Sono cliente dal 2020",
            "Sono vostro cliente",
            "Cliente esistente",
            "Ho un servizio attivo",
            "Il mio account risulta bloccato"
        ]

        for text in test_cases:
            result = detect_text_signals(text)
            assert result is not None, f"Failed to detect signal in: {text}"
            status_value, _ = result
            assert status_value == "existing", f"Wrong status for: {text}"

    def test_new_signal_patterns(self):
        """Test each new customer signal pattern."""
        test_cases = [
            "Sono un nuovo cliente",
            "Sono interessato ai vostri servizi",
            "Vorrei aprire un account",
            "Vorrei diventare cliente",
            "Potenziale cliente"
        ]

        for text in test_cases:
            result = detect_text_signals(text)
            assert result is not None, f"Failed to detect signal in: {text}"
            status_value, _ = result
            assert status_value == "new", f"Wrong status for: {text}"


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_malformed_email_address(self):
        """Test handling of malformed email addresses."""
        crm = MockCRMDatabase(
            known_emails=["valid@example.com"],
            known_domains=[]
        )

        status = compute_customer_status(
            from_email="not-an-email",
            text_body="",
            crm_db=crm
        )

        # Should default to NEW
        assert status.value == CustomerStatusValue.NEW

    def test_none_email(self):
        """Test handling of None email."""
        crm = MockCRMDatabase(
            known_emails=["valid@example.com"],
            known_domains=[]
        )

        status = compute_customer_status(
            from_email=None,
            text_body="",
            crm_db=crm
        )

        assert status.value == CustomerStatusValue.NEW

    def test_very_long_text_body(self):
        """Test performance with very long text body."""
        crm = MockCRMDatabase(known_emails=[], known_domains=[])

        # Generate long text
        long_text = "Some text. " * 10000 + "Sono vostro cliente dal 2020."

        status = compute_customer_status(
            from_email="test@example.com",
            text_body=long_text,
            crm_db=crm
        )

        # Should still detect the signal
        assert status.value == CustomerStatusValue.EXISTING
        assert status.source == "text_signal"

    def test_unicode_in_email(self):
        """Test handling of unicode characters in email."""
        crm = MockCRMDatabase(
            known_emails=["tëst@ëxämplë.com"],
            known_domains=[]
        )

        status = compute_customer_status(
            from_email="tëst@ëxämplë.com",
            text_body="",
            crm_db=crm
        )

        assert status.value == CustomerStatusValue.EXISTING
        assert status.confidence == 1.0
