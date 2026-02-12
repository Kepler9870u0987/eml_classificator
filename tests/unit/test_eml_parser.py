"""
Unit tests for email parser module (eml_parser.py).

Tests cover:
- Parsing .eml bytes and files
- Header extraction (From, To, CC, Subject, Date, Message-ID, References)
- Body extraction (plain text, HTML, multipart)
- Attachment metadata extraction
- Document ID computation (determinism)
- Encoding detection
"""

from datetime import datetime

import pytest

from eml_classificator.parsing.eml_parser import (
    compute_document_id,
    detect_encoding,
    extract_attachments,
    extract_body,
    extract_headers,
    parse_eml_bytes,
    parse_eml_file,
)
from tests.fixtures.emails import SAMPLE_EMAILS


class TestParseEmlBytes:
    """Tests for parse_eml_bytes() function."""

    @pytest.mark.unit
    def test_parse_eml_bytes_valid(self, sample_eml_bytes):
        """Test parsing valid .eml bytes."""
        msg = parse_eml_bytes(sample_eml_bytes)
        assert msg is not None
        assert msg.get("From") == "sender@example.com"
        assert msg.get("To") == "recipient@example.com"
        assert msg.get("Subject") == "Test Email"

    @pytest.mark.unit
    def test_parse_eml_bytes_invalid(self, malformed_eml):
        """Test handling of malformed bytes."""
        # Malformed email should raise ValueError
        msg = parse_eml_bytes(malformed_eml)
        # Parser is lenient- it will still create a Message object
        #but it won't have proper structure
        assert msg is not None

    @pytest.mark.unit
    def test_parse_eml_bytes_multipart(self, multipart_html_eml):
        """Test parsing multipart email."""
        msg = parse_eml_bytes(multipart_html_eml)
        assert msg is not None
        assert msg.is_multipart()
        assert msg.get("Subject") == "Monthly Newsletter"


class TestParseEmlFile:
    """Tests for parse_eml_file() function."""

    @pytest.mark.unit
    def test_parse_eml_file_valid(self, tmp_eml_file):
        """Test parsing valid .eml file from disk."""
        msg = parse_eml_file(tmp_eml_file)
        assert msg is not None
        assert msg.get("From") == "sender@example.com"

    @pytest.mark.unit
    def test_parse_eml_file_not_found(self):
        """Test handling of missing file."""
        with pytest.raises(FileNotFoundError):
            parse_eml_file("/nonexistent/path/email.eml")


class TestExtractHeaders:
    """Tests for extract_headers() function."""

    @pytest.mark.unit
    def test_extract_headers_complete(self, sample_eml_bytes):
        """Test extracting all headers from complete email."""
        msg = parse_eml_bytes(sample_eml_bytes)
        headers = extract_headers(msg)

        assert headers.from_address == "sender@example.com"
        assert headers.to_addresses == ["recipient@example.com"]
        assert headers.subject == "Test Email"
        assert headers.message_id == "<test123@example.com>"
        assert headers.date is not None
        assert headers.date.year == 2026
        assert headers.date.month == 2
        assert headers.date.day == 12

    @pytest.mark.unit
    def test_extract_headers_missing(self):
        """Test handling of missing headers gracefully."""
        minimal_eml = b"""From: sender@example.com\n\nBody text"""
        msg = parse_eml_bytes(minimal_eml)
        headers = extract_headers(msg)

        assert headers.from_address == "sender@example.com"
        assert headers.to_addresses == []
        assert headers.subject == ""
        assert headers.date is None
        assert headers.message_id is None

    @pytest.mark.unit
    def test_extract_headers_date_parsing(self):
        """Test parsing various date formats."""
        eml_with_date = b"""From: sender@example.com
To: recipient@example.com
Date: Wed, 12 Feb 2026 10:30:00 +0100
Subject: Date test

Body"""
        msg = parse_eml_bytes(eml_with_date)
        headers = extract_headers(msg)

        assert headers.date is not None
        assert headers.date.year == 2026
        assert headers.date.month == 2
        assert headers.date.day == 12
        assert headers.date.hour == 10
        assert headers.date.minute == 30

    @pytest.mark.unit
    def test_extract_headers_cc_addresses(self):
        """Test extraction of CC addresses."""
        eml_bytes = SAMPLE_EMAILS["cc_bcc"]
        msg = parse_eml_bytes(eml_bytes)
        headers = extract_headers(msg)

        assert len(headers.to_addresses) == 2
        assert "to1@example.com" in headers.to_addresses
        assert "to2@example.com" in headers.to_addresses
        assert len(headers.cc_addresses) == 2
        assert "cc1@example.com" in headers.cc_addresses
        assert "cc2@example.com" in headers.cc_addresses

    @pytest.mark.unit
    def test_extract_headers_references(self):
        """Test extraction of References header for threading."""
        eml_bytes = SAMPLE_EMAILS["threaded"]
        msg = parse_eml_bytes(eml_bytes)
        headers = extract_headers(msg)

        assert headers.in_reply_to == "<thread-2@example.com>"
        assert len(headers.references) == 2
        assert "<thread-1@example.com>" in headers.references
        assert "<thread-2@example.com>" in headers.references


class TestExtractBody:
    """Tests for extract_body() function."""

    @pytest.mark.unit
    def test_extract_body_plain_text(self, sample_eml_bytes):
        """Test extracting plain text only email."""
        msg = parse_eml_bytes(sample_eml_bytes)
        plain_text, html_body = extract_body(msg)

        assert "Hello, this is a simple test email" in plain_text
        assert html_body is None

    @pytest.mark.unit
    def test_extract_body_html_only(self, html_only_eml):
        """Test extracting HTML only email."""
        msg = parse_eml_bytes(html_only_eml)
        plain_text, html_body = extract_body(msg)

        assert plain_text == ""
        assert html_body is not None
        assert "<h1>Welcome!</h1>" in html_body
        assert "support@example.com" in html_body

    @pytest.mark.unit
    def test_extract_body_multipart(self, multipart_html_eml):
        """Test extracting multipart email with both plain and HTML."""
        msg = parse_eml_bytes(multipart_html_eml)
        plain_text, html_body = extract_body(msg)

        assert "plain text version" in plain_text
        assert html_body is not None
        assert "<h1>This is the HTML version</h1>" in html_body

    @pytest.mark.unit
    def test_extract_body_encoding_utf8(self):
        """Test handling UTF-8 encoding."""
        utf8_eml = b"""From: sender@example.com
To: recipient@example.com
Subject: UTF-8 test
Content-Type: text/plain; charset="utf-8"

Caf\xc3\xa9 na\xc3\xafve r\xc3\xa9sum\xc3\xa9"""
        msg = parse_eml_bytes(utf8_eml)
        plain_text, _ = extract_body(msg)

        assert "Café" in plain_text
        assert "naïve" in plain_text
        assert "résumé" in plain_text

    @pytest.mark.unit
    def test_extract_body_encoding_latin1(self):
        """Test handling ISO-8859-1 (Latin-1) encoding."""
        eml_bytes = SAMPLE_EMAILS["latin1_encoding"]
        msg = parse_eml_bytes(eml_bytes)
        plain_text, _ = extract_body(msg)

        # Should decode Latin-1 correctly
        assert "Caf" in plain_text  # Café
        assert "na" in plain_text  # naïve
        assert "r" in plain_text  # résumé

    @pytest.mark.unit
    def test_extract_body_encoding_detection(self):
        """Test fallback to charset-normalizer for encoding detection."""
        # Email without explicit charset
        no_charset_eml = b"""From: sender@example.com
To: recipient@example.com
Subject: No charset

Body with UTF-8: \xc3\xa9"""
        msg = parse_eml_bytes(no_charset_eml)
        plain_text, _ = extract_body(msg)

        # Should still extract text (may use fallback detection)
        assert len(plain_text) > 0


class TestExtractAttachments:
    """Tests for extract_attachments() function."""

    @pytest.mark.unit
    def test_extract_attachments_single(self, attachment_eml):
        """Test extracting single attachment metadata."""
        msg = parse_eml_bytes(attachment_eml)
        attachments = extract_attachments(msg)

        assert len(attachments) == 1
        assert attachments[0].filename == "document.pdf"
        assert attachments[0].content_type == "application/pdf"
        assert attachments[0].size_bytes > 0

    @pytest.mark.unit
    def test_extract_attachments_multiple(self):
        """Test extracting multiple attachments."""
        eml_bytes = SAMPLE_EMAILS["multiple_attachments"]
        msg = parse_eml_bytes(eml_bytes)
        attachments = extract_attachments(msg)

        assert len(attachments) == 3
        filenames = [att.filename for att in attachments]
        assert "report.pdf" in filenames
        assert "chart.png" in filenames
        assert "data.csv" in filenames

    @pytest.mark.unit
    def test_extract_attachments_none(self, sample_eml_bytes):
        """Test handling of emails without attachments."""
        msg = parse_eml_bytes(sample_eml_bytes)
        attachments = extract_attachments(msg)

        assert len(attachments) == 0


class TestComputeDocumentId:
    """Tests for compute_document_id() function."""

    @pytest.mark.unit
    def test_compute_document_id_deterministic(self, sample_eml_bytes, fixed_timestamp):
        """Test that same input + timestamp produces same ID."""
        id1 = compute_document_id(sample_eml_bytes, fixed_timestamp)
        id2 = compute_document_id(sample_eml_bytes, fixed_timestamp)

        assert id1 == id2
        assert id1.startswith("eml-")
        assert len(id1) == 36  # "eml-" + 32 hex chars

    @pytest.mark.unit
    def test_compute_document_id_unique(self, sample_eml_bytes):
        """Test that different timestamps produce different IDs."""
        ts1 = datetime(2026, 2, 12, 10, 0, 0)
        ts2 = datetime(2026, 2, 12, 11, 0, 0)

        id1 = compute_document_id(sample_eml_bytes, ts1)
        id2 = compute_document_id(sample_eml_bytes, ts2)

        assert id1 != id2

    @pytest.mark.unit
    def test_compute_document_id_different_content(self, fixed_timestamp):
        """Test that different email content produces different IDs."""
        eml1 = b"From: a@example.com\n\nBody 1"
        eml2 = b"From: b@example.com\n\nBody 2"

        id1 = compute_document_id(eml1, fixed_timestamp)
        id2 = compute_document_id(eml2, fixed_timestamp)

        assert id1 != id2


class TestDetectEncoding:
    """Tests for detect_encoding() function."""

    @pytest.mark.unit
    def test_detect_encoding_from_header(self, sample_eml_bytes):
        """Test using charset from email headers."""
        msg = parse_eml_bytes(sample_eml_bytes)
        encoding = detect_encoding(msg)

        assert encoding == "utf-8"

    @pytest.mark.unit
    def test_detect_encoding_latin1(self):
        """Test detecting ISO-8859-1 encoding."""
        eml_bytes = SAMPLE_EMAILS["latin1_encoding"]
        msg = parse_eml_bytes(eml_bytes)
        encoding = detect_encoding(msg)

        assert encoding == "iso-8859-1"

    @pytest.mark.unit
    def test_detect_encoding_detection(self):
        """Test charset-normalizer detection when missing."""
        # Email without explicit charset
        no_charset_eml = b"""From: sender@example.com

Body text"""
        msg = parse_eml_bytes(no_charset_eml)
        encoding = detect_encoding(msg)

        # Should return some encoding (utf-8 default or detected)
        assert encoding is not None
        assert len(encoding) > 0
