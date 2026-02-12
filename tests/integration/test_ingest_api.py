"""
Integration tests for ingestion API endpoint (/api/v1/ingest/eml).

Tests cover:
- Successful ingestion scenarios (plain text, HTML, multipart)
- Different PII redaction levels
- Attachment handling
- Signature and reply chain removal
- Document ID generation
- Pipeline version tracking
- Error handling (invalid files, size limits, malformed emails)
"""

import io

import pytest
import pytest_asyncio

from tests.fixtures.emails import SAMPLE_EMAILS

pytestmark = pytest.mark.asyncio


@pytest_asyncio.fixture
async def client(async_client):
    """Provide async HTTP client for integration tests."""
    yield async_client


class TestIngestEmlSuccess:
    """Tests for successful email ingestion scenarios."""

    @pytest.mark.integration
    async def test_ingest_eml_success_plain_text(self, client):
        """Test ingesting simple plain text email."""
        eml_bytes = SAMPLE_EMAILS["simple_plain_text"]
        files = {"file": ("test.eml", io.BytesIO(eml_bytes), "message/rfc822")}

        response = await client.post("/api/v1/ingest/eml", files=files)

        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert "document" in data
        doc = data["document"]
        assert doc["document_id"].startswith("eml-")
        assert doc["headers"]["from_address"] == "sender@example.com"
        assert "hello, this is a simple test email" in doc["body_text"].lower()
        assert len(doc["canonical_text"]) > 0

    @pytest.mark.integration
    async def test_ingest_eml_success_html(self, client, html_only_eml):
        """Test ingesting HTML-only email (conversion to text)."""
        files = {"file": ("test.eml", io.BytesIO(html_only_eml), "message/rfc822")}

        response = await client.post("/api/v1/ingest/eml", files=files)

        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        doc = data["document"]
        # HTML should be converted to text
        assert "welcome" in doc["canonical_text"].lower()
        assert "support@example.com" not in doc["canonical_text"]  # Should be redacted

    @pytest.mark.integration
    async def test_ingest_eml_success_multipart(self, client, multipart_html_eml):
        """Test ingesting multipart email with both plain and HTML."""
        files = {"file": ("test.eml", io.BytesIO(multipart_html_eml), "message/rfc822")}

        response = await client.post("/api/v1/ingest/eml", files=files)

        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        doc = data["document"]
        assert "newsletter" in doc["body_text"].lower()
        assert doc["body_html"] is None  # Not stored by default


class TestIngestEmlAttachments:
    """Tests for attachment handling."""

    @pytest.mark.integration
    async def test_ingest_eml_with_attachments(self, client, attachment_eml):
        """Test that attachment metadata is extracted."""
        files = {"file": ("test.eml", io.BytesIO(attachment_eml), "message/rfc822")}

        response = await client.post("/api/v1/ingest/eml", files=files)

        assert response.status_code == 200
        data = response.json()
        doc = data["document"]
        assert len(doc["attachments"]) == 1
        att = doc["attachments"][0]
        assert att["filename"] == "document.pdf"
        assert att["content_type"] == "application/pdf"
        assert att["size_bytes"] > 0

    @pytest.mark.integration
    async def test_ingest_eml_with_multiple_attachments(self, client):
        """Test handling multiple attachments."""
        eml_bytes = SAMPLE_EMAILS["multiple_attachments"]
        files = {"file": ("test.eml", io.BytesIO(eml_bytes), "message/rfc822")}

        response = await client.post("/api/v1/ingest/eml", files=files)

        assert response.status_code == 200
        doc = response.json()["document"]
        assert len(doc["attachments"]) == 3
        filenames = [att["filename"] for att in doc["attachments"]]
        assert "report.pdf" in filenames
        assert "chart.png" in filenames
        assert "data.csv" in filenames


class TestIngestEmlPIIRedaction:
    """Tests for PII redaction levels."""

    @pytest.mark.integration
    async def test_ingest_eml_pii_redaction_none(self, client, pii_email_bytes):
        """Test with no PII redaction."""
        files = {"file": ("test.eml", io.BytesIO(pii_email_bytes), "message/rfc822")}
        params = {"pii_redaction_level": "none"}

        response = await client.post("/api/v1/ingest/eml", files=files, params=params)

        assert response.status_code == 200
        doc = response.json()["document"]
        assert doc["pii_redacted"] is False
        assert len(doc["pii_redaction_log"]) == 0
        # PII should still be in canonical text
        assert "@" in doc["canonical_text"] or "gmail" in doc["canonical_text"]

    @pytest.mark.integration
    async def test_ingest_eml_pii_redaction_minimal(self, client, pii_email_bytes):
        """Test minimal PII redaction (emails only)."""
        files = {"file": ("test.eml", io.BytesIO(pii_email_bytes), "message/rfc822")}
        params = {"pii_redaction_level": "minimal"}

        response = await client.post("/api/v1/ingest/eml", files=files, params=params)

        assert response.status_code == 200
        doc = response.json()["document"]
        assert doc["pii_redacted"] is True
        # Should have email redactions only
        email_redactions = [e for e in doc["pii_redaction_log"] if e["type"] == "email"]
        phone_redactions = [e for e in doc["pii_redaction_log"] if e["type"] == "phone"]
        assert len(email_redactions) > 0
        assert len(phone_redactions) == 0

    @pytest.mark.integration
    async def test_ingest_eml_pii_redaction_standard(self, client, pii_email_bytes):
        """Test standard PII redaction (emails + phones)."""
        files = {"file": ("test.eml", io.BytesIO(pii_email_bytes), "message/rfc822")}
        params = {"pii_redaction_level": "standard"}

        response = await client.post("/api/v1/ingest/eml", files=files, params=params)

        assert response.status_code == 200
        doc = response.json()["document"]
        assert doc["pii_redacted"] is True
        # Should have both email and phone redactions
        email_redactions = [e for e in doc["pii_redaction_log"] if e["type"] == "email"]
        phone_redactions = [e for e in doc["pii_redaction_log"] if e["type"] == "phone"]
        assert len(email_redactions) > 0
        assert len(phone_redactions) > 0

    @pytest.mark.integration
    async def test_ingest_eml_pii_audit_log(self, client, pii_email_bytes):
        """Test that PII audit log is correctly populated."""
        files = {"file": ("test.eml", io.BytesIO(pii_email_bytes), "message/rfc822")}

        response = await client.post("/api/v1/ingest/eml", files=files)

        assert response.status_code == 200
        doc = response.json()["document"]
        assert len(doc["pii_redaction_log"]) > 0

        # Check first log entry structure
        entry = doc["pii_redaction_log"][0]
        assert "type" in entry
        assert "position" in entry
        assert "original_hash" in entry
        assert "placeholder" in entry
        assert "pii_redaction_version" in entry
        assert "pii_redaction_level" in entry


class TestIngestEmlStoreHtml:
    """Tests for HTML storage option."""

    @pytest.mark.integration
    async def test_ingest_eml_store_html_true(self, client, html_only_eml):
        """Test returning HTML when store_html=true."""
        files = {"file": ("test.eml", io.BytesIO(html_only_eml), "message/rfc822")}
        params = {"store_html": "true"}

        response = await client.post("/api/v1/ingest/eml", files=files, params=params)

        assert response.status_code == 200
        doc = response.json()["document"]
        assert doc["body_html"] is not None
        assert "<html>" in doc["body_html"] or "<body>" in doc["body_html"]

    @pytest.mark.integration
    async def test_ingest_eml_store_html_false(self, client, html_only_eml):
        """Test that HTML is not returned when store_html=false."""
        files = {"file": ("test.eml", io.BytesIO(html_only_eml), "message/rfc822")}
        params = {"store_html": "false"}

        response = await client.post("/api/v1/ingest/eml", files=files, params=params)

        assert response.status_code == 200
        doc = response.json()["document"]
        assert doc["body_html"] is None


class TestIngestEmlTextProcessing:
    """Tests for text processing (signatures, reply chains)."""

    @pytest.mark.integration
    async def test_ingest_eml_signature_removal(self, client, italian_signature_eml):
        """Test that Italian signatures are removed."""
        files = {"file": ("test.eml", io.BytesIO(italian_signature_eml), "message/rfc822")}

        response = await client.post("/api/v1/ingest/eml", files=files)

        assert response.status_code == 200
        doc = response.json()["document"]
        canonical = doc["canonical_text"]
        # Signature should be removed
        assert "cordiali saluti" not in canonical
        assert "mario rossi" not in canonical
        # But body content should remain
        assert "informazioni" in canonical or "servizio" in canonical

    @pytest.mark.integration
    async def test_ingest_eml_reply_chain_removal(self, client, reply_chain_eml):
        """Test that quoted reply chains are removed."""
        files = {"file": ("test.eml", io.BytesIO(reply_chain_eml), "message/rfc822")}

        response = await client.post("/api/v1/ingest/eml", files=files)

        assert response.status_code == 200
        doc = response.json()["document"]
        canonical = doc["canonical_text"]
        # Reply content should be in canonical text
        assert "thank you" in canonical
        # Quoted text should be removed
        assert ">" not in canonical or canonical.count(">") < 2


class TestIngestEmlDeterminism:
    """Tests for deterministic processing."""

    @pytest.mark.integration
    async def test_ingest_eml_document_id_format(self, client, sample_eml_bytes):
        """Test that document ID has correct format."""
        files = {"file": ("test.eml", io.BytesIO(sample_eml_bytes), "message/rfc822")}

        response = await client.post("/api/v1/ingest/eml", files=files)

        assert response.status_code == 200
        doc = response.json()["document"]
        assert doc["document_id"].startswith("eml-")
        assert len(doc["document_id"]) == 36  # "eml-" + 32 hex chars


class TestIngestEmlVersionTracking:
    """Tests for pipeline version tracking."""

    @pytest.mark.integration
    async def test_ingest_eml_pipeline_version(self, client, sample_eml_bytes):
        """Test that PipelineVersion is included in response."""
        files = {"file": ("test.eml", io.BytesIO(sample_eml_bytes), "message/rfc822")}

        response = await client.post("/api/v1/ingest/eml", files=files)

        assert response.status_code == 200
        doc = response.json()["document"]
        assert "pipeline_version" in doc
        pv = doc["pipeline_version"]
        # Check required version fields
        assert "parser_version" in pv
        assert "canonicalization_version" in pv
        assert "pii_redaction_version" in pv
        assert "pii_redaction_level" in pv


class TestIngestEmlErrorHandling:
    """Tests for error handling."""

    @pytest.mark.integration
    async def test_ingest_eml_invalid_file_extension(self, client):
        """Test rejection of non-.eml files."""
        files = {"file": ("test.txt", io.BytesIO(b"Not an eml"), "text/plain")}

        response = await client.post("/api/v1/ingest/eml", files=files)

        assert response.status_code == 400
        assert "must be .eml format" in response.json()["detail"].lower()

    @pytest.mark.integration
    async def test_ingest_eml_file_too_large(self, client):
        """Test rejection of files exceeding size limit."""
        # Create a file larger than the limit (25MB default)
        large_content = b"X" * (26 * 1024 * 1024)  # 26 MB
        large_eml = b"""From: sender@example.com
To: recipient@example.com
Subject: Large email
Content-Type: text/plain

""" + large_content

        files = {"file": ("large.eml", io.BytesIO(large_eml), "message/rfc822")}

        response = await client.post("/api/v1/ingest/eml", files=files)

        assert response.status_code == 413
        assert "exceeds maximum" in response.json()["detail"].lower()

    @pytest.mark.integration
    async def test_ingest_eml_invalid_rfc5322(self, client, malformed_eml):
        """Test handling of malformed email gracefully."""
        files = {"file": ("bad.eml", io.BytesIO(malformed_eml), "message/rfc822")}

        response = await client.post("/api/v1/ingest/eml", files=files)

        # Should not crash - either succeed with partial data or return error
        assert response.status_code in [200, 400, 500]
        # If successful, document should exist even if minimal
        if response.status_code == 200:
            data = response.json()
            if data["success"]:
                assert "document" in data

    @pytest.mark.integration
    async def test_ingest_eml_empty_body(self, client):
        """Test handling of emails with no body content."""
        empty_body_eml = SAMPLE_EMAILS["empty_body"]
        files = {"file": ("empty.eml", io.BytesIO(empty_body_eml), "message/rfc822")}

        response = await client.post("/api/v1/ingest/eml", files=files)

        assert response.status_code == 200
        doc = response.json()["document"]
        # Should handle empty body gracefully
        assert doc["canonical_text"] == "" or len(doc["canonical_text"]) == 0


class TestIngestEmlMetrics:
    """Tests for metrics and metadata."""

    @pytest.mark.integration
    async def test_ingest_eml_processing_time(self, client, sample_eml_bytes):
        """Test that processing_time_ms is populated."""
        files = {"file": ("test.eml", io.BytesIO(sample_eml_bytes), "message/rfc822")}

        response = await client.post("/api/v1/ingest/eml", files=files)

        assert response.status_code == 200
        doc = response.json()["document"]
        assert "processing_time_ms" in doc
        assert doc["processing_time_ms"] > 0

    @pytest.mark.integration
    async def test_ingest_eml_encoding_detection(self, client):
        """Test that encoding_detected field is populated."""
        eml_bytes = SAMPLE_EMAILS["latin1_encoding"]
        files = {"file": ("latin.eml", io.BytesIO(eml_bytes), "message/rfc822")}

        response = await client.post("/api/v1/ingest/eml", files=files)

        assert response.status_code == 200
        doc = response.json()["document"]
        assert "encoding_detected" in doc
        assert doc["encoding_detected"] is not None

    @pytest.mark.integration
    async def test_ingest_eml_raw_size_bytes(self, client, sample_eml_bytes):
        """Test that raw_size_bytes is tracked."""
        files = {"file": ("test.eml", io.BytesIO(sample_eml_bytes), "message/rfc822")}

        response = await client.post("/api/v1/ingest/eml", files=files)

        assert response.status_code == 200
        doc = response.json()["document"]
        assert "raw_size_bytes" in doc
        assert doc["raw_size_bytes"] == len(sample_eml_bytes)

    @pytest.mark.integration
    async def test_ingest_eml_ingestion_timestamp(self, client, sample_eml_bytes):
        """Test that ingestion_timestamp is recorded."""
        files = {"file": ("test.eml", io.BytesIO(sample_eml_bytes), "message/rfc822")}

        response = await client.post("/api/v1/ingest/eml", files=files)

        assert response.status_code == 200
        doc = response.json()["document"]
        assert "ingestion_timestamp" in doc
        assert doc["ingestion_timestamp"] is not None


class TestRemovedSectionsTracking:
    """Integration tests for removed_sections tracking via API."""

    @pytest.mark.integration
    async def test_removed_sections_default_not_included(self, client):
        """Test that removed_sections is empty by default (no tracking)."""
        eml_bytes = SAMPLE_EMAILS["italian_signature"]
        files = {"file": ("test.eml", io.BytesIO(eml_bytes), "message/rfc822")}

        response = await client.post("/api/v1/ingest/eml", files=files)

        assert response.status_code == 200
        doc = response.json()["document"]
        # Default is include_removed_sections=False, so should be empty
        assert "removed_sections" in doc
        assert len(doc["removed_sections"]) == 0

    @pytest.mark.integration
    async def test_removed_sections_with_tracking_enabled(self, client):
        """Test removed_sections populated when include_removed_sections=true."""
        eml_bytes = SAMPLE_EMAILS["italian_signature"]
        files = {"file": ("test.eml", io.BytesIO(eml_bytes), "message/rfc822")}

        response = await client.post(
            "/api/v1/ingest/eml?include_removed_sections=true", files=files
        )

        assert response.status_code == 200
        doc = response.json()["document"]
        # Should have tracked removals
        assert "removed_sections" in doc
        assert len(doc["removed_sections"]) > 0

        # Check structure of removed sections
        for section in doc["removed_sections"]:
            assert "type" in section
            assert section["type"] in [
                "signature",
                "quote",
                "reply_header",
                "disclaimer",
            ]
            assert "span_start" in section
            assert "span_end" in section
            assert "content" in section
            assert section["span_start"] >= 0
            assert section["span_end"] >= section["span_start"]

    @pytest.mark.integration
    async def test_signature_removal_tracked(self, client):
        """Test that signature removal is logged in removed_sections."""
        eml_bytes = SAMPLE_EMAILS["italian_signature"]
        files = {"file": ("test.eml", io.BytesIO(eml_bytes), "message/rfc822")}

        response = await client.post(
            "/api/v1/ingest/eml?include_removed_sections=true", files=files
        )

        assert response.status_code == 200
        doc = response.json()["document"]

        # Should have signature removal tracked
        removed_signatures = [
            r for r in doc["removed_sections"] if r["type"] == "signature"
        ]
        assert len(removed_signatures) > 0

        # Check content captured
        sig = removed_signatures[0]
        assert len(sig["content"]) > 0
        # Italian signatures typically have "Cordiali saluti" or similar
        assert any(
            phrase in sig["content"].lower()
            for phrase in ["cordiali", "saluti", "distinti", "--"]
        )

    @pytest.mark.integration
    async def test_reply_chain_removal_tracked(self, client):
        """Test that reply chain removal is tracked."""
        eml_bytes = SAMPLE_EMAILS["reply_chain"]
        files = {"file": ("test.eml", io.BytesIO(eml_bytes), "message/rfc822")}

        response = await client.post(
            "/api/v1/ingest/eml?include_removed_sections=true", files=files
        )

        assert response.status_code == 200
        doc = response.json()["document"]

        # Should have reply/quote removal tracked
        removed_replies = [
            r
            for r in doc["removed_sections"]
            if r["type"] in ("quote", "reply_header")
        ]
        assert len(removed_replies) > 0

        # Check content has quote markers or reply headers
        reply_content = " ".join(r["content"] for r in removed_replies)
        assert any(marker in reply_content for marker in [">", "wrote:", "ha scritto:"])

    @pytest.mark.integration
    async def test_removed_sections_span_positions(self, client):
        """Test that span positions are accurate."""
        eml_bytes = SAMPLE_EMAILS["italian_signature"]
        files = {"file": ("test.eml", io.BytesIO(eml_bytes), "message/rfc822")}

        response = await client.post(
            "/api/v1/ingest/eml?include_removed_sections=true", files=files
        )

        assert response.status_code == 200
        doc = response.json()["document"]

        # Verify spans are sensible
        for section in doc["removed_sections"]:
            assert section["span_start"] < section["span_end"]
            assert len(section["content"]) == section["span_end"] - section["span_start"]

