"""
Email document model - canonical representation after ingestion.

This module defines the core data structures for representing parsed and normalized emails.
"""

from datetime import datetime
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field

from .canonicalization import RemovedSection
from .pipeline_version import PipelineVersion


class AttachmentMetadata(BaseModel):
    """Metadata for email attachments (content not stored)."""

    filename: Optional[str] = Field(None, description="Original filename")
    content_type: str = Field(description="MIME type")
    size_bytes: int = Field(description="Size in bytes")
    content_id: Optional[str] = Field(None, description="Content-ID for inline images")


class EmailHeaders(BaseModel):
    """Parsed email headers (subset for classification)."""

    from_address: str = Field(description="From address (potentially redacted)")
    to_addresses: List[str] = Field(
        default_factory=list, description="To addresses"
    )
    cc_addresses: List[str] = Field(
        default_factory=list, description="CC addresses"
    )
    subject: str = Field(description="Email subject")
    date: Optional[datetime] = Field(None, description="Parsed date")
    message_id: Optional[str] = Field(None, description="Message-ID header")
    in_reply_to: Optional[str] = Field(
        None, description="In-Reply-To header for threading"
    )
    references: List[str] = Field(
        default_factory=list, description="References header for threading"
    )

    # Additional headers as dict
    extra_headers: Dict[str, str] = Field(
        default_factory=dict, description="Other headers that might be useful"
    )


class EmailDocument(BaseModel):
    """
    Canonical email document representation after ingestion.

    This is the output of the Ingestion Layer and input to downstream components.
    Guaranteed to have consistent structure with complete version tracking for audit.
    """

    # Identifiers
    document_id: str = Field(
        description="Unique document ID (hash-based for determinism)"
    )

    # Headers
    headers: EmailHeaders = Field(description="Parsed headers")

    # Body content (canonicalized)
    body_text: str = Field(description="Canonicalized plain text body")
    body_html: Optional[str] = Field(None, description="Original HTML body (if present)")
    canonical_text: str = Field(
        description="Final canonicalized text for classification"
    )

    # Attachments
    attachments: List[AttachmentMetadata] = Field(
        default_factory=list, description="Attachment metadata (content not stored)"
    )

    # Canonicalization tracking
    removed_sections: List[RemovedSection] = Field(
        default_factory=list,
        description="Audit log of removed text sections (signatures, quotes, disclaimers)",
    )

    # PII Redaction tracking
    pii_redacted: bool = Field(description="Whether PII redaction was performed")
    pii_redaction_log: List[Dict[str, Any]] = Field(
        default_factory=list, description="Audit log of redaction actions"
    )

    # Processing metadata
    pipeline_version: PipelineVersion = Field(
        description="Version contract for this processing"
    )
    ingestion_timestamp: datetime = Field(
        default_factory=datetime.utcnow, description="When this document was ingested"
    )
    processing_time_ms: Optional[float] = Field(
        None, description="Processing time in milliseconds"
    )

    # Raw email metadata
    raw_size_bytes: int = Field(description="Original .eml file size")
    encoding_detected: Optional[str] = Field(
        None, description="Detected character encoding"
    )

    model_config = {
        "json_schema_extra": {
            "example": {
                "document_id": "eml-sha256-abc123...",
                "headers": {
                    "from_address": "customer@example.com",
                    "to_addresses": ["support@company.com"],
                    "cc_addresses": [],
                    "subject": "Problema con ordine #12345",
                    "date": "2026-02-12T10:30:00Z",
                    "message_id": "<abc123@example.com>",
                    "in_reply_to": None,
                    "references": [],
                    "extra_headers": {},
                },
                "body_text": "Buongiorno, ho un problema...",
                "canonical_text": "buongiorno problema...",
                "pii_redacted": True,
                "attachments": [],
                "processing_time_ms": 125.3,
            }
        }
    }
