"""
API request and response models for FastAPI endpoints.

This module defines the Pydantic models used for API request/response validation.
"""

from typing import Optional
from pydantic import BaseModel, Field

from .email_document import EmailDocument
from .pipeline_version import PipelineVersion


class IngestEmlRequest(BaseModel):
    """Request model for EML ingestion endpoint."""

    pii_redaction_level: Optional[str] = Field(
        default="standard",
        description="PII redaction level: none, minimal, standard, aggressive",
        pattern="^(none|minimal|standard|aggressive)$",
    )

    store_html: bool = Field(
        default=False, description="Whether to store original HTML body"
    )


class IngestEmlResponse(BaseModel):
    """Response model for EML ingestion endpoint."""

    success: bool = Field(description="Whether ingestion succeeded")
    document: Optional[EmailDocument] = Field(None, description="Ingested document")
    error: Optional[str] = Field(None, description="Error message if failed")


class HealthResponse(BaseModel):
    """Health check response."""

    status: str = Field(description="Service status", example="healthy")
    version: str = Field(description="API version", example="1.0.0")
    uptime_seconds: float = Field(description="Service uptime")


class VersionResponse(BaseModel):
    """Version information response."""

    api_version: str = Field(description="API version")
    pipeline_version: PipelineVersion = Field(
        description="Current pipeline version"
    )
