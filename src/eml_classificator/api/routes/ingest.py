"""
Email ingestion endpoint - orchestrates the complete ingestion pipeline.
"""

from datetime import datetime, timezone
from time import time
from fastapi import APIRouter, File, HTTPException, Query, UploadFile
from fastapi.responses import JSONResponse
import structlog

from ...models.api_models import IngestEmlResponse
from ...models.email_document import EmailDocument
from ...parsing import (
    compute_document_id,
    detect_encoding,
    extract_attachments,
    extract_body,
    extract_headers,
    parse_eml_bytes,
)
from ...canonicalization import canonicalize_text, html_to_text
from ...pii import PIIRedactor, RedactionLevel
from ...version import get_current_pipeline_version
from ...config import settings

logger = structlog.get_logger(__name__)
router = APIRouter()


@router.post("/eml", response_model=IngestEmlResponse)
async def ingest_eml_file(
    file: UploadFile = File(..., description=".eml file to ingest"),
    pii_redaction_level: str = Query(
        default="standard",
        pattern="^(none|minimal|standard|aggressive)$",
        description="PII redaction level"
    ),
    store_html: bool = Query(default=False, description="Whether to store original HTML"),
    include_removed_sections: bool = Query(
        default=False,
        description="Whether to track and return removed text sections (signatures, quotes, disclaimers)"
    )
) -> IngestEmlResponse:
    """
    Ingest a .eml file and return canonicalized EmailDocument.

    This endpoint orchestrates the complete ingestion pipeline:
    1. Parses the .eml file (RFC5322/MIME)
    2. Extracts headers, body, and attachment metadata
    3. Converts HTML to text if needed
    4. Canonicalizes text (normalize, signatures, quotes)
    5. Redacts PII (emails, phone numbers)
    6. Returns EmailDocument with full audit trail

    Args:
        file: Uploaded .eml file
        pii_redaction_level: none, minimal, standard, aggressive
        store_html: Whether to include original HTML in response
        include_removed_sections: Whether to track removed sections (default False for smaller responses)

    Returns:
        IngestEmlResponse with EmailDocument or error
    """
    start_time = time()
    ingestion_timestamp = datetime.now(timezone.utc)

    try:
        # Validate file type
        if not file.filename or not file.filename.endswith('.eml'):
            raise HTTPException(
                status_code=400,
                detail="File must be .eml format"
            )

        # Read file content
        eml_bytes = await file.read()

        # Check size limit
        size_mb = len(eml_bytes) / (1024 * 1024)
        if size_mb > settings.max_email_size_mb:
            raise HTTPException(
                status_code=413,
                detail=f"File size ({size_mb:.1f}MB) exceeds maximum ({settings.max_email_size_mb}MB)"
            )

        logger.info(
            "Starting email ingestion",
            filename=file.filename,
            size_bytes=len(eml_bytes),
            pii_level=pii_redaction_level,
        )

        # 1. Parse email
        msg = parse_eml_bytes(eml_bytes)

        # 2. Extract components
        headers = extract_headers(msg)
        body_text, body_html = extract_body(msg)
        attachments = extract_attachments(msg)
        encoding = detect_encoding(msg)

        logger.info(
            "Email parsed",
            subject=headers.subject,
            from_address=headers.from_address,
            attachments_count=len(attachments),
            has_html=body_html is not None,
        )

        # 3. Convert HTML to text if no plain text available
        if not body_text and body_html:
            body_text = html_to_text(body_html)
            logger.info("Converted HTML to text")

        # Handle empty body
        if not body_text:
            body_text = ""

        # 4. Canonicalize text
        canonical_text, removed_sections = canonicalize_text(
            body_text,
            remove_signatures=True,
            keep_audit_info=include_removed_sections
        )

        logger.info(
            "Text canonicalized",
            original_length=len(body_text),
            canonical_length=len(canonical_text),
            removed_sections_count=len(removed_sections),
        )

        # 5. PII redaction
        redactor = PIIRedactor(
            level=RedactionLevel(pii_redaction_level),
            salt=settings.pii_salt
        )
        canonical_text, redaction_log = redactor.redact_text(canonical_text)

        # Check attachment limit
        if len(attachments) > settings.max_attachments:
            logger.warning(
                "Too many attachments",
                count=len(attachments),
                limit=settings.max_attachments,
            )
            attachments = attachments[:settings.max_attachments]

        logger.info(
            "PII redaction complete",
            redactions_count=len(redaction_log),
            pii_redacted=len(redaction_log) > 0,
        )

        # 6. Create EmailDocument
        document = EmailDocument(
            document_id=compute_document_id(eml_bytes, ingestion_timestamp),
            headers=headers,
            body_text=body_text,
            body_html=body_html if store_html else None,
            canonical_text=canonical_text,
            attachments=attachments,
            removed_sections=removed_sections,
            pii_redacted=len(redaction_log) > 0,
            pii_redaction_log=redaction_log,
            pipeline_version=get_current_pipeline_version(pii_redaction_level),
            ingestion_timestamp=ingestion_timestamp,
            raw_size_bytes=len(eml_bytes),
            encoding_detected=encoding,
            processing_time_ms=(time() - start_time) * 1000,
        )

        logger.info(
            "Email ingested successfully",
            document_id=document.document_id,
            subject=headers.subject,
            processing_time_ms=document.processing_time_ms,
            pii_redactions=len(redaction_log),
        )

        return IngestEmlResponse(success=True, document=document)

    except HTTPException:
        # Re-raise HTTP exceptions
        raise

    except Exception as e:
        logger.error(
            "Email ingestion failed",
            error=str(e),
            filename=file.filename if file else None,
            exc_info=True,
        )
        return IngestEmlResponse(
            success=False,
            error=f"Ingestion failed: {str(e)}"
        )
