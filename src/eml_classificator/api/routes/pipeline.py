"""
Full pipeline API route (Phase 1 → 2 → 3).

Provides end-to-end email processing:
- POST /api/v1/pipeline/complete - Full pipeline from .eml to classification
"""

from typing import Optional
from fastapi import APIRouter, HTTPException, UploadFile, File, Form, status
from pydantic import BaseModel
import structlog

from eml_classificator.parsing.eml_parser import parse_eml_bytes
from eml_classificator.models.email_document import EmailDocument
from eml_classificator.candidates import extract_candidates
from eml_classificator.classification.classifier import classify_email
from eml_classificator.classification.schemas import ClassificationResult


logger = structlog.get_logger(__name__)

router = APIRouter(prefix="/api/v1/pipeline", tags=["pipeline"])


# ============================================================================
# REQUEST/RESPONSE MODELS
# ============================================================================

class PipelineResponse(BaseModel):
    """Response for complete pipeline."""
    success: bool
    classification_result: Optional[ClassificationResult] = None
    error: Optional[str] = None

    # Phase outputs for debugging
    email_document_id: Optional[str] = None
    candidates_count: Optional[int] = None


# ============================================================================
# PIPELINE ENDPOINT
# ============================================================================

@router.post("/complete", response_model=PipelineResponse, status_code=status.HTTP_200_OK)
async def pipeline_complete_endpoint(
    eml_file: UploadFile = File(..., description=".eml file to process"),
    model_override: Optional[str] = Form(default=None, description="LLM model override"),
    dictionary_version: int = Form(default=1, description="Dictionary version")
) -> PipelineResponse:
    """
    Execute complete pipeline: Phase 1 → 2 → 3.

    Processes .eml file through all phases:
    1. Parse and normalize email
    2. Extract candidate keywords
    3. LLM classification

    Args:
        eml_file: Email file (.eml format)
        model_override: Optional LLM model (e.g., "ollama/deepseek-r1:7b")
        dictionary_version: Dictionary version for audit

    Returns:
        PipelineResponse with classification result

    Raises:
        HTTPException: On processing errors
    """
    logger.info(
        "pipeline_request_received",
        filename=eml_file.filename,
        model_override=model_override
    )

    try:
        # Phase 1: Parse and normalize email
        logger.debug("pipeline_phase_1_parsing")
        eml_bytes = await eml_file.read()

        # TODO: Implement full Phase 1 pipeline (parsing, canonicalization, PII redaction)
        # For now, this requires reimplementing the full ingestion logic from ingest route
        raise HTTPException(
            status_code=status.HTTP_501_NOT_IMPLEMENTED,
            detail="Full pipeline endpoint requires complete Phase 1 integration. "
                   "Use /api/v1/ingest + /api/v1/candidates + /api/v1/classify separately for now."
        )

        msg = parse_eml_bytes(eml_bytes)
        # email_document = build_email_document(msg)  # Function doesn't exist yet

        logger.info(
            "pipeline_phase_1_completed"
        )

        # Phase 2: Extract candidates
        logger.debug("pipeline_phase_2_candidates")
        candidates_result = extract_candidates(email_document)

        logger.info(
            "pipeline_phase_2_completed",
            candidates_count=len(candidates_result.rich_candidates)
        )

        # Phase 3: Classify
        logger.debug("pipeline_phase_3_classification")
        classification_result = classify_email(
            email_document=email_document.model_dump(),
            candidates=[c.model_dump() for c in candidates_result.rich_candidates],
            dictionary_version=dictionary_version,
            model_override=model_override
        )

        logger.info(
            "pipeline_completed",
            message_id=email_document.message_id,
            topics_count=len(classification_result.topics),
            total_latency_ms=classification_result.metadata.latency_ms
        )

        return PipelineResponse(
            success=True,
            classification_result=classification_result,
            email_document_id=email_document.message_id,
            candidates_count=len(candidates_result.rich_candidates)
        )

    except ValueError as e:
        logger.error("pipeline_validation_error", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Validation error: {str(e)}"
        )

    except Exception as e:
        logger.error("pipeline_failed", error=str(e), exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Pipeline failed: {str(e)}"
        )


@router.get("/health", status_code=status.HTTP_200_OK)
async def pipeline_health():
    """Check pipeline health (all phases)."""
    from eml_classificator.config import settings

    return {
        "status": "healthy",
        "phases": {
            "phase_1": "ready",  # Parsing always ready
            "phase_2": settings.enable_embeddings,  # KeyBERT enabled?
            "phase_3": bool(settings.llm_provider and settings.llm_model)
        },
        "llm_provider": settings.llm_provider,
        "llm_model": settings.llm_model
    }
