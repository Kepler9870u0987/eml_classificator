"""
Classification API routes for Phase 3.

Provides REST endpoints for email classification:
- POST /api/v1/classify - Single email classification
- POST /api/v1/classify/batch - Batch classification
"""

from typing import Dict, Any, List, Optional
from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel, Field
import structlog

from eml_classificator.classification.classifier import classify_email, classify_batch
from eml_classificator.classification.schemas import ClassificationResult


logger = structlog.get_logger(__name__)

router = APIRouter(prefix="/api/v1", tags=["classification"])


# ============================================================================
# REQUEST/RESPONSE MODELS
# ============================================================================

class ClassifyRequest(BaseModel):
    """Request model for email classification."""
    email_document: Dict[str, Any] = Field(..., description="EmailDocument from Phase 1")
    candidates: List[Dict[str, Any]] = Field(..., description="Candidates from Phase 2")
    dictionary_version: int = Field(..., description="Dictionary version for audit")
    model_override: Optional[str] = Field(
        default=None,
        description="Override LLM model (e.g., 'ollama/deepseek-r1:7b')"
    )
    priority_weights_override: Optional[Dict[str, float]] = Field(
        default=None,
        description="Override priority scoring weights"
    )

    class Config:
        json_schema_extra = {
            "example": {
                "email_document": {
                    "message_id": "<msg123@example.com>",
                    "subject": "Problema con fattura",
                    "from_raw": "cliente@example.com",
                    "body_canonical": "Buongiorno, ho ricevuto la fattura n. 12345..."
                },
                "candidates": [
                    {
                        "candidate_id": "abc123",
                        "term": "fattura",
                        "lemma": "fattura",
                        "count": 3,
                        "source": "body",
                        "composite_score": 0.95
                    }
                ],
                "dictionary_version": 1,
                "model_override": "ollama/deepseek-r1:7b"
            }
        }


class ClassifyBatchRequest(BaseModel):
    """Request model for batch classification."""
    items: List[ClassifyRequest] = Field(..., description="List of classification requests")
    model_override: Optional[str] = Field(default=None, description="Model override for all items")


class ClassifyResponse(BaseModel):
    """Response model for classification."""
    success: bool
    result: Optional[ClassificationResult] = None
    error: Optional[str] = None


class ClassifyBatchResponse(BaseModel):
    """Response model for batch classification."""
    success: bool
    total_count: int
    success_count: int
    failure_count: int
    results: List[ClassificationResult]
    errors: List[Dict[str, str]] = Field(default_factory=list)


# ============================================================================
# ENDPOINTS
# ============================================================================

@router.post("/classify", response_model=ClassifyResponse, status_code=status.HTTP_200_OK)
async def classify_endpoint(request: ClassifyRequest) -> ClassifyResponse:
    """
    Classify a single email.

    Args:
        request: Classification request with email, candidates, config

    Returns:
        ClassificationResult with topics, sentiment, priority, customer_status

    Raises:
        HTTPException: On classification errors
    """
    logger.info(
        "classification_request_received",
        message_id=request.email_document.get("message_id"),
        model_override=request.model_override
    )

    try:
        result = classify_email(
            email_document=request.email_document,
            candidates=request.candidates,
            dictionary_version=request.dictionary_version,
            model_override=request.model_override,
            priority_weights_override=request.priority_weights_override
        )

        logger.info(
            "classification_completed",
            message_id=request.email_document.get("message_id"),
            topics_count=len(result.topics),
            latency_ms=result.metadata.latency_ms
        )

        return ClassifyResponse(success=True, result=result)

    except ValueError as e:
        logger.error("classification_validation_error", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Validation error: {str(e)}"
        )

    except Exception as e:
        logger.error("classification_failed", error=str(e), exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Classification failed: {str(e)}"
        )


@router.post("/classify/batch", response_model=ClassifyBatchResponse, status_code=status.HTTP_200_OK)
async def classify_batch_endpoint(request: ClassifyBatchRequest) -> ClassifyBatchResponse:
    """
    Classify multiple emails in batch.

    Args:
        request: Batch classification request

    Returns:
        ClassifyBatchResponse with results and statistics
    """
    logger.info("batch_classification_request_received", items_count=len(request.items))

    results = []
    errors = []

    # Extract model override
    model_override = request.model_override

    for idx, item in enumerate(request.items):
        try:
            # Use batch-level model override if not specified per-item
            effective_model = item.model_override or model_override

            result = classify_email(
                email_document=item.email_document,
                candidates=item.candidates,
                dictionary_version=item.dictionary_version,
                model_override=effective_model,
                priority_weights_override=item.priority_weights_override
            )

            results.append(result)

        except Exception as e:
            message_id = item.email_document.get("message_id", f"item_{idx}")
            logger.error(
                "batch_item_classification_failed",
                message_id=message_id,
                error=str(e)
            )
            errors.append({
                "message_id": message_id,
                "error": str(e)
            })

    logger.info(
        "batch_classification_completed",
        total_count=len(request.items),
        success_count=len(results),
        failure_count=len(errors)
    )

    return ClassifyBatchResponse(
        success=len(errors) == 0,
        total_count=len(request.items),
        success_count=len(results),
        failure_count=len(errors),
        results=results,
        errors=errors
    )


# ============================================================================
# HEALTH CHECK
# ============================================================================

@router.get("/classify/health", status_code=status.HTTP_200_OK)
async def classification_health() -> Dict[str, Any]:
    """
    Check classification service health.

    Returns:
        Health status with provider info
    """
    from eml_classificator.config import settings
    from eml_classificator.classification.customer_status import check_crm_availability

    # Check LLM provider configured
    llm_configured = bool(settings.llm_provider and settings.llm_model)

    # Check CRM availability
    crm_available = check_crm_availability()

    # Check Ollama connectivity (if configured)
    ollama_available = False
    if settings.llm_provider == "ollama":
        try:
            import ollama
            client = ollama.Client(host=settings.llm_api_base_url)
            models = client.list()
            ollama_available = True
        except Exception as e:
            logger.warning("ollama_health_check_failed", error=str(e))

    return {
        "status": "healthy" if llm_configured else "degraded",
        "llm_provider": settings.llm_provider,
        "llm_model": settings.llm_model,
        "llm_configured": llm_configured,
        "crm_available": crm_available,
        "ollama_available": ollama_available if settings.llm_provider == "ollama" else None
    }
