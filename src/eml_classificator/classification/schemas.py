"""
Classification schemas for Phase 3: LLM Layer

Defines Pydantic models for:
- Topic assignments (multi-label)
- Sentiment analysis
- Priority triage
- Customer status
- Classification results

Based on brainstorming v2/v3 specifications with strict validation.
"""

from enum import Enum
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field, field_validator, model_validator
from datetime import datetime


# ============================================================================
# ENUMS
# ============================================================================

class TopicLabel(str, Enum):
    """
    Italian topic taxonomy for customer service emails.

    From brainstorming v3, section 6.
    """
    FATTURAZIONE = "FATTURAZIONE"
    ASSISTENZA_TECNICA = "ASSISTENZA_TECNICA"
    RECLAMO = "RECLAMO"
    INFO_COMMERCIALI = "INFO_COMMERCIALI"
    DOCUMENTI = "DOCUMENTI"
    APPUNTAMENTO = "APPUNTAMENTO"
    CONTRATTO = "CONTRATTO"
    GARANZIA = "GARANZIA"
    SPEDIZIONE = "SPEDIZIONE"
    UNKNOWN_TOPIC = "UNKNOWN_TOPIC"


class SentimentValue(str, Enum):
    """Sentiment analysis values."""
    POSITIVE = "positive"
    NEUTRAL = "neutral"
    NEGATIVE = "negative"


class PriorityValue(str, Enum):
    """Priority triage values (ordinal)."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    URGENT = "urgent"


class CustomerStatusValue(str, Enum):
    """Customer status values (deterministic via CRM)."""
    NEW = "new"
    EXISTING = "existing"
    UNKNOWN = "unknown"


# ============================================================================
# KEYWORD & EVIDENCE MODELS
# ============================================================================

class KeywordInText(BaseModel):
    """
    Keyword extracted from email text.

    MUST reference a candidate_id from the input candidate list.
    Validated in multi-stage validation (validators.py).
    """
    candidate_id: str = Field(
        ...,
        description="MUST match a candidate_id from input candidate list"
    )
    lemma: str = Field(..., description="Lemmatized form of the keyword")
    count: int = Field(..., ge=1, description="Frequency count in text")
    spans: Optional[List[List[int]]] = Field(
        default=None,
        description="Optional [[start, end], ...] positions in text"
    )

    @field_validator('spans')
    @classmethod
    def validate_spans(cls, v):
        """Validate span format: [[start, end], ...]"""
        if v is not None:
            for span in v:
                if not isinstance(span, list) or len(span) != 2:
                    raise ValueError("Each span must be [start, end]")
                if not all(isinstance(x, int) for x in span):
                    raise ValueError("Span positions must be integers")
                if span[0] >= span[1]:
                    raise ValueError(f"Invalid span {span}: start >= end")
        return v


class Evidence(BaseModel):
    """
    Text evidence supporting a topic assignment.

    Provides audit trail for classification decisions.
    """
    quote: str = Field(
        ...,
        max_length=200,
        description="Exact quote from email supporting this topic"
    )
    span: Optional[List[int]] = Field(
        default=None,
        description="Optional [start, end] position in text"
    )

    @field_validator('span')
    @classmethod
    def validate_span(cls, v):
        """Validate span format: [start, end]"""
        if v is not None:
            if len(v) != 2:
                raise ValueError("Span must be [start, end]")
            if not all(isinstance(x, int) for x in v):
                raise ValueError("Span positions must be integers")
            if v[0] >= v[1]:
                raise ValueError(f"Invalid span {v}: start >= end")
        return v


# ============================================================================
# TOPIC ASSIGNMENT
# ============================================================================

class TopicAssignment(BaseModel):
    """
    Single topic assignment with keywords and evidence.

    Part of multi-label classification result.
    Constraints from PARSE optimization (v2, section 3.2):
    - maxItems for keywords: 15
    - maxItems for evidence: 2
    - minItems: 1 each
    """
    label_id: TopicLabel = Field(..., description="Topic label from taxonomy")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence score 0-1")

    keywords_in_text: List[KeywordInText] = Field(
        ...,
        min_length=1,
        max_length=15,
        description="Keywords from candidate list supporting this topic"
    )

    evidence: List[Evidence] = Field(
        ...,
        min_length=1,
        max_length=2,
        description="Text evidence justifying topic assignment"
    )

    # Adjusted confidence after post-processing (see confidence.py)
    confidence_adjusted: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Adjusted confidence after keyword quality + evidence + collision analysis"
    )


# ============================================================================
# SENTIMENT, PRIORITY, CUSTOMER STATUS
# ============================================================================

class Sentiment(BaseModel):
    """Sentiment analysis result."""
    value: SentimentValue = Field(..., description="Sentiment classification")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence score 0-1")


class Priority(BaseModel):
    """
    Priority triage result.

    Computed via rule-based scoring (see priority_scorer.py), not LLM opinion.
    """
    value: PriorityValue = Field(..., description="Priority level")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence score 0-1")

    signals: List[str] = Field(
        default_factory=list,
        max_length=10,
        description="Keywords/phrases justifying priority (audit trail)"
    )

    raw_score: Optional[float] = Field(
        default=None,
        description="Raw priority score before bucketing"
    )


class CustomerStatus(BaseModel):
    """
    Customer status determination.

    MUST be deterministic via CRM lookup (see customer_status.py), not LLM opinion.
    """
    value: CustomerStatusValue = Field(..., description="Customer status")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence score 0-1")

    source: str = Field(
        ...,
        description="Source of determination: 'crm_exact_match' | 'crm_domain_match' | 'text_signal' | 'no_crm_no_signal' | 'lookup_failed'"
    )


# ============================================================================
# CLASSIFICATION REQUEST & RESULT
# ============================================================================

class ClassificationRequest(BaseModel):
    """
    Request for email classification (Phase 3 input).

    Receives EmailDocument from Phase 1 and CandidateExtractionResult from Phase 2.
    """
    # Input data (simplified - in practice would import from other modules)
    email_document: Dict[str, Any] = Field(..., description="EmailDocument from Phase 1")
    candidates: Dict[str, Any] = Field(..., description="CandidateExtractionResult from Phase 2")

    # Configuration overrides
    model_override: Optional[str] = Field(
        default=None,
        description="Override LLM model (format: 'provider/model-name', e.g., 'ollama/deepseek-r1:7b')"
    )

    priority_weights_override: Optional[Dict[str, float]] = Field(
        default=None,
        description="Override priority scoring weights"
    )

    # Audit trail
    dictionary_version: int = Field(..., description="Dictionary version for determinism")

    class Config:
        json_schema_extra = {
            "example": {
                "email_document": {
                    "message_id": "<msg123@example.com>",
                    "subject": "Problema con fattura",
                    "body_canonical": "Buongiorno, ho ricevuto una fattura errata..."
                },
                "candidates": {
                    "rich_candidates": [
                        {"candidate_id": "abc123", "term": "fattura", "composite_score": 0.95}
                    ]
                },
                "model_override": "ollama/deepseek-r1:7b",
                "dictionary_version": 1
            }
        }


class ClassificationMetadata(BaseModel):
    """Metadata about the classification process."""
    pipeline_version: Dict[str, Any] = Field(..., description="Complete pipeline version info")
    model_used: str = Field(..., description="LLM model used (provider/model)")

    # Token usage tracking
    tokens_input: Optional[int] = Field(default=None, description="Input tokens sent to LLM")
    tokens_output: Optional[int] = Field(default=None, description="Output tokens from LLM")
    tokens_total: Optional[int] = Field(default=None, description="Total tokens")

    # Performance metrics
    latency_ms: int = Field(..., description="Total classification latency in milliseconds")
    llm_latency_ms: Optional[int] = Field(default=None, description="LLM call latency only")

    # Validation info
    validation_warnings: List[str] = Field(
        default_factory=list,
        description="Non-fatal validation warnings"
    )
    retry_count: int = Field(default=0, description="Number of LLM retries due to validation failures")

    # Timestamp
    classified_at: datetime = Field(default_factory=datetime.utcnow, description="Classification timestamp")


class ClassificationResult(BaseModel):
    """
    Complete classification result (Phase 3 output).

    Combines:
    - LLM classification (topics, sentiment)
    - Rule-based priority scoring
    - Deterministic customer status
    - Adjusted confidence scores
    """
    # Dictionary version for audit trail
    dictionary_version: int = Field(..., description="Dictionary version used")

    # Main classification outputs
    topics: List[TopicAssignment] = Field(
        ...,
        min_length=1,
        max_length=5,
        description="Multi-label topic assignments (1-5 topics)"
    )

    sentiment: Sentiment = Field(..., description="Sentiment analysis result")
    priority: Priority = Field(..., description="Priority triage result")
    customer_status: CustomerStatus = Field(..., description="Customer status determination")

    # Metadata
    metadata: ClassificationMetadata = Field(..., description="Classification process metadata")

    @model_validator(mode='after')
    def validate_unknown_topic_threshold(self):
        """
        Ensure at least one topic is assigned.
        If no topics or all low confidence, should include UNKNOWN_TOPIC.
        """
        if not self.topics:
            raise ValueError("At least one topic must be assigned")
        return self

    class Config:
        json_schema_extra = {
            "example": {
                "dictionary_version": 1,
                "topics": [
                    {
                        "label_id": "FATTURAZIONE",
                        "confidence": 0.92,
                        "keywords_in_text": [
                            {"candidate_id": "abc123", "lemma": "fattura", "count": 3}
                        ],
                        "evidence": [
                            {"quote": "ho ricevuto una fattura errata"}
                        ]
                    }
                ],
                "sentiment": {
                    "value": "negative",
                    "confidence": 0.85
                },
                "priority": {
                    "value": "high",
                    "confidence": 0.88,
                    "signals": ["fattura", "errata", "problema"],
                    "raw_score": 5.2
                },
                "customer_status": {
                    "value": "existing",
                    "confidence": 0.95,
                    "source": "crm_exact_match"
                },
                "metadata": {
                    "model_used": "ollama/deepseek-r1:7b",
                    "latency_ms": 2341,
                    "validation_warnings": []
                }
            }
        }


# ============================================================================
# LLM RAW OUTPUT (for validation)
# ============================================================================

class LLMRawOutput(BaseModel):
    """
    Raw LLM output before validation and post-processing.

    Used internally for validation pipeline (validators.py).
    This is what the LLM returns directly via tool calling.
    """
    dictionary_version: int
    topics: List[TopicAssignment]
    sentiment: Sentiment
    priority: Priority
    customer_status: CustomerStatus

    class Config:
        # Allow extra fields from LLM (will be filtered)
        extra = "allow"
