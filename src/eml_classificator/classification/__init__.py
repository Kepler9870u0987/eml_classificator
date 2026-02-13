"""
Classification package for Phase 3: LLM Layer

This module implements email classification using LLM-based multi-label classification
with tool calling, validation, and post-processing.

Main components:
- schemas: Pydantic models for classification I/O
- llm_client: LLM client abstraction (Ollama, OpenAI, DeepSeek)
- classifier: Main orchestration logic
- validators: Multi-stage validation pipeline
- prompts: Versioned prompt templates
- customer_status: Mock CRM implementation
- priority_scorer: Rule-based priority scoring
- confidence: Confidence adjustment logic
"""

from eml_classificator.classification.schemas import (
    # Enums
    TopicLabel,
    SentimentValue,
    PriorityValue,
    CustomerStatusValue,

    # Models
    KeywordInText,
    Evidence,
    TopicAssignment,
    Sentiment,
    Priority,
    CustomerStatus,
    ClassificationRequest,
    ClassificationResult,
    ClassificationMetadata,
    LLMRawOutput,
)

__all__ = [
    # Enums
    "TopicLabel",
    "SentimentValue",
    "PriorityValue",
    "CustomerStatusValue",

    # Models
    "KeywordInText",
    "Evidence",
    "TopicAssignment",
    "Sentiment",
    "Priority",
    "CustomerStatus",
    "ClassificationRequest",
    "ClassificationResult",
    "ClassificationMetadata",
    "LLMRawOutput",
]
