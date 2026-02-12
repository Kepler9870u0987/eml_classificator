"""
Pipeline version model for deterministic processing.

This module defines the PipelineVersion model that tracks all component versions
to ensure statistical determinism: same version parameters + same input = same output.
"""

from pydantic import BaseModel, Field
from typing import Literal


class PipelineVersion(BaseModel):
    """
    Immutable version contract for statistical determinism.

    Same version parameters guarantee same output for same input across all pipeline stages.
    All version fields are tracked for audit trail and backtesting.
    """

    # Core component versions
    dictionary_version: int = Field(
        description="Dynamic dictionary version for keyword/NER matching", example=1
    )
    model_version: str = Field(
        description="LLM model identifier", example="deepseek-r1-2026-01"
    )
    parser_version: str = Field(
        description="Email parser version", example="eml-parser-1.0.0"
    )
    stoplist_version: str = Field(
        description="Italian stopwords list version", example="stopwords-it-2025.1"
    )
    ner_model_version: str = Field(
        description="spaCy NER model version", example="it_core_news_lg-3.8.2"
    )
    schema_version: str = Field(
        description="JSON schema version for LLM output", example="json-schema-v2.2"
    )
    tool_calling_version: str = Field(
        description="LLM tool calling API version", example="openai-tool-calling-2026"
    )

    # Ingestion-specific versions
    canonicalization_version: str = Field(
        description="Text canonicalization algorithm version", example="canon-1.0.0"
    )
    pii_redaction_version: str = Field(
        description="PII redaction algorithm version", example="pii-redact-1.0.0"
    )
    pii_redaction_level: Literal["none", "minimal", "standard", "aggressive"] = Field(
        description="PII redaction aggressiveness level", example="standard"
    )

    model_config = {
        "frozen": True,  # Immutable
        "json_schema_extra": {
            "example": {
                "dictionary_version": 1,
                "model_version": "deepseek-r1-2026-01",
                "parser_version": "eml-parser-1.0.0",
                "stoplist_version": "stopwords-it-2025.1",
                "ner_model_version": "it_core_news_lg-3.8.2",
                "schema_version": "json-schema-v2.2",
                "tool_calling_version": "openai-tool-calling-2026",
                "canonicalization_version": "canon-1.0.0",
                "pii_redaction_version": "pii-redact-1.0.0",
                "pii_redaction_level": "standard",
            }
        },
    }

    def to_repr(self) -> str:
        """
        Short representation for logging and metrics.

        Returns:
            Compact string representation with key version components.
        """
        return (
            f"Pipeline-{self.dictionary_version}-"
            f"{self.model_version}-{self.tool_calling_version}"
        )
