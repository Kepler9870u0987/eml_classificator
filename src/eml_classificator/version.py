"""
Version constants for the email classification pipeline.

This module defines all version constants used throughout the pipeline to ensure
deterministic processing and complete audit trail.
"""

from .models.pipeline_version import PipelineVersion

# API Version
API_VERSION = "1.0.0"

# Component versions (update these when implementations change)
PARSER_VERSION = "eml-parser-1.0.0"
CANONICALIZATION_VERSION = "canon-1.0.0"
PII_REDACTION_VERSION = "pii-redact-1.0.0"

# Placeholder versions for downstream components (not yet implemented)
DICTIONARY_VERSION = 1
MODEL_VERSION = "deepseek-r1-2026-01"
STOPLIST_VERSION = "stopwords-it-2025.1"
NER_MODEL_VERSION = "it_core_news_lg-3.8.2"
SCHEMA_VERSION = "json-schema-v2.2"
TOOL_CALLING_VERSION = "openai-tool-calling-2026"


def get_current_pipeline_version(pii_redaction_level: str = "standard") -> PipelineVersion:
    """
    Get current pipeline version configuration.

    Args:
        pii_redaction_level: PII redaction level being used

    Returns:
        PipelineVersion instance with current versions
    """
    return PipelineVersion(
        dictionary_version=DICTIONARY_VERSION,
        model_version=MODEL_VERSION,
        parser_version=PARSER_VERSION,
        stoplist_version=STOPLIST_VERSION,
        ner_model_version=NER_MODEL_VERSION,
        schema_version=SCHEMA_VERSION,
        tool_calling_version=TOOL_CALLING_VERSION,
        canonicalization_version=CANONICALIZATION_VERSION,
        pii_redaction_version=PII_REDACTION_VERSION,
        pii_redaction_level=pii_redaction_level,
    )
