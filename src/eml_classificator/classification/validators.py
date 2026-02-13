"""
Multi-stage validation pipeline for LLM classification outputs.

Implements validation strategy from brainstorming v2 section 3.5:
1. JSON Parse - validate parseable JSON
2. Schema Validation - Pydantic model validation
3. Business Rules - candidate_id exists, label_id in enum, etc.
4. Quality Checks - warnings for low confidence, missing data, etc.

Provides detailed error reporting and deduplication of topics/keywords.
"""

import json
from typing import Dict, Any, List, Set, Optional
from dataclasses import dataclass, field

import structlog
from pydantic import ValidationError

from eml_classificator.classification.schemas import (
    LLMRawOutput,
    TopicLabel,
    SentimentValue,
    PriorityValue,
    CustomerStatusValue
)
from eml_classificator.classification.tool_definitions import (
    get_topics_enum,
    get_sentiment_enum,
    get_priority_enum,
    get_customer_status_enum
)


logger = structlog.get_logger(__name__)


# ============================================================================
# VALIDATION RESULT
# ============================================================================

@dataclass
class ValidationResult:
    """
    Result of multi-stage validation.

    Contains validation status, errors, warnings, and cleaned data.
    """
    valid: bool
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    cleaned_data: Optional[Dict[str, Any]] = None

    def add_error(self, error: str):
        """Add an error to the result."""
        self.errors.append(error)
        self.valid = False

    def add_warning(self, warning: str):
        """Add a warning to the result (non-fatal)."""
        self.warnings.append(warning)


# ============================================================================
# VALIDATION PIPELINE
# ============================================================================

def validate_llm_output_multistage(
    output_json: str,
    candidate_ids: Set[str],
    allowed_topics: Optional[List[str]] = None
) -> ValidationResult:
    """
    Multi-stage validation of LLM output.

    Args:
        output_json: JSON string from LLM
        candidate_ids: Set of valid candidate IDs from input
        allowed_topics: List of allowed topic labels (default: from tool_definitions)

    Returns:
        ValidationResult with status, errors, warnings, cleaned data
    """
    result = ValidationResult(valid=True)

    if allowed_topics is None:
        allowed_topics = get_topics_enum()

    # Stage 1: Parse JSON
    logger.debug("validation_stage_1_json_parse")
    try:
        data = json.loads(output_json)
    except json.JSONDecodeError as e:
        result.add_error(f"Invalid JSON: {e}")
        return result

    # Stage 2: Schema validation with Pydantic
    logger.debug("validation_stage_2_schema")
    try:
        llm_output = LLMRawOutput(**data)
        # Convert back to dict for further processing
        data = llm_output.model_dump()
    except ValidationError as e:
        for error in e.errors():
            field_path = ".".join(str(loc) for loc in error["loc"])
            result.add_error(f"Schema violation at {field_path}: {error['msg']}")
        return result

    # Stage 3: Business rules validation
    logger.debug("validation_stage_3_business_rules")

    # 3.1: Check each topic
    for topic_idx, topic in enumerate(data.get("topics", [])):
        # Check label_id in enum
        label_id = topic.get("label_id")
        if label_id not in allowed_topics:
            result.add_error(
                f"Invalid label_id in topic {topic_idx}: {label_id}. "
                f"Allowed: {allowed_topics}"
            )

        # Check candidate_ids exist
        for kw_idx, kw in enumerate(topic.get("keywords_in_text", [])):
            cid = kw.get("candidate_id")
            if cid not in candidate_ids:
                result.add_error(
                    f"Invalid candidate_id in topic {topic_idx}, keyword {kw_idx}: {cid}. "
                    f"Not in input candidate list (invented by LLM)"
                )

        # Check confidence range
        confidence = topic.get("confidence")
        if confidence is not None and not (0.0 <= confidence <= 1.0):
            result.add_error(
                f"Confidence out of range in topic {topic_idx}: {confidence}. "
                f"Must be in [0.0, 1.0]"
            )

    # 3.2: Check sentiment
    sentiment = data.get("sentiment", {})
    if sentiment.get("value") not in get_sentiment_enum():
        result.add_error(f"Invalid sentiment value: {sent

iment.get('value')}")

    # 3.3: Check priority
    priority = data.get("priority", {})
    if priority.get("value") not in get_priority_enum():
        result.add_error(f"Invalid priority value: {priority.get('value')}")

    # 3.4: Check customer_status
    customer_status = data.get("customer_status", {})
    if customer_status.get("value") not in get_customer_status_enum():
        result.add_error(f"Invalid customer_status value: {customer_status.get('value')}")

    # Stage 4: Quality checks (warnings)
    logger.debug("validation_stage_4_quality_checks")

    for topic_idx, topic in enumerate(data.get("topics", [])):
        # Low confidence warning
        confidence = topic.get("confidence", 0.0)
        if confidence < 0.2:
            result.add_warning(
                f"Very low confidence for topic {topic_idx} ({topic.get('label_id')}): {confidence}"
            )

        # No keywords warning
        keywords = topic.get("keywords_in_text", [])
        if len(keywords) == 0:
            result.add_warning(
                f"No keywords for topic {topic_idx} ({topic.get('label_id')})"
            )

        # No evidence warning
        evidence = topic.get("evidence", [])
        if len(evidence) == 0:
            result.add_warning(
                f"No evidence for topic {topic_idx} ({topic.get('label_id')})"
            )

    # Stage 5: Deduplication
    logger.debug("validation_stage_5_deduplication")
    data = deduplicate_topics_and_keywords(data, result)

    # Set cleaned data if valid
    if result.valid:
        result.cleaned_data = data

    logger.info(
        "llm_output_validation_complete",
        valid=result.valid,
        errors_count=len(result.errors),
        warnings_count=len(result.warnings)
    )

    return result


# ============================================================================
# DEDUPLICATION
# ============================================================================

def deduplicate_topics_and_keywords(
    data: Dict[str, Any],
    result: ValidationResult
) -> Dict[str, Any]:
    """
    Remove duplicate topics and keywords.

    Keeps first occurrence of each unique label_id and candidate_id.

    Args:
        data: LLM output data
        result: ValidationResult to add warnings to

    Returns:
        Cleaned data with duplicates removed
    """
    # Deduplicate topics
    seen_labels = set()
    unique_topics = []

    for topic in data.get("topics", []):
        label_id = topic.get("label_id")
        if label_id not in seen_labels:
            unique_topics.append(topic)
            seen_labels.add(label_id)
        else:
            result.add_warning(f"Duplicate topic removed: {label_id}")

    data["topics"] = unique_topics

    # Deduplicate keywords within each topic
    for topic in data["topics"]:
        seen_candidate_ids = set()
        unique_keywords = []

        for kw in topic.get("keywords_in_text", []):
            cid = kw.get("candidate_id")
            if cid not in seen_candidate_ids:
                unique_keywords.append(kw)
                seen_candidate_ids.add(cid)
            else:
                result.add_warning(
                    f"Duplicate keyword removed in topic {topic['label_id']}: {cid}"
                )

        topic["keywords_in_text"] = unique_keywords

    return data


# ============================================================================
# SPECIFIC VALIDATORS
# ============================================================================

def validate_candidate_id_references(
    topics: List[Dict[str, Any]],
    candidate_ids: Set[str]
) -> List[str]:
    """
    Validate that all candidate_ids in topics exist in the input candidate list.

    Args:
        topics: List of topic assignments
        candidate_ids: Set of valid candidate IDs

    Returns:
        List of error messages (empty if all valid)
    """
    errors = []

    for topic_idx, topic in enumerate(topics):
        for kw_idx, kw in enumerate(topic.get("keywords_in_text", [])):
            cid = kw.get("candidate_id")
            if cid not in candidate_ids:
                errors.append(
                    f"Invalid candidate_id in topic {topic_idx} "
                    f"({topic.get('label_id')}), keyword {kw_idx}: {cid}"
                )

    return errors


def validate_topic_labels(
    topics: List[Dict[str, Any]],
    allowed_topics: List[str]
) -> List[str]:
    """
    Validate that all topic labels are in the allowed taxonomy.

    Args:
        topics: List of topic assignments
        allowed_topics: List of allowed label IDs

    Returns:
        List of error messages (empty if all valid)
    """
    errors = []

    for topic_idx, topic in enumerate(topics):
        label_id = topic.get("label_id")
        if label_id not in allowed_topics:
            errors.append(
                f"Invalid label_id in topic {topic_idx}: {label_id}. "
                f"Allowed: {allowed_topics}"
            )

    return errors


def validate_confidence_ranges(data: Dict[str, Any]) -> List[str]:
    """
    Validate that all confidence scores are in [0.0, 1.0].

    Args:
        data: LLM output data

    Returns:
        List of error messages (empty if all valid)
    """
    errors = []

    # Topic confidences
    for topic_idx, topic in enumerate(data.get("topics", [])):
        conf = topic.get("confidence")
        if conf is not None and not (0.0 <= conf <= 1.0):
            errors.append(
                f"Topic {topic_idx} confidence out of range: {conf}"
            )

    # Sentiment confidence
    sentiment_conf = data.get("sentiment", {}).get("confidence")
    if sentiment_conf is not None and not (0.0 <= sentiment_conf <= 1.0):
        errors.append(f"Sentiment confidence out of range: {sentiment_conf}")

    # Priority confidence
    priority_conf = data.get("priority", {}).get("confidence")
    if priority_conf is not None and not (0.0 <= priority_conf <= 1.0):
        errors.append(f"Priority confidence out of range: {priority_conf}")

    # Customer status confidence
    status_conf = data.get("customer_status", {}).get("confidence")
    if status_conf is not None and not (0.0 <= status_conf <= 1.0):
        errors.append(f"Customer status confidence out of range: {status_conf}")

    return errors


# ============================================================================
# RETRY HELPER
# ============================================================================

def should_retry_on_validation_error(result: ValidationResult) -> bool:
    """
    Determine if validation error is worth retrying the LLM call.

    Some errors are transient (LLM hallucinated, formatting issue),
    others are systematic (model doesn't support tool calling).

    Args:
        result: Validation result

    Returns:
        True if worth retrying, False if error is likely systematic
    """
    # If no errors, no need to retry
    if result.valid:
        return False

    # Check for systematic errors that won't be fixed by retry
    systematic_patterns = [
        "Schema violation",  # Model doesn't understand schema
        "Invalid JSON",  # Model not returning JSON at all
    ]

    for error in result.errors:
        for pattern in systematic_patterns:
            if pattern in error:
                logger.warning(
                    "systematic_validation_error_detected",
                    error=error,
                    pattern=pattern
                )
                return False

    # For other errors (candidate_id issues, label issues), retry is worth it
    return True


# ============================================================================
# VALIDATION REPORT
# ============================================================================

def format_validation_report(result: ValidationResult) -> str:
    """
    Format validation result as human-readable report.

    Args:
        result: Validation result

    Returns:
        Formatted string report
    """
    report = "=== LLM Output Validation Report ===\n\n"

    report += f"Status: {'✓ VALID' if result.valid else '✗ INVALID'}\n\n"

    if result.errors:
        report += f"Errors ({len(result.errors)}):\n"
        for i, error in enumerate(result.errors, 1):
            report += f"  {i}. {error}\n"
        report += "\n"

    if result.warnings:
        report += f"Warnings ({len(result.warnings)}):\n"
        for i, warning in enumerate(result.warnings, 1):
            report += f"  {i}. {warning}\n"
        report += "\n"

    if result.valid:
        report += "✓ Output passed all validation stages\n"
        report += f"✓ Topics: {len(result.cleaned_data.get('topics', []))}\n"

    return report
