"""
Main email classifier orchestrating the complete Phase 3 pipeline.

Coordinates:
1. Prompt building from EmailDocument + candidates
2. LLM call with tool calling
3. Multi-stage validation (with retries)
4. Post-processing (customer_status, priority, confidence)
5. Result assembly

This is the main entry point for email classification.
"""

import time
from typing import Dict, Any, List, Optional
from datetime import datetime

import structlog

from eml_classificator.classification.schemas import (
    ClassificationResult,
    ClassificationMetadata,
    TopicAssignment,
    Sentiment,
    Priority,
    CustomerStatus
)
from eml_classificator.classification.llm_client import (
    LLMClient,
    create_llm_client_from_model_string,
    LLMResponse
)
from eml_classificator.classification.prompts import (
    EmailToClassify,
    build_classification_prompt
)
from eml_classificator.classification.validators import (
    validate_llm_output_multistage,
    ValidationResult,
    should_retry_on_validation_error
)
from eml_classificator.classification.customer_status import (
    override_customer_status_with_crm
)
from eml_classificator.classification.priority_scorer import (
    override_priority_with_rules
)
from eml_classificator.classification.confidence import (
    adjust_all_topic_confidences
)
from eml_classificator.version import get_current_pipeline_version
from eml_classificator.config import settings


logger = structlog.get_logger(__name__)


# ============================================================================
# EMAIL CLASSIFIER
# ============================================================================

class EmailClassifier:
    """
    Main email classifier for Phase 3.

    Orchestrates LLM classification with validation and post-processing.
    """

    def __init__(
        self,
        llm_client: Optional[LLMClient] = None,
        model_override: Optional[str] = None
    ):
        """
        Initialize classifier.

        Args:
            llm_client: Optional pre-configured LLM client
            model_override: Optional model string (e.g., "ollama/deepseek-r1:7b")
        """
        if llm_client is None:
            if model_override:
                llm_client = create_llm_client_from_model_string(model_override)
            else:
                # Use default from config
                llm_client = create_llm_client_from_model_string(
                    f"{settings.llm_provider}/{settings.llm_model}"
                )

        self.llm_client = llm_client
        self.logger = logger.bind(
            classifier="EmailClassifier",
            model=llm_client.model
        )

    def classify(
        self,
        email_document: Dict[str, Any],
        candidates: List[Dict[str, Any]],
        dictionary_version: int,
        priority_weights_override: Optional[Dict[str, float]] = None
    ) -> ClassificationResult:
        """
        Classify an email using the complete pipeline.

        Args:
            email_document: EmailDocument dict from Phase 1
            candidates: Candidate keywords list from Phase 2
            dictionary_version: Dictionary version for audit
            priority_weights_override: Optional priority weight overrides

        Returns:
            ClassificationResult with topics, sentiment, priority, customer_status

        Raises:
            Exception: On unrecoverable errors
        """
        start_time = time.time()
        retry_count = 0

        self.logger.info(
            "classification_started",
            message_id=email_document.get("message_id"),
            candidates_count=len(candidates)
        )

        # Prepare email for classification
        email = EmailToClassify(
            message_id=email_document.get("message_id", ""),
            subject=email_document.get("subject", ""),
            from_address=email_document.get("from_raw", ""),
            body_canonical=email_document.get("body_canonical", ""),
            dictionary_version=dictionary_version
        )

        # Build prompts
        system_prompt, user_prompt = build_classification_prompt(
            email=email,
            candidates=candidates
        )

        # Prepare candidate ID set for validation
        candidate_ids = {c["candidate_id"] for c in candidates}

        # Call LLM with validation retry loop
        llm_response = None
        validation_result = None

        for attempt in range(settings.llm_max_retries):
            try:
                # Call LLM
                llm_start = time.time()
                llm_response = self.llm_client.classify(
                    system_prompt=system_prompt,
                    user_prompt=user_prompt
                )
                llm_latency_ms = int((time.time() - llm_start) * 1000)

                self.logger.debug(
                    "llm_call_completed",
                    attempt=attempt + 1,
                    latency_ms=llm_latency_ms,
                    tokens_total=llm_response.tokens_total
                )

                # Validate output
                import json
                output_json = json.dumps(llm_response.tool_result)

                validation_result = validate_llm_output_multistage(
                    output_json=output_json,
                    candidate_ids=candidate_ids
                )

                if validation_result.valid:
                    # Success!
                    break
                else:
                    # Validation failed
                    self.logger.warning(
                        "llm_output_validation_failed",
                        attempt=attempt + 1,
                        errors=validation_result.errors,
                        warnings=validation_result.warnings
                    )

                    # Check if retry is worth it
                    if not should_retry_on_validation_error(validation_result):
                        raise ValueError(
                            f"Systematic validation error: {validation_result.errors}"
                        )

                    retry_count += 1

                    if attempt == settings.llm_max_retries - 1:
                        raise ValueError(
                            f"Validation failed after {settings.llm_max_retries} attempts: "
                            f"{validation_result.errors}"
                        )

            except Exception as e:
                self.logger.error(
                    "classification_attempt_failed",
                    attempt=attempt + 1,
                    error=str(e)
                )

                if attempt == settings.llm_max_retries - 1:
                    raise

                time.sleep(settings.llm_retry_delay_seconds * (2 ** attempt))

        if not validation_result or not validation_result.valid:
            raise ValueError("Classification validation failed")

        # Extract validated data
        validated_data = validation_result.cleaned_data

        # Parse into schema models
        topics_raw = [
            TopicAssignment(**t) for t in validated_data["topics"]
        ]
        sentiment_raw = Sentiment(**validated_data["sentiment"])
        priority_raw = Priority(**validated_data["priority"])
        customer_status_raw = CustomerStatus(**validated_data["customer_status"])

        # Post-processing: Override customer_status with CRM
        customer_status_final = override_customer_status_with_crm(
            llm_customer_status=customer_status_raw,
            from_email=email.from_address,
            text_body=email.body_canonical
        )

        # Post-processing: Override priority with rules
        priority_final = override_priority_with_rules(
            llm_priority=priority_raw,
            subject=email.subject,
            body_canonical=email.body_canonical,
            sentiment_value=sentiment_raw.value,
            customer_value=customer_status_final.value,
            vip_status=False  # TODO: Get from CRM
        )

        # Post-processing: Adjust topic confidences
        topics_adjusted = adjust_all_topic_confidences(
            topics=topics_raw,
            candidates=candidates
        )

        # Calculate total latency
        total_latency_ms = int((time.time() - start_time) * 1000)

        # Build metadata
        pipeline_version = get_current_pipeline_version()

        metadata = ClassificationMetadata(
            pipeline_version=pipeline_version.model_dump(),
            model_used=f"{llm_response.provider}/{llm_response.model}",
            tokens_input=llm_response.tokens_input,
            tokens_output=llm_response.tokens_output,
            tokens_total=llm_response.tokens_total,
            latency_ms=total_latency_ms,
            llm_latency_ms=llm_response.latency_ms,
            validation_warnings=validation_result.warnings,
            retry_count=retry_count,
            classified_at=datetime.utcnow()
        )

        # Assemble final result
        result = ClassificationResult(
            dictionary_version=dictionary_version,
            topics=topics_adjusted,
            sentiment=sentiment_raw,
            priority=priority_final,
            customer_status=customer_status_final,
            metadata=metadata
        )

        self.logger.info(
            "classification_completed",
            message_id=email.message_id,
            topics_count=len(result.topics),
            latency_ms=total_latency_ms,
            retry_count=retry_count
        )

        return result


# ============================================================================
# CONVENIENCE FUNCTIONS
# ============================================================================

def classify_email(
    email_document: Dict[str, Any],
    candidates: List[Dict[str, Any]],
    dictionary_version: int,
    model_override: Optional[str] = None,
    priority_weights_override: Optional[Dict[str, float]] = None
) -> ClassificationResult:
    """
    Classify an email using default or custom configuration.

    This is the main entry point for classification.

    Args:
        email_document: EmailDocument dict from Phase 1
        candidates: Candidate keywords from Phase 2
        dictionary_version: Dictionary version
        model_override: Optional model override (e.g., "ollama/qwen2.5:32b")
        priority_weights_override: Optional priority weights

    Returns:
        ClassificationResult

    Example:
        >>> result = classify_email(
        ...     email_document=doc,
        ...     candidates=candidates,
        ...     dictionary_version=1,
        ...     model_override="ollama/deepseek-r1:7b"
        ... )
        >>> print(result.topics[0].label_id)
        'FATTURAZIONE'
    """
    classifier = EmailClassifier(model_override=model_override)

    return classifier.classify(
        email_document=email_document,
        candidates=candidates,
        dictionary_version=dictionary_version,
        priority_weights_override=priority_weights_override
    )


# ============================================================================
# BATCH CLASSIFICATION
# ============================================================================

def classify_batch(
    emails_with_candidates: List[tuple[Dict[str, Any], List[Dict[str, Any]]]],
    dictionary_version: int,
    model_override: Optional[str] = None
) -> List[ClassificationResult]:
    """
    Classify multiple emails in batch.

    Args:
        emails_with_candidates: List of (email_document, candidates) tuples
        dictionary_version: Dictionary version
        model_override: Optional model override

    Returns:
        List of ClassificationResults
    """
    classifier = EmailClassifier(model_override=model_override)

    results = []
    for email_doc, candidates in emails_with_candidates:
        try:
            result = classifier.classify(
                email_document=email_doc,
                candidates=candidates,
                dictionary_version=dictionary_version
            )
            results.append(result)
        except Exception as e:
            logger.error(
                "batch_classification_item_failed",
                message_id=email_doc.get("message_id"),
                error=str(e)
            )
            # Continue with next email
            continue

    logger.info(
        "batch_classification_completed",
        total_count=len(emails_with_candidates),
        success_count=len(results),
        failure_count=len(emails_with_candidates) - len(results)
    )

    return results
