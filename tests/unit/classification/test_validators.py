"""
Unit tests for multi-stage validation pipeline.

Tests JSON parsing, schema validation, business rules, quality checks,
and deduplication logic.
"""

import json
import pytest
from pydantic import ValidationError

from eml_classificator.classification.validators import (
    ValidationResult,
    validate_llm_output_multistage,
    deduplicate_topics_and_keywords,
    validate_candidate_id_references,
    validate_topic_labels,
    validate_confidence_ranges,
    should_retry_on_validation_error,
    format_validation_report
)


class TestValidationResult:
    """Test ValidationResult dataclass."""

    def test_validation_result_initial_valid(self):
        """Test ValidationResult starts as valid."""
        result = ValidationResult(valid=True)

        assert result.valid is True
        assert len(result.errors) == 0
        assert len(result.warnings) == 0
        assert result.cleaned_data is None

    def test_add_error_marks_invalid(self):
        """Test adding error marks result as invalid."""
        result = ValidationResult(valid=True)
        result.add_error("Test error")

        assert result.valid is False
        assert len(result.errors) == 1
        assert result.errors[0] == "Test error"

    def test_add_warning_keeps_valid(self):
        """Test adding warning doesn't affect validity."""
        result = ValidationResult(valid=True)
        result.add_warning("Test warning")

        assert result.valid is True
        assert len(result.warnings) == 1
        assert result.warnings[0] == "Test warning"

    def test_multiple_errors(self):
        """Test adding multiple errors."""
        result = ValidationResult(valid=True)
        result.add_error("Error 1")
        result.add_error("Error 2")
        result.add_error("Error 3")

        assert len(result.errors) == 3
        assert result.valid is False


class TestMultistageValidation:
    """Test complete multi-stage validation pipeline."""

    def test_valid_output(self):
        """Test validation passes for valid LLM output."""
        valid_output = {
            "dictionary_version": 1,
            "topics": [
                {
                    "label_id": "FATTURAZIONE",
                    "confidence": 0.92,
                    "keywords_in_text": [
                        {
                            "candidate_id": "abc123",
                            "lemma": "fattura",
                            "count": 3
                        }
                    ],
                    "evidence": [
                        {"quote": "ho ricevuto la fattura"}
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
                "signals": ["fattura", "problema"]
            },
            "customer_status": {
                "value": "existing",
                "confidence": 1.0,
                "source": "llm_guess"
            }
        }

        result = validate_llm_output_multistage(
            output_json=json.dumps(valid_output),
            candidate_ids={"abc123"}
        )

        assert result.valid is True
        assert len(result.errors) == 0
        assert result.cleaned_data is not None

    def test_invalid_json(self):
        """Test validation fails for invalid JSON."""
        invalid_json = "{invalid json"

        result = validate_llm_output_multistage(
            output_json=invalid_json,
            candidate_ids=set()
        )

        assert result.valid is False
        assert len(result.errors) > 0
        assert "Invalid JSON" in result.errors[0]

    def test_schema_violation_missing_required_field(self):
        """Test schema validation catches missing required fields."""
        output_missing_sentiment = {
            "dictionary_version": 1,
            "topics": [
                {
                    "label_id": "FATTURAZIONE",
                    "confidence": 0.9,
                    "keywords_in_text": [{"candidate_id": "abc", "lemma": "x", "count": 1}],
                    "evidence": [{"quote": "test"}]
                }
            ],
            # Missing sentiment, priority, customer_status
        }

        result = validate_llm_output_multistage(
            output_json=json.dumps(output_missing_sentiment),
            candidate_ids={"abc"}
        )

        assert result.valid is False
        assert any("Schema violation" in err for err in result.errors)

    def test_invalid_candidate_id(self):
        """Test validation catches invented candidate_ids."""
        output_invalid_cid = {
            "dictionary_version": 1,
            "topics": [
                {
                    "label_id": "FATTURAZIONE",
                    "confidence": 0.9,
                    "keywords_in_text": [
                        {"candidate_id": "invented123", "lemma": "fake", "count": 1}
                    ],
                    "evidence": [{"quote": "test"}]
                }
            ],
            "sentiment": {"value": "neutral", "confidence": 0.9},
            "priority": {"value": "low", "confidence": 0.7, "signals": []},
            "customer_status": {"value": "unknown", "confidence": 0.5, "source": "llm_guess"}
        }

        result = validate_llm_output_multistage(
            output_json=json.dumps(output_invalid_cid),
            candidate_ids={"abc123", "def456"}  # invented123 not in this set
        )

        assert result.valid is False
        assert any("invented123" in err for err in result.errors)

    def test_invalid_topic_label(self):
        """Test validation catches invalid topic labels."""
        output_invalid_label = {
            "dictionary_version": 1,
            "topics": [
                {
                    "label_id": "INVALID_TOPIC",  # Not in taxonomy
                    "confidence": 0.9,
                    "keywords_in_text": [{"candidate_id": "abc", "lemma": "x", "count": 1}],
                    "evidence": [{"quote": "test"}]
                }
            ],
            "sentiment": {"value": "neutral", "confidence": 0.9},
            "priority": {"value": "low", "confidence": 0.7, "signals": []},
            "customer_status": {"value": "unknown", "confidence": 0.5, "source": "llm_guess"}
        }

        result = validate_llm_output_multistage(
            output_json=json.dumps(output_invalid_label),
            candidate_ids={"abc"}
        )

        assert result.valid is False
        assert any("label_id" in err for err in result.errors)

    def test_confidence_out_of_range(self):
        """Test validation catches confidence values outside [0, 1]."""
        output_bad_confidence = {
            "dictionary_version": 1,
            "topics": [
                {
                    "label_id": "FATTURAZIONE",
                    "confidence": 1.5,  # Out of range
                    "keywords_in_text": [{"candidate_id": "abc", "lemma": "x", "count": 1}],
                    "evidence": [{"quote": "test"}]
                }
            ],
            "sentiment": {"value": "neutral", "confidence": 0.9},
            "priority": {"value": "low", "confidence": 0.7, "signals": []},
            "customer_status": {"value": "unknown", "confidence": 0.5, "source": "llm_guess"}
        }

        result = validate_llm_output_multistage(
            output_json=json.dumps(output_bad_confidence),
            candidate_ids={"abc"}
        )

        assert result.valid is False
        assert any("Confidence out of range" in err or "confidence" in err.lower()
                   for err in result.errors)

    def test_low_confidence_warning(self):
        """Test quality check generates warning for very low confidence."""
        output_low_conf = {
            "dictionary_version": 1,
            "topics": [
                {
                    "label_id": "FATTURAZIONE",
                    "confidence": 0.1,  # Very low confidence
                    "keywords_in_text": [{"candidate_id": "abc", "lemma": "x", "count": 1}],
                    "evidence": [{"quote": "test"}]
                }
            ],
            "sentiment": {"value": "neutral", "confidence": 0.9},
            "priority": {"value": "low", "confidence": 0.7, "signals": []},
            "customer_status": {"value": "unknown", "confidence": 0.5, "source": "llm_guess"}
        }

        result = validate_llm_output_multistage(
            output_json=json.dumps(output_low_conf),
            candidate_ids={"abc"}
        )

        assert result.valid is True  # Still valid, just warning
        assert len(result.warnings) > 0
        assert any("low confidence" in warn.lower() for warn in result.warnings)

    def test_invalid_sentiment_value(self):
        """Test validation catches invalid sentiment values."""
        output_bad_sentiment = {
            "dictionary_version": 1,
            "topics": [
                {
                    "label_id": "FATTURAZIONE",
                    "confidence": 0.9,
                    "keywords_in_text": [{"candidate_id": "abc", "lemma": "x", "count": 1}],
                    "evidence": [{"quote": "test"}]
                }
            ],
            "sentiment": {"value": "invalid_sentiment", "confidence": 0.9},
            "priority": {"value": "low", "confidence": 0.7, "signals": []},
            "customer_status": {"value": "unknown", "confidence": 0.5, "source": "llm_guess"}
        }

        result = validate_llm_output_multistage(
            output_json=json.dumps(output_bad_sentiment),
            candidate_ids={"abc"}
        )

        assert result.valid is False
        assert any("sentiment" in err.lower() for err in result.errors)

    def test_invalid_priority_value(self):
        """Test validation catches invalid priority values."""
        output_bad_priority = {
            "dictionary_version": 1,
            "topics": [
                {
                    "label_id": "FATTURAZIONE",
                    "confidence": 0.9,
                    "keywords_in_text": [{"candidate_id": "abc", "lemma": "x", "count": 1}],
                    "evidence": [{"quote": "test"}]
                }
            ],
            "sentiment": {"value": "neutral", "confidence": 0.9},
            "priority": {"value": "super_urgent", "confidence": 0.7, "signals": []},
            "customer_status": {"value": "unknown", "confidence": 0.5, "source": "llm_guess"}
        }

        result = validate_llm_output_multistage(
            output_json=json.dumps(output_bad_priority),
            candidate_ids={"abc"}
        )

        assert result.valid is False
        assert any("priority" in err.lower() for err in result.errors)


class TestDeduplication:
    """Test deduplication logic."""

    def test_deduplicate_topics(self):
        """Test duplicate topics are removed."""
        data = {
            "topics": [
                {
                    "label_id": "FATTURAZIONE",
                    "confidence": 0.9,
                    "keywords_in_text": [{"candidate_id": "abc", "lemma": "x", "count": 1}]
                },
                {
                    "label_id": "FATTURAZIONE",  # Duplicate
                    "confidence": 0.8,
                    "keywords_in_text": [{"candidate_id": "def", "lemma": "y", "count": 1}]
                },
                {
                    "label_id": "RECLAMO",
                    "confidence": 0.7,
                    "keywords_in_text": [{"candidate_id": "ghi", "lemma": "z", "count": 1}]
                }
            ]
        }

        result = ValidationResult(valid=True)
        cleaned = deduplicate_topics_and_keywords(data, result)

        assert len(cleaned["topics"]) == 2  # One duplicate removed
        assert cleaned["topics"][0]["label_id"] == "FATTURAZIONE"
        assert cleaned["topics"][1]["label_id"] == "RECLAMO"
        assert len(result.warnings) > 0
        assert any("Duplicate topic" in warn for warn in result.warnings)

    def test_deduplicate_keywords(self):
        """Test duplicate keywords within topic are removed."""
        data = {
            "topics": [
                {
                    "label_id": "FATTURAZIONE",
                    "confidence": 0.9,
                    "keywords_in_text": [
                        {"candidate_id": "abc", "lemma": "fattura", "count": 3},
                        {"candidate_id": "abc", "lemma": "fattura", "count": 2},  # Duplicate
                        {"candidate_id": "def", "lemma": "pagamento", "count": 1}
                    ]
                }
            ]
        }

        result = ValidationResult(valid=True)
        cleaned = deduplicate_topics_and_keywords(data, result)

        keywords = cleaned["topics"][0]["keywords_in_text"]
        assert len(keywords) == 2  # One duplicate removed
        assert keywords[0]["candidate_id"] == "abc"
        assert keywords[1]["candidate_id"] == "def"
        assert len(result.warnings) > 0
        assert any("Duplicate keyword" in warn for warn in result.warnings)

    def test_no_duplicates(self):
        """Test deduplication with no duplicates."""
        data = {
            "topics": [
                {
                    "label_id": "FATTURAZIONE",
                    "confidence": 0.9,
                    "keywords_in_text": [
                        {"candidate_id": "abc", "lemma": "x", "count": 1},
                        {"candidate_id": "def", "lemma": "y", "count": 1}
                    ]
                },
                {
                    "label_id": "RECLAMO",
                    "confidence": 0.8,
                    "keywords_in_text": [
                        {"candidate_id": "ghi", "lemma": "z", "count": 1}
                    ]
                }
            ]
        }

        result = ValidationResult(valid=True)
        cleaned = deduplicate_topics_and_keywords(data, result)

        assert len(cleaned["topics"]) == 2
        assert len(result.warnings) == 0


class TestSpecificValidators:
    """Test specific validation helper functions."""

    def test_validate_candidate_id_references_valid(self):
        """Test candidate ID validation with valid references."""
        topics = [
            {
                "label_id": "FATTURAZIONE",
                "keywords_in_text": [
                    {"candidate_id": "abc123", "lemma": "x"},
                    {"candidate_id": "def456", "lemma": "y"}
                ]
            }
        ]
        candidate_ids = {"abc123", "def456", "ghi789"}

        errors = validate_candidate_id_references(topics, candidate_ids)
        assert len(errors) == 0

    def test_validate_candidate_id_references_invalid(self):
        """Test candidate ID validation catches invalid references."""
        topics = [
            {
                "label_id": "FATTURAZIONE",
                "keywords_in_text": [
                    {"candidate_id": "abc123", "lemma": "x"},
                    {"candidate_id": "invalid", "lemma": "y"}  # Not in candidate_ids
                ]
            }
        ]
        candidate_ids = {"abc123", "def456"}

        errors = validate_candidate_id_references(topics, candidate_ids)
        assert len(errors) > 0
        assert any("invalid" in err for err in errors)

    def test_validate_topic_labels_valid(self):
        """Test topic label validation with valid labels."""
        topics = [
            {"label_id": "FATTURAZIONE"},
            {"label_id": "RECLAMO"}
        ]
        allowed = ["FATTURAZIONE", "RECLAMO", "ASSISTENZA_TECNICA"]

        errors = validate_topic_labels(topics, allowed)
        assert len(errors) == 0

    def test_validate_topic_labels_invalid(self):
        """Test topic label validation catches invalid labels."""
        topics = [
            {"label_id": "FATTURAZIONE"},
            {"label_id": "INVALID_TOPIC"}
        ]
        allowed = ["FATTURAZIONE", "RECLAMO"]

        errors = validate_topic_labels(topics, allowed)
        assert len(errors) > 0
        assert any("INVALID_TOPIC" in err for err in errors)

    def test_validate_confidence_ranges_valid(self):
        """Test confidence range validation with valid values."""
        data = {
            "topics": [
                {"label_id": "FATTURAZIONE", "confidence": 0.92}
            ],
            "sentiment": {"confidence": 0.85},
            "priority": {"confidence": 0.88},
            "customer_status": {"confidence": 1.0, "source": "llm_guess"}
        }

        errors = validate_confidence_ranges(data)
        assert len(errors) == 0

    def test_validate_confidence_ranges_invalid(self):
        """Test confidence range validation catches out-of-range values."""
        data = {
            "topics": [
                {"label_id": "FATTURAZIONE", "confidence": 1.5}  # Out of range
            ],
            "sentiment": {"confidence": -0.1},  # Out of range
            "priority": {"confidence": 0.88},
            "customer_status": {"confidence": 1.0, "source": "llm_guess"}
        }

        errors = validate_confidence_ranges(data)
        assert len(errors) >= 2  # At least 2 errors


class TestRetryHelper:
    """Test retry decision logic."""

    def test_should_not_retry_valid_result(self):
        """Test no retry needed for valid result."""
        result = ValidationResult(valid=True)

        assert should_retry_on_validation_error(result) is False

    def test_should_not_retry_systematic_json_error(self):
        """Test systematic JSON errors shouldn't be retried."""
        result = ValidationResult(valid=False)
        result.add_error("Invalid JSON: Expecting value")

        assert should_retry_on_validation_error(result) is False

    def test_should_not_retry_systematic_schema_error(self):
        """Test systematic schema errors shouldn't be retried."""
        result = ValidationResult(valid=False)
        result.add_error("Schema violation at topics: field required")

        assert should_retry_on_validation_error(result) is False

    def test_should_retry_transient_errors(self):
        """Test transient errors (candidate_id, label issues) should be retried."""
        result = ValidationResult(valid=False)
        result.add_error("Invalid candidate_id in topic 0: xyz123")

        assert should_retry_on_validation_error(result) is True

    def test_should_retry_label_errors(self):
        """Test label validation errors should be retried."""
        result = ValidationResult(valid=False)
        result.add_error("Invalid label_id in topic 0: WRONG_LABEL")

        assert should_retry_on_validation_error(result) is True


class TestValidationReport:
    """Test validation report formatting."""

    def test_format_report_valid(self):
        """Test formatting report for valid result."""
        result = ValidationResult(valid=True)
        result.cleaned_data = {"topics": [{"label_id": "FATTURAZIONE"}]}

        report = format_validation_report(result)

        assert "✓ VALID" in report
        assert "Topics: 1" in report

    def test_format_report_with_errors(self):
        """Test formatting report with errors."""
        result = ValidationResult(valid=False)
        result.add_error("Error 1")
        result.add_error("Error 2")

        report = format_validation_report(result)

        assert "✗ INVALID" in report
        assert "Errors (2)" in report
        assert "Error 1" in report
        assert "Error 2" in report

    def test_format_report_with_warnings(self):
        """Test formatting report with warnings."""
        result = ValidationResult(valid=True)
        result.add_warning("Warning 1")
        result.add_warning("Warning 2")
        result.cleaned_data = {"topics": []}

        report = format_validation_report(result)

        assert "✓ VALID" in report
        assert "Warnings (2)" in report
        assert "Warning 1" in report
        assert "Warning 2" in report

    def test_format_report_errors_and_warnings(self):
        """Test formatting report with both errors and warnings."""
        result = ValidationResult(valid=False)
        result.add_error("Critical error")
        result.add_warning("Minor warning")

        report = format_validation_report(result)

        assert "✗ INVALID" in report
        assert "Errors (1)" in report
        assert "Warnings (1)" in report
        assert "Critical error" in report
        assert "Minor warning" in report
