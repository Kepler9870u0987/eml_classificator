"""
Unit tests for main email classifier orchestrator.

Tests the complete classification pipeline including prompt building,
LLM call, validation, and post-processing.
"""

import json
import pytest
from unittest.mock import Mock, MagicMock, patch
from datetime import datetime

from eml_classificator.classification.classifier import (
    EmailClassifier,
    classify_email,
    classify_batch
)
from eml_classificator.classification.llm_client import LLMResponse
from eml_classificator.classification.schemas import (
    TopicLabel,
    SentimentValue,
    PriorityValue,
    CustomerStatusValue,
    ClassificationResult,
    TopicAssignment,
    KeywordInText,
    Evidence,
    Priority,
    Sentiment,
    CustomerStatus
)


# Mock LLM response for testing
MOCK_VALID_LLM_OUTPUT = {
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
                {"quote": "ho ricevuto la fattura errata"}
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


class TestEmailClassifier:
    """Test EmailClassifier class."""

    @patch('eml_classificator.classification.classifier.LLMClient')
    def test_classifier_initialization(self, mock_llm_client_class):
        """Test EmailClassifier initialization."""
        mock_client = Mock()
        mock_llm_client_class.return_value = mock_client

        classifier = EmailClassifier(llm_client=mock_client)

        assert classifier.llm_client == mock_client

    @patch('eml_classificator.classification.classifier.create_llm_client_from_model_string')
    def test_classifier_initialization_with_model_override(self, mock_create_client):
        """Test EmailClassifier with model override."""
        mock_client = Mock()
        mock_create_client.return_value = mock_client

        classifier = EmailClassifier(model_override="ollama/deepseek-r1:7b")

        mock_create_client.assert_called_once_with("ollama/deepseek-r1:7b")
        assert classifier.llm_client == mock_client

    @patch('eml_classificator.classification.classifier.create_llm_client_from_model_string')
    @patch('eml_classificator.classification.classifier.settings')
    def test_classifier_initialization_default_model(self, mock_settings, mock_create_client):
        """Test EmailClassifier uses default model from settings."""
        mock_settings.llm_provider = "ollama"
        mock_settings.llm_model = "qwen2.5:32b"
        mock_client = Mock()
        mock_create_client.return_value = mock_client

        classifier = EmailClassifier()

        mock_create_client.assert_called_once_with("ollama/qwen2.5:32b")

    @patch('eml_classificator.classification.classifier.EmailClassifier.__init__')
    def test_classify_success(self, mock_init):
        """Test successful classification flow."""
        mock_init.return_value = None

        # Create classifier with mocks
        classifier = EmailClassifier.__new__(EmailClassifier)
        classifier.logger = Mock()

        # Mock LLM client
        mock_llm_client = Mock()
        mock_llm_client.model = "test-model"
        mock_llm_client.classify.return_value = LLMResponse(
            tool_result=MOCK_VALID_LLM_OUTPUT,
            model="test-model",
            provider="test",
            tokens_input=150,
            tokens_output=100,
            tokens_total=250,
            latency_ms=2000
        )
        classifier.llm_client = mock_llm_client

        # Mock email document
        email_document = {
            "message_id": "test@123",
            "subject": "Problema con fattura",
            "from_raw": "john@company.com",
            "body_canonical": "Ho ricevuto la fattura errata."
        }

        # Mock candidates
        candidates = [
            {
                 "candidate_id": "abc123",
                "lemma": "fattura",
                "composite_score": 0.95
            }
        ]

        with patch('eml_classificator.classification.classifier.build_classification_prompt') as mock_build_prompt, \
             patch('eml_classificator.classification.classifier.validate_llm_output_multistage') as mock_validate, \
             patch('eml_classificator.classification.classifier.override_customer_status_with_crm') as mock_override_status, \
             patch('eml_classificator.classification.classifier.override_priority_with_rules') as mock_override_priority, \
             patch('eml_classificator.classification.classifier.adjust_all_topic_confidences') as mock_adjust_conf, \
             patch('eml_classificator.classification.classifier.get_current_pipeline_version') as mock_version:

            # Setup return values
            mock_build_prompt.return_value = ("system prompt", "user prompt")

            mock_validation_result = Mock()
            mock_validation_result.valid = True
            mock_validation_result.warnings = []
            mock_validation_result.cleaned_data = MOCK_VALID_LLM_OUTPUT
            mock_validate.return_value = mock_validation_result

            mock_override_status.return_value = CustomerStatus(
                value=CustomerStatusValue.EXISTING,
                confidence=1.0,
                source="crm_exact_match"
            )

            mock_override_priority.return_value = Priority(
                value=PriorityValue.HIGH,
                confidence=0.9,
                signals=["test"],
                raw_score=5.0
            )

            mock_adjust_conf.return_value = [
                TopicAssignment(
                    label_id=TopicLabel.FATTURAZIONE,
                    confidence=0.92,
                    keywords_in_text=[KeywordInText(candidate_id="abc123", lemma="fattura", count=1)],
                    evidence=[Evidence(quote="test evidence")]
                )
            ]

            mock_version.return_value = Mock(model_dump=lambda: {"phase": 3})

            # Call classify
            result = classifier.classify(
                email_document=email_document,
                candidates=candidates,
                dictionary_version=1
            )

            # Verify result
            assert isinstance(result, ClassificationResult)
            assert result.dictionary_version == 1
            assert len(result.topics) > 0

    @patch('eml_classificator.classification.classifier.EmailClassifier.__init__')
    @patch('eml_classificator.classification.classifier.settings')
    def test_classify_with_validation_retry(self, mock_settings, mock_init):
        """Test classification retries on validation failure."""
        mock_settings.llm_max_retries = 3
        mock_settings.llm_retry_delay_seconds = 0.01
        mock_init.return_value = None

        # Create classifier with mocks
        classifier = EmailClassifier.__new__(EmailClassifier)
        classifier.logger = Mock()

        # Mock LLM client that fails first time, succeeds second time
        mock_llm_client = Mock()
        mock_llm_client.model = "test-model"

        # First call returns invalid, second call returns valid
        invalid_output = {"invalid": "data"}
        valid_output = MOCK_VALID_LLM_OUTPUT

        mock_llm_client.classify.side_effect = [
            LLMResponse(
                tool_result=invalid_output,
                model="test-model",
                provider="test",
                tokens_total=100,
                latency_ms=1000
            ),
            LLMResponse(
                tool_result=valid_output,
                model="test-model",
                provider="test",
                tokens_total=150,
                latency_ms=1500
            )
        ]
        classifier.llm_client = mock_llm_client

        email_document = {
            "message_id": "test@123",
            "subject": "Test",
            "from_raw": "test@example.com",
            "body_canonical": "Test body"
        }

        candidates = [{"candidate_id": "abc", "lemma": "test", "composite_score": 0.9}]

        with patch('eml_classificator.classification.classifier.build_classification_prompt') as mock_build_prompt, \
             patch('eml_classificator.classification.classifier.validate_llm_output_multistage') as mock_validate, \
             patch('eml_classificator.classification.classifier.should_retry_on_validation_error') as mock_should_retry, \
             patch('eml_classificator.classification.classifier.override_customer_status_with_crm') as mock_override_status, \
             patch('eml_classificator.classification.classifier.override_priority_with_rules') as mock_override_priority, \
             patch('eml_classificator.classification.classifier.adjust_all_topic_confidences') as mock_adjust_conf, \
             patch('eml_classificator.classification.classifier.get_current_pipeline_version') as mock_version:

            mock_build_prompt.return_value = ("system", "user")

            # First validation fails, second succeeds
            invalid_result = Mock(valid=False, errors=["test error"], warnings=[])
            valid_result = Mock(valid=True, warnings=[], cleaned_data=valid_output)
            mock_validate.side_effect = [invalid_result, valid_result]

            mock_should_retry.return_value = True

            mock_override_status.return_value = CustomerStatus(
                value=CustomerStatusValue.NEW,
                confidence=0.8,
                source="default"
            )

            mock_override_priority.return_value = Priority(
                value=PriorityValue.LOW,
                confidence=0.7,
                signals=[],
                raw_score=1.0
            )

            mock_adjust_conf.return_value = [
                TopicAssignment(
                    label_id=TopicLabel.UNKNOWN_TOPIC,
                    confidence=0.5,
                    keywords_in_text=[KeywordInText(candidate_id="abc", lemma="test", count=1)],
                    evidence=[Evidence(quote="test")]
                )
            ]

            mock_version.return_value = Mock(model_dump=lambda: {})

            # Should succeed after retry
            result = classifier.classify(
                email_document=email_document,
                candidates=candidates,
                dictionary_version=1
            )

            assert isinstance(result, ClassificationResult)
            # Should have called LLM twice
            assert mock_llm_client.classify.call_count == 2

    @patch('eml_classificator.classification.classifier.EmailClassifier.__init__')
    @patch('eml_classificator.classification.classifier.settings')
    def test_classify_max_retries_exceeded(self, mock_settings, mock_init):
        """Test classification fails after max retries."""
        mock_settings.llm_max_retries = 2
        mock_settings.llm_retry_delay_seconds = 0.01
        mock_init.return_value = None

        classifier = EmailClassifier.__new__(EmailClassifier)
        classifier.logger = Mock()

        # Mock LLM client that always returns invalid
        mock_llm_client = Mock()
        mock_llm_client.model = "test-model"
        mock_llm_client.classify.return_value = LLMResponse(
            tool_result={"invalid": "data"},
            model="test-model",
            provider="test",
            tokens_total=100,
            latency_ms=1000
        )
        classifier.llm_client = mock_llm_client

        email_document = {
            "message_id": "test@123",
            "subject": "Test",
            "from_raw": "test@example.com",
            "body_canonical": "Test"
        }
        candidates = [{"candidate_id": "abc", "lemma": "test", "composite_score": 0.9}]

        with patch('eml_classificator.classification.classifier.build_classification_prompt') as mock_build_prompt, \
             patch('eml_classificator.classification.classifier.validate_llm_output_multistage') as mock_validate, \
             patch('eml_classificator.classification.classifier.should_retry_on_validation_error') as mock_should_retry:

            mock_build_prompt.return_value = ("system", "user")

            invalid_result = Mock(valid=False, errors=["persistent error"], warnings=[])
            mock_validate.return_value = invalid_result

            mock_should_retry.return_value = True

            # Should raise ValueError after max retries
            with pytest.raises(ValueError, match="Validation failed after"):
                classifier.classify(
                    email_document=email_document,
                    candidates=candidates,
                    dictionary_version=1
                )


class TestClassifyEmailConvenienceFunction:
    """Test classify_email convenience function."""

    @patch('eml_classificator.classification.classifier.EmailClassifier')
    def test_classify_email_default(self, mock_classifier_class):
        """Test classify_email creates classifier and calls classify."""
        mock_classifier = Mock()
        mock_classifier.classify.return_value = Mock(spec=ClassificationResult)
        mock_classifier_class.return_value = mock_classifier

        email_document = {"message_id": "test", "subject": "Test"}
        candidates = [{"candidate_id": "abc", "lemma": "test"}]

        result = classify_email(
            email_document=email_document,
            candidates=candidates,
            dictionary_version=1
        )

        mock_classifier_class.assert_called_once_with(model_override=None)
        mock_classifier.classify.assert_called_once()

    @patch('eml_classificator.classification.classifier.EmailClassifier')
    def test_classify_email_with_model_override(self, mock_classifier_class):
        """Test classify_email with model override."""
        mock_classifier = Mock()
        mock_classifier.classify.return_value = Mock(spec=ClassificationResult)
        mock_classifier_class.return_value = mock_classifier

        email_document = {"message_id": "test"}
        candidates = []

        result = classify_email(
            email_document=email_document,
            candidates=candidates,
            dictionary_version=1,
            model_override="ollama/qwen2.5:32b"
        )

        mock_classifier_class.assert_called_once_with(model_override="ollama/qwen2.5:32b")


class TestClassifyBatch:
    """Test batch classification functionality."""

    @patch('eml_classificator.classification.classifier.EmailClassifier')
    def test_classify_batch_success(self, mock_classifier_class):
        """Test successful batch classification."""
        mock_classifier = Mock()
        mock_results = [
            Mock(spec=ClassificationResult),
            Mock(spec=ClassificationResult)
        ]
        mock_classifier.classify.side_effect = mock_results
        mock_classifier_class.return_value = mock_classifier

        emails_with_candidates = [
            ({"message_id": "1"}, [{"candidate_id": "abc"}]),
            ({"message_id": "2"}, [{"candidate_id": "def"}])
        ]

        results = classify_batch(
            emails_with_candidates=emails_with_candidates,
            dictionary_version=1
        )

        assert len(results) == 2
        assert mock_classifier.classify.call_count == 2

    @patch('eml_classificator.classification.classifier.EmailClassifier')
    def test_classify_batch_partial_failure(self, mock_classifier_class):
        """Test batch classification continues after individual failures."""
        mock_classifier = Mock()

        # First succeeds, second fails, third succeeds
        mock_results = [
            Mock(spec=ClassificationResult),
            Exception("Classification failed"),
            Mock(spec=ClassificationResult)
        ]
        mock_classifier.classify.side_effect = mock_results
        mock_classifier_class.return_value = mock_classifier

        emails_with_candidates = [
            ({"message_id": "1"}, []),
            ({"message_id": "2"}, []),
            ({"message_id": "3"}, [])
        ]

        results = classify_batch(
            emails_with_candidates=emails_with_candidates,
            dictionary_version=1
        )

        # Should have 2 successful results (1 and 3)
        assert len(results) == 2

    @patch('eml_classificator.classification.classifier.logger')
    @patch('eml_classificator.classification.classifier.EmailClassifier')
    def test_classify_batch_logs_errors(self, mock_classifier_class, mock_logger):
        """Test batch classification logs errors for failed items."""
        mock_classifier = Mock()
        mock_classifier.classify.side_effect = Exception("Test error")
        mock_classifier_class.return_value = mock_classifier

        emails_with_candidates = [
            ({"message_id": "fail1"}, [])
        ]

        results = classify_batch(
            emails_with_candidates=emails_with_candidates,
            dictionary_version=1
        )

        # Should log error
        assert mock_logger.error.called

    @patch('eml_classificator.classification.classifier.EmailClassifier')
    def test_classify_batch_empty(self, mock_classifier_class):
        """Test batch classification with empty list."""
        mock_classifier = Mock()
        mock_classifier_class.return_value = mock_classifier

        results = classify_batch(
            emails_with_candidates=[],
            dictionary_version=1
        )

        assert len(results) == 0
        assert mock_classifier.classify.call_count == 0


class TestPostProcessing:
    """Test post-processing steps in classifier."""

    @patch('eml_classificator.classification.classifier.EmailClassifier.__init__')
    def test_customer_status_override_applied(self, mock_init):
        """Test customer status post-processing is applied."""
        mock_init.return_value = None

        classifier = EmailClassifier.__new__(EmailClassifier)
        classifier.logger = Mock()

        mock_llm_client = Mock()
        mock_llm_client.model = "test-model"
        mock_llm_client.classify.return_value = LLMResponse(
            tool_result=MOCK_VALID_LLM_OUTPUT,
            model="test-model",
            provider="test",
            tokens_total=100,
            latency_ms=1000
        )
        classifier.llm_client = mock_llm_client

        email_document = {
            "message_id": "test",
            "subject": "Test",
            "from_raw": "known@crm.com",  # Known customer
            "body_canonical": "Test"
        }
        candidates = [{"candidate_id": "abc123", "lemma": "test", "composite_score": 0.9}]

        with patch('eml_classificator.classification.classifier.build_classification_prompt') as mock_build, \
             patch('eml_classificator.classification.classifier.validate_llm_output_multistage') as mock_validate, \
             patch('eml_classificator.classification.classifier.override_customer_status_with_crm') as mock_override_status, \
             patch('eml_classificator.classification.classifier.override_priority_with_rules') as mock_override_priority, \
             patch('eml_classificator.classification.classifier.adjust_all_topic_confidences') as mock_adjust, \
             patch('eml_classificator.classification.classifier.get_current_pipeline_version') as mock_version:

            mock_build.return_value = ("sys", "user")
            mock_validate.return_value = Mock(
                valid=True,
                warnings=[],
                cleaned_data=MOCK_VALID_LLM_OUTPUT
            )

            # Mock override to verify it's called
            mock_override_status.return_value = CustomerStatus(
                value=CustomerStatusValue.EXISTING,
                confidence=1.0,
                source="crm_exact_match"
            )

            mock_override_priority.return_value = Priority(
                value=PriorityValue.HIGH,
                confidence=0.9,
                signals=[],
                raw_score=5.0
            )

            mock_adjust.return_value = [
                TopicAssignment(
                    label_id=TopicLabel.FATTURAZIONE,
                    confidence=0.9,
                    keywords_in_text=[KeywordInText(candidate_id="abc123", lemma="fattura", count=1)],
                    evidence=[Evidence(quote="test")]
                )
            ]
            mock_version.return_value = Mock(model_dump=lambda: {})

            result = classifier.classify(
                email_document=email_document,
                candidates=candidates,
                dictionary_version=1
            )

            # Verify override was called
            assert mock_override_status.called
            assert mock_override_priority.called
            assert mock_adjust.called


class TestMetadataGeneration:
    """Test classification metadata generation."""

    @patch('eml_classificator.classification.classifier.EmailClassifier.__init__')
    def test_metadata_includes_tokens(self, mock_init):
        """Test metadata includes token usage."""
        mock_init.return_value = None

        classifier = EmailClassifier.__new__(EmailClassifier)
        classifier.logger = Mock()

        mock_llm_client = Mock()
        mock_llm_client.model = "test-model"
        mock_llm_client.classify.return_value = LLMResponse(
            tool_result=MOCK_VALID_LLM_OUTPUT,
            model="test-model",
            provider="ollama",
            tokens_input=150,
            tokens_output=100,
            tokens_total=250,
            latency_ms=2341
        )
        classifier.llm_client = mock_llm_client

        email_document = {
            "message_id": "test",
            "subject": "Test",
            "from_raw": "test@test.com",
            "body_canonical": "Test"
        }
        candidates = [{"candidate_id": "abc123", "lemma": "test", "composite_score": 0.9}]

        with patch('eml_classificator.classification.classifier.build_classification_prompt') as mock_build, \
             patch('eml_classificator.classification.classifier.validate_llm_output_multistage') as mock_validate, \
             patch('eml_classificator.classification.classifier.override_customer_status_with_crm') as mock_override_status, \
             patch('eml_classificator.classification.classifier.override_priority_with_rules') as mock_override_priority, \
             patch('eml_classificator.classification.classifier.adjust_all_topic_confidences') as mock_adjust, \
             patch('eml_classificator.classification.classifier.get_current_pipeline_version') as mock_version:

            mock_build.return_value = ("sys", "user")
            mock_validate.return_value = Mock(
                valid=True,
                warnings=["test warning"],
                cleaned_data=MOCK_VALID_LLM_OUTPUT
            )
            mock_override_status.return_value = CustomerStatus(value=CustomerStatusValue.NEW, confidence=0.8, source="default")
            mock_override_priority.return_value = Priority(value=PriorityValue.LOW, confidence=0.7, signals=[], raw_score=1.0)
            mock_adjust.return_value = [
                TopicAssignment(
                    label_id=TopicLabel.FATTURAZIONE,
                    confidence=0.9,
                    keywords_in_text=[KeywordInText(candidate_id="abc123", lemma="fattura", count=1)],
                    evidence=[Evidence(quote="test")]
                )
            ]
            mock_version.return_value = Mock(model_dump=lambda: {"phase": 3})

            result = classifier.classify(
                email_document=email_document,
                candidates=candidates,
                dictionary_version=1
            )

            # Verify metadata
            assert result.metadata.tokens_input == 150
            assert result.metadata.tokens_output == 100
            assert result.metadata.tokens_total == 250
            assert result.metadata.model_used == "ollama/test-model"
            assert result.metadata.validation_warnings == ["test warning"]
            assert result.metadata.retry_count == 0
