"""
Unit tests for classification schemas.

Tests Pydantic models, enums, and validation logic.
"""

import pytest
from pydantic import ValidationError

from eml_classificator.classification.schemas import (
    TopicLabel,
    SentimentValue,
    PriorityValue,
    CustomerStatusValue,
    KeywordInText,
    Evidence,
    TopicAssignment,
    Sentiment,
    Priority,
    CustomerStatus,
    ClassificationResult,
    ClassificationMetadata,
)


class TestEnums:
    """Test enum definitions."""

    def test_topic_label_enum(self):
        """Test TopicLabel enum has all expected values."""
        expected_topics = [
            "FATTURAZIONE",
            "ASSISTENZA_TECNICA",
            "RECLAMO",
            "INFO_COMMERCIALI",
            "DOCUMENTI",
            "APPUNTAMENTO",
            "CONTRATTO",
            "GARANZIA",
            "SPEDIZIONE",
            "UNKNOWN_TOPIC",
        ]

        for topic in expected_topics:
            assert hasattr(TopicLabel, topic)
            assert TopicLabel[topic].value == topic

    def test_sentiment_enum(self):
        """Test SentimentValue enum."""
        assert SentimentValue.POSITIVE.value == "positive"
        assert SentimentValue.NEUTRAL.value == "neutral"
        assert SentimentValue.NEGATIVE.value == "negative"

    def test_priority_enum(self):
        """Test PriorityValue enum."""
        assert PriorityValue.LOW.value == "low"
        assert PriorityValue.MEDIUM.value == "medium"
        assert PriorityValue.HIGH.value == "high"
        assert PriorityValue.URGENT.value == "urgent"

    def test_customer_status_enum(self):
        """Test CustomerStatusValue enum."""
        assert CustomerStatusValue.NEW.value == "new"
        assert CustomerStatusValue.EXISTING.value == "existing"
        assert CustomerStatusValue.UNKNOWN.value == "unknown"


class TestKeywordInText:
    """Test KeywordInText model."""

    def test_valid_keyword(self):
        """Test valid keyword creation."""
        kw = KeywordInText(
            candidate_id="abc123",
            lemma="fattura",
            count=3
        )

        assert kw.candidate_id == "abc123"
        assert kw.lemma == "fattura"
        assert kw.count == 3
        assert kw.spans is None

    def test_keyword_with_spans(self):
        """Test keyword with span positions."""
        kw = KeywordInText(
            candidate_id="abc123",
            lemma="fattura",
            count=2,
            spans=[[10, 18], [50, 58]]
        )

        assert len(kw.spans) == 2
        assert kw.spans[0] == [10, 18]

    def test_keyword_invalid_count(self):
        """Test keyword with invalid count (must be >= 1)."""
        with pytest.raises(ValidationError) as exc_info:
            KeywordInText(
                candidate_id="abc123",
                lemma="fattura",
                count=0  # Invalid
            )

        assert "count" in str(exc_info.value)

    def test_keyword_invalid_span_format(self):
        """Test keyword with invalid span format."""
        with pytest.raises(ValidationError) as exc_info:
            KeywordInText(
                candidate_id="abc123",
                lemma="fattura",
                count=1,
                spans=[[10]]  # Invalid - must be [start, end]
            )

        assert "span" in str(exc_info.value).lower()


class TestEvidence:
    """Test Evidence model."""

    def test_valid_evidence(self):
        """Test valid evidence creation."""
        ev = Evidence(quote="ho ricevuto la fattura errata")

        assert ev.quote == "ho ricevuto la fattura errata"
        assert ev.span is None

    def test_evidence_with_span(self):
        """Test evidence with span."""
        ev = Evidence(
            quote="fattura errata",
            span=[45, 60]
        )

        assert ev.span == [45, 60]

    def test_evidence_quote_too_long(self):
        """Test evidence quote exceeds max length (200 chars)."""
        long_quote = "a" * 201

        with pytest.raises(ValidationError) as exc_info:
            Evidence(quote=long_quote)

        assert "quote" in str(exc_info.value).lower()

    def test_evidence_invalid_span(self):
        """Test evidence with invalid span (start >= end)."""
        with pytest.raises(ValidationError) as exc_info:
            Evidence(
                quote="test",
                span=[60, 45]  # Invalid: start > end
            )

        assert "span" in str(exc_info.value).lower()


class TestTopicAssignment:
    """Test TopicAssignment model."""

    def test_valid_topic_assignment(self):
        """Test valid topic assignment."""
        topic = TopicAssignment(
            label_id=TopicLabel.FATTURAZIONE,
            confidence=0.92,
            keywords_in_text=[
                KeywordInText(candidate_id="abc123", lemma="fattura", count=3)
            ],
            evidence=[
                Evidence(quote="ho ricevuto la fattura")
            ]
        )

        assert topic.label_id == TopicLabel.FATTURAZIONE
        assert topic.confidence == 0.92
        assert len(topic.keywords_in_text) == 1
        assert len(topic.evidence) == 1

    def test_topic_confidence_out_of_range(self):
        """Test topic with confidence out of [0, 1] range."""
        with pytest.raises(ValidationError) as exc_info:
            TopicAssignment(
                label_id=TopicLabel.FATTURAZIONE,
                confidence=1.5,  # Invalid
                keywords_in_text=[
                    KeywordInText(candidate_id="abc123", lemma="fattura", count=1)
                ],
                evidence=[Evidence(quote="test")]
            )

        assert "confidence" in str(exc_info.value).lower()

    def test_topic_too_many_keywords(self):
        """Test topic exceeds max keywords (15)."""
        keywords = [
            KeywordInText(candidate_id=f"id{i}", lemma=f"term{i}", count=1)
            for i in range(20)  # Exceeds max
        ]

        with pytest.raises(ValidationError) as exc_info:
            TopicAssignment(
                label_id=TopicLabel.FATTURAZIONE,
                confidence=0.9,
                keywords_in_text=keywords,
                evidence=[Evidence(quote="test")]
            )

        assert "keywords_in_text" in str(exc_info.value).lower()

    def test_topic_no_keywords(self):
        """Test topic with no keywords (must have at least 1)."""
        with pytest.raises(ValidationError) as exc_info:
            TopicAssignment(
                label_id=TopicLabel.FATTURAZIONE,
                confidence=0.9,
                keywords_in_text=[],  # Empty - invalid
                evidence=[Evidence(quote="test")]
            )

        assert "keywords_in_text" in str(exc_info.value).lower()


class TestSentiment:
    """Test Sentiment model."""

    def test_valid_sentiment(self):
        """Test valid sentiment."""
        sentiment = Sentiment(
            value=SentimentValue.NEGATIVE,
            confidence=0.85
        )

        assert sentiment.value == SentimentValue.NEGATIVE
        assert sentiment.confidence == 0.85

    def test_sentiment_invalid_confidence(self):
        """Test sentiment with invalid confidence."""
        with pytest.raises(ValidationError):
            Sentiment(
                value=SentimentValue.POSITIVE,
                confidence=-0.1  # Invalid
            )


class TestPriority:
    """Test Priority model."""

    def test_valid_priority(self):
        """Test valid priority."""
        priority = Priority(
            value=PriorityValue.HIGH,
            confidence=0.88,
            signals=["fattura", "urgente", "problema"]
        )

        assert priority.value == PriorityValue.HIGH
        assert len(priority.signals) == 3
        assert priority.raw_score is None

    def test_priority_with_raw_score(self):
        """Test priority with raw score."""
        priority = Priority(
            value=PriorityValue.URGENT,
            confidence=0.95,
            signals=["bloccante"],
            raw_score=7.5
        )

        assert priority.raw_score == 7.5

    def test_priority_too_many_signals(self):
        """Test priority exceeds max signals (10)."""
        with pytest.raises(ValidationError):
            Priority(
                value=PriorityValue.HIGH,
                confidence=0.9,
                signals=[f"signal{i}" for i in range(15)]  # Exceeds max
            )


class TestCustomerStatus:
    """Test CustomerStatus model."""

    def test_valid_customer_status(self):
        """Test valid customer status."""
        status = CustomerStatus(
            value=CustomerStatusValue.EXISTING,
            confidence=1.0,
            source="crm_exact_match"
        )

        assert status.value == CustomerStatusValue.EXISTING
        assert status.confidence == 1.0
        assert status.source == "crm_exact_match"

    def test_customer_status_text_signal(self):
        """Test customer status from text signal."""
        status = CustomerStatus(
            value=CustomerStatusValue.EXISTING,
            confidence=0.5,
            source="text_signal"
        )

        assert status.source == "text_signal"


class TestClassificationResult:
    """Test ClassificationResult model."""

    def test_valid_classification_result(self):
        """Test valid complete classification result."""
        from datetime import datetime

        result = ClassificationResult(
            dictionary_version=1,
            topics=[
                TopicAssignment(
                    label_id=TopicLabel.FATTURAZIONE,
                    confidence=0.92,
                    keywords_in_text=[
                        KeywordInText(candidate_id="abc123", lemma="fattura", count=3)
                    ],
                    evidence=[Evidence(quote="ho ricevuto la fattura")]
                )
            ],
            sentiment=Sentiment(
                value=SentimentValue.NEGATIVE,
                confidence=0.85
            ),
            priority=Priority(
                value=PriorityValue.HIGH,
                confidence=0.88,
                signals=["fattura", "problema"]
            ),
            customer_status=CustomerStatus(
                value=CustomerStatusValue.EXISTING,
                confidence=1.0,
                source="crm_exact_match"
            ),
            metadata=ClassificationMetadata(
                pipeline_version={},
                model_used="ollama/deepseek-r1:7b",
                latency_ms=2341,
                classified_at=datetime.utcnow()
            )
        )

        assert len(result.topics) == 1
        assert result.sentiment.value == SentimentValue.NEGATIVE
        assert result.priority.value == PriorityValue.HIGH
        assert result.customer_status.value == CustomerStatusValue.EXISTING

    def test_classification_result_no_topics(self):
        """Test classification result with no topics (must have at least 1)."""
        from datetime import datetime

        with pytest.raises(ValidationError) as exc_info:
            ClassificationResult(
                dictionary_version=1,
                topics=[],  # Empty - invalid
                sentiment=Sentiment(value=SentimentValue.NEUTRAL, confidence=0.9),
                priority=Priority(value=PriorityValue.LOW, confidence=0.7, signals=[]),
                customer_status=CustomerStatus(
                    value=CustomerStatusValue.UNKNOWN,
                    confidence=0.5,
                    source="no_crm"
                ),
                metadata=ClassificationMetadata(
                    pipeline_version={},
                    model_used="test",
                    latency_ms=1000,
                    classified_at=datetime.utcnow()
                )
            )

        assert "topics" in str(exc_info.value).lower()

    def test_classification_result_too_many_topics(self):
        """Test classification result exceeds max topics (5)."""
        from datetime import datetime

        topics = [
            TopicAssignment(
                label_id=list(TopicLabel)[i % len(TopicLabel)],
                confidence=0.8,
                keywords_in_text=[
                    KeywordInText(candidate_id=f"id{i}", lemma=f"term{i}", count=1)
                ],
                evidence=[Evidence(quote="test")]
            )
            for i in range(7)  # Exceeds max
        ]

        with pytest.raises(ValidationError):
            ClassificationResult(
                dictionary_version=1,
                topics=topics,
                sentiment=Sentiment(value=SentimentValue.NEUTRAL, confidence=0.9),
                priority=Priority(value=PriorityValue.LOW, confidence=0.7, signals=[]),
                customer_status=CustomerStatus(
                    value=CustomerStatusValue.UNKNOWN,
                    confidence=0.5,
                    source="no_crm"
                ),
                metadata=ClassificationMetadata(
                    pipeline_version={},
                    model_used="test",
                    latency_ms=1000,
                    classified_at=datetime.utcnow()
                )
            )
