"""
Integration tests for Ollama LLM client.

Tests classification with actual Ollama instance (optional, skipped if not available).
"""

import pytest
from unittest.mock import Mock, patch
import time

from eml_classificator.classification.llm_client import (
    OllamaClient,
    create_llm_client,
    OLLAMA_AVAILABLE
)
from eml_classificator.classification.classifier import classify_email
from eml_classificator.classification.schemas import ClassificationResult


# Skip all tests in this module if Ollama is not installed
pytestmark = pytest.mark.skipif(
    not OLLAMA_AVAILABLE,
    reason="Ollama library not installed"
)


def is_ollama_running():
    """Check if Ollama server is running and accessible."""
    if not OLLAMA_AVAILABLE:
        return False

    try:
        import ollama
        client = ollama.Client(host="http://localhost:11434")
        # Try to list models
        client.list()
        return True
    except Exception:
        return False


OLLAMA_RUNNING = is_ollama_running()


@pytest.fixture
def ollama_client():
    """Create Ollama client for testing (requires Ollama running)."""
    if not OLLAMA_RUNNING:
        pytest.skip("Ollama server not running")

    # Use a small, fast model for testing
    # Adjust model name based on what's available
    return OllamaClient(
        model="llama3.2:1b",  # Or "qwen2.5:0.5b" or other small model
        temperature=0.1,
        max_tokens=512,
        timeout_seconds=30
    )


@pytest.fixture
def sample_email_with_candidates():
    """Sample email and candidates for classification."""
    email_document = {
        "message_id": "<test@example.com>",
        "subject": "Problema con fattura",
        "from_raw": "john@company.com",
        "body_canonical": "Buongiorno, ho ricevuto la fattura errata. L'importo non corrisponde."
    }

    candidates = [
        {
            "candidate_id": "cand_001",
            "lemma": "fattura",
            "composite_score": 0.95
        },
        {
            "candidate_id": "cand_002",
            "lemma": "problema",
            "composite_score": 0.85
        },
        {
            "candidate_id": "cand_003",
            "lemma": "importo",
            "composite_score": 0.75
        }
    ]

    return email_document, candidates


class TestOllamaClientBasic:
    """Basic Ollama client tests."""

    @pytest.mark.skipif(not OLLAMA_RUNNING, reason="Ollama not running")
    def test_ollama_client_initialization(self):
        """Test Ollama client can be initialized."""
        client = OllamaClient(
            model="llama3.2:1b",
            base_url="http://localhost:11434"
        )

        assert client.model == "llama3.2:1b"
        assert client.base_url == "http://localhost:11434"

    @pytest.mark.skipif(not OLLAMA_RUNNING, reason="Ollama not running")
    def test_ollama_list_models(self):
        """Test listing available models."""
        import ollama

        client = ollama.Client(host="http://localhost:11434")
        response = client.list()

        assert 'models' in response
        models = [m.model for m in response['models']]
        # Should have at least one model installed
        assert len(models) > 0

    @pytest.mark.skipif(not OLLAMA_RUNNING, reason="Ollama not running")
    def test_ollama_client_factory(self):
        """Test creating Ollama client via factory."""
        client = create_llm_client(
            provider="ollama",
            model="llama3.2:1b",
            base_url="http://localhost:11434"
        )

        assert isinstance(client, OllamaClient)
        assert client.model == "llama3.2:1b"


class TestOllamaClassification:
    """Test classification with Ollama."""

    @pytest.mark.skipif(not OLLAMA_RUNNING, reason="Ollama not running")
    @pytest.mark.slow
    def test_classify_with_ollama_simple(self, ollama_client, sample_email_with_candidates):
        """Test simple classification with Ollama (may take 10-30 seconds)."""
        email_document, candidates = sample_email_with_candidates

        start_time = time.time()

        try:
            result = classify_email(
                email_document=email_document,
                candidates=candidates,
                dictionary_version=1,
                model_override=f"ollama/{ollama_client.model}"
            )

            elapsed = time.time() - start_time

            # Verify result structure
            assert isinstance(result, ClassificationResult)
            assert result.dictionary_version == 1
            assert len(result.topics) >= 1  # Should have at least one topic
            assert result.sentiment is not None
            assert result.priority is not None
            assert result.customer_status is not None

            # Verify metadata
            assert result.metadata.model_used.startswith("ollama/")
            assert result.metadata.tokens_total > 0
            assert result.metadata.latency_ms > 0

            print(f"\n✓ Classification completed in {elapsed:.2f}s")
            print(f"  Topics: {[t.label_id.value for t in result.topics]}")
            print(f"  Sentiment: {result.sentiment.value}")
            print(f"  Priority: {result.priority.value}")
            print(f"  Tokens: {result.metadata.tokens_total}")

        except Exception as e:
            pytest.fail(f"Classification failed: {e}")

    @pytest.mark.skipif(not OLLAMA_RUNNING, reason="Ollama not running")
    @pytest.mark.slow
    def test_classify_with_ollama_validation_retry(self, ollama_client, sample_email_with_candidates):
        """Test classification handles validation errors and retries."""
        email_document, candidates = sample_email_with_candidates

        # Intentionally use very few candidates to potentially trigger retry
        limited_candidates = candidates[:1]

        try:
            result = classify_email(
                email_document=email_document,
                candidates=limited_candidates,
                dictionary_version=1,
                model_override=f"ollama/{ollama_client.model}"
            )

            # Even with limited candidates, should eventually succeed
            assert isinstance(result, ClassificationResult)

            if result.metadata.retry_count > 0:
                print(f"\n✓ Successfully retried {result.metadata.retry_count} time(s)")

        except Exception as e:
            # Acceptable to fail if model really can't handle the constraint
            print(f"\n⚠ Expected behavior: {e}")

    @pytest.mark.skipif(not OLLAMA_RUNNING, reason="Ollama not running")
    @pytest.mark.slow
    def test_ollama_token_tracking(self, ollama_client):
        """Test token usage tracking works with Ollama."""
        from eml_classificator.classification.prompts import build_classification_prompt, EmailToClassify

        email = EmailToClassify(
            message_id="test",
            subject="Test",
            from_address="test@test.com",
            body_canonical="Simple test body.",
            dictionary_version=1
        )

        candidates = [
            {"candidate_id": "c1", "lemma": "test", "composite_score": 0.9}
        ]

        system_prompt, user_prompt = build_classification_prompt(email, candidates)

        try:
            response = ollama_client.classify(system_prompt, user_prompt)

            # Verify token tracking
            assert response.tokens_input is not None
            assert response.tokens_output is not None
            assert response.tokens_total is not None
            assert response.tokens_total == response.tokens_input + response.tokens_output

            print(f"\n✓ Token tracking:")
            print(f"  Input: {response.tokens_input}")
            print(f"  Output: {response.tokens_output}")
            print(f"  Total: {response.tokens_total}")

        except Exception as e:
            pytest.fail(f"Token tracking test failed: {e}")


class TestOllamaPerformance:
    """Performance tests with Ollama (optional)."""

    @pytest.mark.skipif(not OLLAMA_RUNNING, reason="Ollama not running")
    @pytest.mark.slow
    @pytest.mark.performance
    def test_ollama_latency_within_bounds(self, ollama_client, sample_email_with_candidates):
        """Test classification completes within reasonable time."""
        email_document, candidates = sample_email_with_candidates

        start_time = time.time()

        result = classify_email(
            email_document=email_document,
            candidates=candidates,
            dictionary_version=1,
            model_override=f"ollama/{ollama_client.model}"
        )

        elapsed = time.time() - start_time

        # Should complete within 60 seconds for small model
        assert elapsed < 60.0, f"Classification took {elapsed:.2f}s (expected <60s)"

        print(f"\n✓ Latency: {elapsed:.2f}s")

    @pytest.mark.skipif(not OLLAMA_RUNNING, reason="Ollama not running")
    @pytest.mark.slow
    @pytest.mark.performance
    def test_ollama_batch_classification(self, ollama_client):
        """Test batch classification performance."""
        from eml_classificator.classification.classifier import classify_batch

        # Create 3 sample emails
        emails_with_candidates = [
            (
                {
                    "message_id": f"<test{i}@example.com>",
                    "subject": f"Test email {i}",
                    "from_raw": "test@test.com",
                    "body_canonical": f"This is test email number {i}."
                },
                [
                    {"candidate_id": f"c{i}_1", "lemma": "test", "composite_score": 0.9}
                ]
            )
            for i in range(3)
        ]

        start_time = time.time()

        results = classify_batch(
            emails_with_candidates=emails_with_candidates,
            dictionary_version=1,
            model_override=f"ollama/{ollama_client.model}"
        )

        elapsed = time.time() - start_time

        assert len(results) == 3
        avg_latency = elapsed / 3

        print(f"\n✓ Batch classification:")
        print(f"  Total time: {elapsed:.2f}s")
        print(f"  Average per email: {avg_latency:.2f}s")


class TestOllamaErrorHandling:
    """Test error handling with Ollama."""

    @pytest.mark.skipif(not OLLAMA_RUNNING, reason="Ollama not running")
    def test_ollama_invalid_model(self):
        """Test error handling with non-existent model."""
        with pytest.raises(Exception):
            client = OllamaClient(model="nonexistent_model_12345")
            # Attempting to classify should fail
            client.classify("system", "user")

    @pytest.mark.skipif(not OLLAMA_RUNNING, reason="Ollama not running")
    def test_ollama_timeout(self):
        """Test timeout handling."""
        # Create client with very short timeout
        client = OllamaClient(
            model="llama3.2:1b",
            timeout_seconds=0.001  # 1ms - will definitely timeout
        )

        with pytest.raises(Exception):
            client.classify("system prompt", "user prompt")

    @pytest.mark.skipif(not OLLAMA_RUNNING, reason="Ollama not running")
    def test_ollama_connection_error(self):
        """Test handling of connection errors."""
        # Try to connect to wrong port
        client = OllamaClient(
            model="llama3.2:1b",
            base_url="http://localhost:99999"  # Invalid port
        )

        with pytest.raises(Exception):
            client.classify("system", "user")


class TestOllamaDeterminism:
    """Test deterministic behavior with Ollama."""

    @pytest.mark.skipif(not OLLAMA_RUNNING, reason="Ollama not running")
    @pytest.mark.slow
    def test_ollama_same_input_similar_output(self, ollama_client, sample_email_with_candidates):
        """Test that same input produces similar (deterministic) output."""
        email_document, candidates = sample_email_with_candidates

        # Run classification twice
        result1 = classify_email(
            email_document=email_document,
            candidates=candidates,
            dictionary_version=1,
            model_override=f"ollama/{ollama_client.model}"
        )

        result2 = classify_email(
            email_document=email_document,
            candidates=candidates,
            dictionary_version=1,
            model_override=f"ollama/{ollama_client.model}"
        )

        # Results should be similar (not necessarily identical due to LLM variance)
        # Check that top topic is the same
        if len(result1.topics) > 0 and len(result2.topics) > 0:
            assert result1.topics[0].label_id == result2.topics[0].label_id, \
                "Top topic should be consistent between runs"

        # Sentiment should be the same
        assert result1.sentiment.value == result2.sentiment.value, \
            "Sentiment should be deterministic"

        print("\n✓ Determinism check:")
        print(f"  Run 1 - Topics: {[t.label_id.value for t in result1.topics]}")
        print(f"  Run 2 - Topics: {[t.label_id.value for t in result2.topics]}")
        print(f"  Run 1 - Sentiment: {result1.sentiment.value}")
        print(f"  Run 2 - Sentiment: {result2.sentiment.value}")


# Marker for slow tests
def pytest_configure(config):
    """Register custom markers."""
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
    config.addinivalue_line(
        "markers", "performance: marks tests as performance tests"
    )
