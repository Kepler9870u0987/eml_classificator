"""
Integration tests for classification API endpoint.

Tests the POST /api/v1/classify endpoint with real FastAPI test client.
"""

import pytest
from fastapi.testclient import TestClient
from unittest.mock import Mock, patch
import json

from eml_classificator.api.app import app
from eml_classificator.classification.schemas import TopicLabel


@pytest.fixture
def client():
    """Create test client for FastAPI app."""
    return TestClient(app)


@pytest.fixture
def sample_email_document():
    """Sample email document for testing."""
    return {
        "message_id": "<test@example.com>",
        "subject": "Problema con fattura di dicembre",
        "from_raw": "john@company.com",
        "from_name": "John Doe",
        "from_email": "john@company.com",
        "to_raw": "support@service.com",
        "date_parsed": "2024-01-15T10:30:00Z",
        "body_canonical": "Buongiorno, ho ricevuto la fattura n. 12345 del mese di dicembre ma l'importo non è corretto. Risulta un addebito di 150 euro invece dei 100 euro concordati. Potete verificare e inviare una fattura corretta? Grazie.",
        "body_original_html": "<html><body>Buongiorno...</body></html>",
        "body_original_text": "Buongiorno, ho ricevuto la fattura...",
        "attachments_count": 0,
        "has_html_body": True
    }


@pytest.fixture
def sample_candidates():
    """Sample candidate keywords for testing."""
    return [
        {
            "candidate_id": "cand_001",
            "lemma": "fattura",
            "original_forms": ["fattura"],
            "count": 2,
            "ngram_size": 1,
            "positions": [[0, 8], [120, 128]],
            "keybert_score": 0.85,
            "composite_score": 0.88
        },
        {
            "candidate_id": "cand_002",
            "lemma": "problema",
            "original_forms": ["Problema"],
            "count": 1,
            "ngram_size": 1,
            "positions": [[0, 8]],
            "keybert_score": 0.75,
            "composite_score": 0.78
        },
        {
            "candidate_id": "cand_003",
            "lemma": "importo",
            "original_forms": ["importo"],
            "count": 1,
            "ngram_size": 1,
            "positions": [[95, 102]],
            "keybert_score": 0.65,
            "composite_score": 0.68
        }
    ]


class TestClassificationEndpoint:
    """Test classification API endpoint."""

    @patch('eml_classificator.api.routes.classification.classify_email')
    def test_classify_endpoint_success(self, mock_classify, client, sample_email_document, sample_candidates):
        """Test successful classification request."""
        # Mock classification result
        mock_result = Mock()
        mock_result.model_dump.return_value = {
            "dictionary_version": 1,
            "topics": [
                {
                    "label_id": "FATTURAZIONE",
                    "confidence": 0.92,
                    "keywords_in_text": [
                        {
                            "candidate_id": "cand_001",
                            "lemma": "fattura",
                            "count": 2
                        }
                    ],
                    "evidence": [
                        {"quote": "ho ricevuto la fattura n. 12345"}
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
                "signals": ["fattura", "problema"],
                "raw_score": 5.2
            },
            "customer_status": {
                "value": "existing",
                "confidence": 1.0,
                "source": "crm_exact_match"
            },
            "metadata": {
                "pipeline_version": {},
                "model_used": "ollama/deepseek-r1:7b",
                "tokens_input": 150,
                "tokens_output": 100,
                "tokens_total": 250,
                "latency_ms": 2341,
                "llm_latency_ms": 2000,
                "validation_warnings": [],
                "retry_count": 0,
                "classified_at": "2024-01-15T12:00:00Z"
            }
        }
        mock_classify.return_value = mock_result

        # Make request
        response = client.post(
            "/api/v1/classify",
            json={
                "email_document": sample_email_document,
                "candidates": sample_candidates,
                "dictionary_version": 1
            }
        )

        # Verify response
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert "result" in data
        assert data["result"]["topics"][0]["label_id"] == "FATTURAZIONE"
        assert data["result"]["sentiment"]["value"] == "negative"
        assert data["result"]["priority"]["value"] == "high"

    @patch('eml_classificator.api.routes.classification.classify_email')
    def test_classify_endpoint_with_model_override(self, mock_classify, client, sample_email_document, sample_candidates):
        """Test classification with model override."""
        mock_result = Mock()
        mock_result.model_dump.return_value = {
            "dictionary_version": 1,
            "topics": [],
            "sentiment": {"value": "neutral", "confidence": 0.9},
            "priority": {"value": "low", "confidence": 0.7, "signals": []},
            "customer_status": {"value": "unknown", "confidence": 0.5, "source": "default"},
            "metadata": {"model_used": "ollama/qwen2.5:32b"}
        }
        mock_classify.return_value = mock_result

        response = client.post(
            "/api/v1/classify",
            json={
                "email_document": sample_email_document,
                "candidates": sample_candidates,
                "dictionary_version": 1,
                "model_override": "ollama/qwen2.5:32b"
            }
        )

        assert response.status_code == 200
        # Verify model override was passed
        mock_classify.assert_called_once()
        call_kwargs = mock_classify.call_args[1]
        assert call_kwargs.get("model_override") == "ollama/qwen2.5:32b"

    def test_classify_endpoint_missing_required_fields(self, client):
        """Test classification fails with missing required fields."""
        response = client.post(
            "/api/v1/classify",
            json={
                "email_document": {},
                # Missing candidates
                "dictionary_version": 1
            }
        )

        assert response.status_code == 422  # Validation error

    def test_classify_endpoint_invalid_email_document(self, client, sample_candidates):
        """Test classification with invalid email document."""
        response = client.post(
            "/api/v1/classify",
            json={
                "email_document": "invalid",  # Should be dict
                "candidates": sample_candidates,
                "dictionary_version": 1
            }
        )

        assert response.status_code == 422

    @patch('eml_classificator.api.routes.classification.classify_email')
    def test_classify_endpoint_classification_error(self, mock_classify, client, sample_email_document, sample_candidates):
        """Test classification endpoint handles errors gracefully."""
        mock_classify.side_effect = Exception("LLM timeout")

        response = client.post(
            "/api/v1/classify",
            json={
                "email_document": sample_email_document,
                "candidates": sample_candidates,
                "dictionary_version": 1
            }
        )

        assert response.status_code == 500
        data = response.json()
        assert data["success"] is False
        assert "error" in data


class TestBatchClassificationEndpoint:
    """Test batch classification API endpoint."""

    @patch('eml_classificator.api.routes.classification.classify_batch')
    def test_batch_classify_endpoint_success(self, mock_classify_batch, client, sample_email_document, sample_candidates):
        """Test successful batch classification."""
        # Mock batch results
        mock_results = [
            Mock(model_dump=lambda: {
                "dictionary_version": 1,
                "topics": [{"label_id": "FATTURAZIONE", "confidence": 0.9}],
                "sentiment": {"value": "negative", "confidence": 0.85},
                "priority": {"value": "high", "confidence": 0.88, "signals": []},
                "customer_status": {"value": "existing", "confidence": 1.0, "source": "crm"},
                "metadata": {}
            }),
            Mock(model_dump=lambda: {
                "dictionary_version": 1,
                "topics": [{"label_id": "RECLAMO", "confidence": 0.85}],
                "sentiment": {"value": "negative", "confidence": 0.9},
                "priority": {"value": "urgent", "confidence": 0.95, "signals": []},
                "customer_status": {"value": "new", "confidence": 0.8, "source": "default"},
                "metadata": {}
            })
        ]
        mock_classify_batch.return_value = mock_results

        response = client.post(
            "/api/v1/classify/batch",
            json={
                "batch": [
                    {
                        "email_document": sample_email_document,
                        "candidates": sample_candidates,
                        "dictionary_version": 1
                    },
                    {
                        "email_document": sample_email_document,
                        "candidates": sample_candidates,
                        "dictionary_version": 1
                    }
                ]
            }
        )

        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert len(data["results"]) == 2
        assert data["results"][0]["topics"][0]["label_id"] == "FATTURAZIONE"
        assert data["results"][1]["topics"][0]["label_id"] == "RECLAMO"

    def test_batch_classify_endpoint_empty_batch(self, client):
        """Test batch classification with empty batch."""
        response = client.post(
            "/api/v1/classify/batch",
            json={"batch": []}
        )

        assert response.status_code == 422  # Should require at least 1 item


class TestHealthEndpoint:
    """Test health check endpoint."""

    def test_health_endpoint(self, client):
        """Test classification health check."""
        response = client.get("/api/v1/classify/health")

        assert response.status_code == 200
        data = response.json()
        assert "status" in data
        assert "llm_available" in data


class TestEndToEndClassification:
    """Test end-to-end classification flow (mocked LLM)."""

    @patch('eml_classificator.classification.llm_client.OllamaClient')
    def test_e2e_classification_with_mocked_llm(self, mock_ollama_class, client, sample_email_document, sample_candidates):
        """Test complete classification flow with mocked LLM client."""
        # Mock Ollama client
        mock_client_instance = Mock()
        mock_client_instance.model = "test-model"

        # Mock LLM response
        mock_llm_response = Mock()
        mock_llm_response.tool_result = {
            "dictionary_version": 1,
            "topics": [
                {
                    "label_id": "FATTURAZIONE",
                    "confidence": 0.92,
                    "keywords_in_text": [
                        {
                            "candidate_id": "cand_001",
                            "lemma": "fattura",
                            "count": 2
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
                "confidence": 1.0
            }
        }
        mock_llm_response.model = "test-model"
        mock_llm_response.provider = "ollama"
        mock_llm_response.tokens_input = 150
        mock_llm_response.tokens_output = 100
        mock_llm_response.tokens_total = 250
        mock_llm_response.latency_ms = 2000

        mock_client_instance.classify.return_value = mock_llm_response
        mock_client_instance.list.return_value = {'models': []}
        mock_ollama_class.return_value = mock_client_instance

        # Make real API call
        response = client.post(
            "/api/v1/classify",
            json={
                "email_document": sample_email_document,
                "candidates": sample_candidates,
                "dictionary_version": 1,
                "model_override": "ollama/test-model"
            }
        )

        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert data["result"]["topics"][0]["label_id"] == "FATTURAZIONE"
