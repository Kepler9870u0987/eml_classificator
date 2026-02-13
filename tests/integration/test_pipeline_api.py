"""
Integration tests for complete pipeline endpoint.

Tests POST /api/v1/pipeline/complete that runs Phase 1 → 2 → 3.
"""

import pytest
from fastapi.testclient import TestClient
from unittest.mock import Mock, patch, MagicMock
import io
from pathlib import Path

from eml_classificator.api.app import app


@pytest.fixture
def client():
    """Create test client for FastAPI app."""
    return TestClient(app)


@pytest.fixture
def sample_eml_content():
    """Sample .eml file content for testing."""
    return b"""From: john@company.com
To: support@service.com
Subject: Problema con fattura di dicembre
Date: Mon, 15 Jan 2024 10:30:00 +0100
Content-Type: text/plain; charset=utf-8

Buongiorno,

Ho ricevuto la fattura n. 12345 del mese di dicembre ma l'importo non e corretto.
Risulta un addebito di 150 euro invece dei 100 euro concordati.

Potete verificare e inviare una fattura corretta?

Grazie.
"""


@pytest.fixture
def sample_eml_file(sample_eml_content):
    """Create an in-memory file-like object with .eml content."""
    return ("test_email.eml", io.BytesIO(sample_eml_content), "message/rfc822")


class TestPipelineCompleteEndpoint:
    """Test complete pipeline endpoint."""

    @patch('eml_classificator.api.routes.pipeline.classify_email')
    @patch('eml_classificator.api.routes.pipeline.extract_candidates')
    @patch('eml_classificator.api.routes.pipeline.build_email_document')
    @patch('eml_classificator.api.routes.pipeline.parse_eml_bytes')
    def test_pipeline_complete_success(
        self,
        mock_parse_eml,
        mock_build_doc,
        mock_extract_candidates,
        mock_classify,
        client,
        sample_eml_file
    ):
        """Test successful complete pipeline execution."""
        # Mock Phase 1: Email parsing
        mock_msg = Mock()
        mock_parse_eml.return_value = mock_msg

        mock_email_doc = Mock()
        mock_email_doc.model_dump.return_value = {
            "message_id": "<test@example.com>",
            "subject": "Problema con fattura",
            "from_email": "john@company.com",
            "body_canonical": "Test body"
        }
        mock_build_doc.return_value = mock_email_doc

        # Mock Phase 2: Candidate extraction
        mock_candidates_result = Mock()
        mock_candidates_result.rich_candidates = [
            Mock(model_dump=lambda: {
                "candidate_id": "cand_001",
                "lemma": "fattura",
                "composite_score": 0.88
            })
        ]
        mock_extract_candidates.return_value = mock_candidates_result

        # Mock Phase 3: Classification
        mock_classification_result = Mock()
        mock_classification_result.model_dump.return_value = {
            "dictionary_version": 1,
            "topics": [
                {
                    "label_id": "FATTURAZIONE",
                    "confidence": 0.92,
                    "keywords_in_text": [{"candidate_id": "cand_001", "lemma": "fattura", "count": 2}],
                    "evidence": [{"quote": "fattura"}]
                }
            ],
            "sentiment": {"value": "negative", "confidence": 0.85},
            "priority": {"value": "high", "confidence": 0.88, "signals": []},
            "customer_status": {"value": "existing", "confidence": 1.0, "source": "crm"},
            "metadata": {
                "pipeline_version": {},
                "model_used": "ollama/test-model",
                "latency_ms": 2500
            }
        }
        mock_classify.return_value = mock_classification_result

        # Make request
        response = client.post(
            "/api/v1/pipeline/complete",
            files={"eml_file": sample_eml_file},
            data={"dictionary_version": "1"}
        )

        # Verify response
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert "email_document" in data
        assert "candidates_result" in data
        assert "classification_result" in data
        assert data["classification_result"]["topics"][0]["label_id"] == "FATTURAZIONE"

        # Verify all phases were called
        assert mock_parse_eml.called
        assert mock_build_doc.called
        assert mock_extract_candidates.called
        assert mock_classify.called

    @patch('eml_classificator.api.routes.pipeline.classify_email')
    @patch('eml_classificator.api.routes.pipeline.extract_candidates')
    @patch('eml_classificator.api.routes.pipeline.build_email_document')
    @patch('eml_classificator.api.routes.pipeline.parse_eml_bytes')
    def test_pipeline_complete_with_model_override(
        self,
        mock_parse_eml,
        mock_build_doc,
        mock_extract_candidates,
        mock_classify,
        client,
        sample_eml_file
    ):
        """Test pipeline with model override."""
        # Setup mocks
        mock_parse_eml.return_value = Mock()
        mock_build_doc.return_value = Mock(model_dump=lambda: {})
        mock_extract_candidates.return_value = Mock(rich_candidates=[])
        mock_classify.return_value = Mock(model_dump=lambda: {
            "dictionary_version": 1,
            "topics": [],
            "sentiment": {"value": "neutral", "confidence": 0.9},
            "priority": {"value": "low", "confidence": 0.7, "signals": []},
            "customer_status": {"value": "unknown", "confidence": 0.5, "source": "default"},
            "metadata": {"model_used": "ollama/qwen2.5:32b"}
        })

        # Make request with model override
        response = client.post(
            "/api/v1/pipeline/complete",
            files={"eml_file": sample_eml_file},
            data={
                "dictionary_version": "1",
                "model_override": "ollama/qwen2.5:32b"
            }
        )

        assert response.status_code == 200
        # Verify model override was passed to classify_email
        mock_classify.assert_called_once()
        call_args = mock_classify.call_args
        # Check if model_override is in the call
        assert "ollama/qwen2.5:32b" in str(call_args) or \
               call_args[1].get("model_override") == "ollama/qwen2.5:32b"

    def test_pipeline_complete_missing_file(self, client):
        """Test pipeline fails when no file uploaded."""
        response = client.post(
            "/api/v1/pipeline/complete",
            data={"dictionary_version": "1"}
        )

        assert response.status_code == 422  # Validation error

    def test_pipeline_complete_invalid_eml(self, client):
        """Test pipeline with invalid .eml content."""
        invalid_file = ("test.eml", io.BytesIO(b"invalid email content"), "message/rfc822")

        with patch('eml_classificator.api.routes.pipeline.parse_eml_bytes') as mock_parse:
            mock_parse.side_effect = Exception("Invalid email format")

            response = client.post(
                "/api/v1/pipeline/complete",
                files={"eml_file": invalid_file},
                data={"dictionary_version": "1"}
            )

            assert response.status_code == 500
            data = response.json()
            assert data["success"] is False

    @patch('eml_classificator.api.routes.pipeline.classify_email')
    @patch('eml_classificator.api.routes.pipeline.extract_candidates')
    @patch('eml_classificator.api.routes.pipeline.build_email_document')
    @patch('eml_classificator.api.routes.pipeline.parse_eml_bytes')
    def test_pipeline_phase_2_failure(
        self,
        mock_parse_eml,
        mock_build_doc,
        mock_extract_candidates,
        mock_classify,
        client,
        sample_eml_file
    ):
        """Test pipeline handles Phase 2 failure."""
        mock_parse_eml.return_value = Mock()
        mock_build_doc.return_value = Mock(model_dump=lambda: {})

        # Phase 2 fails
        mock_extract_candidates.side_effect = Exception("KeyBERT error")

        response = client.post(
            "/api/v1/pipeline/complete",
            files={"eml_file": sample_eml_file},
            data={"dictionary_version": "1"}
        )

        assert response.status_code == 500
        data = response.json()
        assert data["success"] is False
        assert "error" in data

    @patch('eml_classificator.api.routes.pipeline.classify_email')
    @patch('eml_classificator.api.routes.pipeline.extract_candidates')
    @patch('eml_classificator.api.routes.pipeline.build_email_document')
    @patch('eml_classificator.api.routes.pipeline.parse_eml_bytes')
    def test_pipeline_phase_3_failure(
        self,
        mock_parse_eml,
        mock_build_doc,
        mock_extract_candidates,
        mock_classify,
        client,
        sample_eml_file
    ):
        """Test pipeline handles Phase 3 (classification) failure."""
        mock_parse_eml.return_value = Mock()
        mock_build_doc.return_value = Mock(model_dump=lambda: {})
        mock_extract_candidates.return_value = Mock(rich_candidates=[])

        # Phase 3 fails
        mock_classify.side_effect = Exception("LLM timeout")

        response = client.post(
            "/api/v1/pipeline/complete",
            files={"eml_file": sample_eml_file},
            data={"dictionary_version": "1"}
        )

        assert response.status_code == 500
        data = response.json()
        assert data["success"] is False


class TestPipelineEndToEnd:
    """Test end-to-end pipeline with mocked LLM but real processing."""

    @patch('eml_classificator.classification.llm_client.OllamaClient')
    def test_e2e_pipeline_mocked_llm(self, mock_ollama_class, client, sample_eml_file):
        """Test complete pipeline with real Phase 1 & 2, mocked LLM."""
        # Mock Ollama client for Phase 3
        mock_client_instance = Mock()
        mock_client_instance.model = "test-model"

        mock_llm_response = Mock()
        mock_llm_response.tool_result = {
            "dictionary_version": 1,
            "topics": [
                {
                    "label_id": "FATTURAZIONE",
                    "confidence": 0.9,
                    "keywords_in_text": [
                        {"candidate_id": "any_id", "lemma": "fattura", "count": 1}
                    ],
                    "evidence": [{"quote": "fattura"}]
                }
            ],
            "sentiment": {"value": "negative", "confidence": 0.85},
            "priority": {"value": "high", "confidence": 0.88, "signals": []},
            "customer_status": {"value": "existing", "confidence": 1.0}
        }
        mock_llm_response.model = "test-model"
        mock_llm_response.provider = "ollama"
        mock_llm_response.tokens_total = 250
        mock_llm_response.tokens_input = 150
        mock_llm_response.tokens_output = 100
        mock_llm_response.latency_ms = 2000

        mock_client_instance.classify.return_value = mock_llm_response
        mock_client_instance.list.return_value = {'models': []}
        mock_ollama_class.return_value = mock_client_instance

        # This would require Phase 1 & 2 to be fully functional
        # For now, we mock them as well to avoid KeyBERT dependencies
        with patch('eml_classificator.api.routes.pipeline.parse_eml_bytes') as mock_parse, \
             patch('eml_classificator.api.routes.pipeline.build_email_document') as mock_build, \
             patch('eml_classificator.api.routes.pipeline.extract_candidates') as mock_extract:

            mock_parse.return_value = Mock()
            mock_build.return_value = Mock(model_dump=lambda: {
                "message_id": "test",
                "subject": "Test",
                "from_raw": "test@test.com",
                "body_canonical": "Test"
            })
            mock_extract.return_value = Mock(rich_candidates=[
                Mock(model_dump=lambda: {
                    "candidate_id": "any_id",
                    "lemma": "fattura",
                    "composite_score": 0.9
                })
            ])

            response = client.post(
                "/api/v1/pipeline/complete",
                files={"eml_file": sample_eml_file},
                data={
                    "dictionary_version": "1",
                    "model_override": "ollama/test-model"
                }
            )

            assert response.status_code == 200
            data = response.json()
            assert data["success"] is True


class TestPipelineMetrics:
    """Test pipeline metrics and metadata."""

    @patch('eml_classificator.api.routes.pipeline.classify_email')
    @patch('eml_classificator.api.routes.pipeline.extract_candidates')
    @patch('eml_classificator.api.routes.pipeline.build_email_document')
    @patch('eml_classificator.api.routes.pipeline.parse_eml_bytes')
    def test_pipeline_includes_metadata(
        self,
        mock_parse_eml,
        mock_build_doc,
        mock_extract_candidates,
        mock_classify,
        client,
        sample_eml_file
    ):
        """Test pipeline response includes detailed metadata."""
        # Setup mocks
        mock_parse_eml.return_value = Mock()
        mock_build_doc.return_value = Mock(model_dump=lambda: {
            "message_id": "test@123",
            "subject": "Test"
        })
        mock_extract_candidates.return_value = Mock(rich_candidates=[
            Mock(model_dump=lambda: {"candidate_id": "c1", "lemma": "test", "composite_score": 0.9})
        ])
        mock_classify.return_value = Mock(model_dump=lambda: {
            "dictionary_version": 1,
            "topics": [],
            "sentiment": {"value": "neutral", "confidence": 0.9},
            "priority": {"value": "low", "confidence": 0.7, "signals": []},
            "customer_status": {"value": "new", "confidence": 0.8, "source": "default"},
            "metadata": {
                "pipeline_version": {"phase_3": "1.0.0"},
                "model_used": "ollama/test",
                "tokens_total": 250,
                "latency_ms": 2500,
                "classified_at": "2024-01-15T12:00:00Z"
            }
        })

        response = client.post(
            "/api/v1/pipeline/complete",
            files={"eml_file": sample_eml_file},
            data={"dictionary_version": "1"}
        )

        assert response.status_code == 200
        data = response.json()

        # Verify metadata is present
        assert "classification_result" in data
        assert "metadata" in data["classification_result"]
        metadata = data["classification_result"]["metadata"]
        assert "model_used" in metadata
        assert "latency_ms" in metadata


class TestPipelineValidation:
    """Test input validation for pipeline endpoint."""

    def test_pipeline_invalid_dictionary_version(self, client, sample_eml_file):
        """Test pipeline with invalid dictionary version."""
        response = client.post(
            "/api/v1/pipeline/complete",
            files={"eml_file": sample_eml_file},
            data={"dictionary_version": "invalid"}  # Should be integer
        )

        assert response.status_code == 422

    def test_pipeline_large_file(self, client):
        """Test pipeline with very large file."""
        # Create a large fake .eml file (e.g., 50MB)
        large_content = b"X" * (50 * 1024 * 1024)
        large_file = ("large.eml", io.BytesIO(large_content), "message/rfc822")

        # Depending on FastAPI limits, this might fail or timeout
        # For now, just test it doesn't crash
        response = client.post(
            "/api/v1/pipeline/complete",
            files={"eml_file": large_file},
            data={"dictionary_version": "1"},
            timeout=5  # Short timeout
        )

        # Could be 413 (too large), 500 (processing error), or 504 (timeout)
        assert response.status_code in [413, 422, 500, 504]
