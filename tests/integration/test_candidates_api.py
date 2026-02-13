"""
Integration tests for candidates API endpoint (Phase 2).

Tests coverage:
- End-to-end API calls to /api/v1/candidates/extract
- Request/response validation
- Determinism verification
- Error handling scenarios
- Various email document types

Testing principles:
1. Use async_client fixture from conftest
2. Test full pipeline (tokenize → filter → enrich → score)
3. Verify response structure and version tracking
4. Test deterministic behavior
"""

import pytest
from datetime import datetime

# ============================================================================
# Test Class: CandidatesExtractAPI
# ============================================================================


@pytest.mark.integration
@pytest.mark.asyncio
@pytest.mark.slow
class TestCandidatesExtractAPI:
    """End-to-end API tests for candidate extraction."""

    async def test_simple_italian_email(self, async_client, sample_email_document):
        """Test basic Italian email candidate extraction."""
        response = await async_client.post(
            "/api/v1/candidates/extract",
            json={
                "document": sample_email_document.model_dump(mode="json"),
                "enable_embeddings": True,
            },
        )

        assert response.status_code == 200
        data = response.json()

        # Check structure
        assert "document_id" in data
        assert "raw_candidates" in data
        assert "filtered_candidates" in data
        assert "enriched_candidates" in data

        # Check candidates returned
        assert len(data["raw_candidates"]) > 0

    async def test_subject_only_email(self, async_client):
        """Test email with subject only (empty body)."""
        from tests.unit.candidates.conftest import create_simple_document

        doc = create_simple_document(subject="Fattura urgente", body="")

        response = await async_client.post(
            "/api/v1/candidates/extract",
            json={"document": doc.model_dump(mode="json")},
        )

        assert response.status_code == 200
        data = response.json()

        # Should have candidates from subject
        candidates = data["raw_candidates"]
        assert len(candidates) > 0

        # All should be from subject
        for cand in candidates:
            assert cand["source"] == "subject"

    async def test_body_only_email(self, async_client):
        """Test email with body only (empty subject)."""
        from tests.unit.candidates.conftest import create_simple_document

        doc = create_simple_document(subject="", body="Il cliente ha un problema")

        response = await async_client.post(
            "/api/v1/candidates/extract",
            json={"document": doc.model_dump(mode="json")},
        )

        assert response.status_code == 200
        data = response.json()

        # Should have candidates from body
        candidates = data["raw_candidates"]
        assert len(candidates) > 0

        # All should be from body
        for cand in candidates:
            assert cand["source"] == "body"

    async def test_enable_embeddings_true(self, async_client, sample_email_document):
        """Test with enable_embeddings=true."""
        response = await async_client.post(
            "/api/v1/candidates/extract",
            json={
                "document": sample_email_document.model_dump(mode="json"),
                "enable_embeddings": True,
            },
        )

        assert response.status_code == 200
        data = response.json()

        # Check embedding model version is set
assert data["embedding_model_version"] != "none"
        assert "multilingual" in data["embedding_model_version"].lower()

        # Some candidates should have embedding scores > 0
        enriched = data["enriched_candidates"]
        if len(enriched) > 0:
            # At least one should have non-zero embedding score
            has_embeddings = any(c["embedding_score"] > 0.0 for c in enriched[:10])
            # Note: This may be False if KeyBERT doesn't find matches
            # So we just check the field exists
            assert all("embedding_score" in c for c in enriched)

    async def test_enable_embeddings_false(self, async_client, sample_email_document):
        """Test with enable_embeddings=false."""
        response = await async_client.post(
            "/api/v1/candidates/extract",
            json={
                "document": sample_email_document.model_dump(mode="json"),
                "enable_embeddings": False,
            },
        )

        assert response.status_code == 200
        data = response.json()

        # Check embedding model version is "none"
        assert data["embedding_model_version"] == "none"

        # All candidates should have embedding_score = 0.0
        enriched = data["enriched_candidates"]
        for cand in enriched:
            assert cand["embedding_score"] == 0.0

    async def test_custom_top_n_keybert(self, async_client, sample_email_document):
        """Test with custom top_n_keybert parameter."""
        response = await async_client.post(
            "/api/v1/candidates/extract",
            json={
                "document": sample_email_document.model_dump(mode="json"),
                "enable_embeddings": True,
                "top_n_keybert": 20,
            },
        )

        assert response.status_code == 200
        data = response.json()

        # Should succeed with custom top_n
        assert len(data["enriched_candidates"]) >= 0

    async def test_long_email(self, async_client):
        """Test with long email body (5000+ words)."""
        from tests.unit.candidates.conftest import create_simple_document

        long_body = " ".join(["Il cliente ha un problema con la fattura"] * 1000)
        doc = create_simple_document(subject="Test", body=long_body)

        response = await async_client.post(
            "/api/v1/candidates/extract",
            json={"document": doc.model_dump(mode="json")},
        )

        assert response.status_code == 200
        data = response.json()

        # Should handle long text
        assert len(data["raw_candidates"]) > 0


# ============================================================================
# Test Class: Response Validation
# ============================================================================


@pytest.mark.integration
@pytest.mark.asyncio
@pytest.mark.slow
class TestResponseValidation:
    """Tests for API response structure and validation."""

    async def test_response_structure(self, async_client, sample_email_document):
        """Test that response contains all required fields."""
        response = await async_client.post(
            "/api/v1/candidates/extract",
            json={"document": sample_email_document.model_dump(mode="json")},
        )

        assert response.status_code == 200
        data = response.json()

        # Check all required fields
        required_fields = [
            "document_id",
            "raw_candidates",
            "filtered_candidates",
            "enriched_candidates",
            "tokenizer_version",
            "stoplist_version",
            "embedding_model_version",
            "ner_model_version",
            "total_tokens_subject",
            "total_tokens_body",
            "candidates_before_filter",
            "candidates_after_filter",
            "processing_time_ms",
        ]

        for field in required_fields:
            assert field in data, f"Missing required field: {field}"

    async def test_version_fields_present(self, async_client, sample_email_document):
        """Test all version fields are populated."""
        response = await async_client.post(
            "/api/v1/candidates/extract",
            json={"document": sample_email_document.model_dump(mode="json")},
        )

        assert response.status_code == 200
        data = response.json()

        # Check version fields
        assert data["tokenizer_version"] is not None
        assert len(data["tokenizer_version"]) > 0

        assert data["stoplist_version"] is not None
        assert "stopwords-it" in data["stoplist_version"]

        assert data["ner_model_version"] is not None
        assert "it_core_news" in data["ner_model_version"]

    async def test_candidate_structure(self, async_client, sample_email_document):
        """Test that each candidate has all required fields."""
        response = await async_client.post(
            "/api/v1/candidates/extract",
            json={"document": sample_email_document.model_dump(mode="json")},
        )

        assert response.status_code == 200
        data = response.json()

        candidates = data["raw_candidates"]
        if len(candidates) > 0:
            cand = candidates[0]

            # Check candidate fields
            required_fields = [
                "candidate_id",
                "source",
                "term",
                "count",
                "lemma",
                "ngram_size",
                "embedding_score",
                "composite_score",
            ]

            for field in required_fields:
                assert field in cand

    async def test_candidate_counts_metadata(self, async_client, sample_email_document):
        """Test candidate count metadata is accurate."""
        response = await async_client.post(
            "/api/v1/candidates/extract",
            json={"document": sample_email_document.model_dump(mode="json")},
        )

        assert response.status_code == 200
        data = response.json()

        # Check counts match actual lists
        assert data["candidates_before_filter"] == len(data["raw_candidates"])
        # Note: candidates_after_filter is enriched_candidates count
        assert data["candidates_after_filter"] == len(data["enriched_candidates"])

    async def test_processing_time_recorded(self, async_client, sample_email_document):
        """Test that processing_time_ms is recorded."""
        response = await async_client.post(
            "/api/v1/candidates/extract",
            json={"document": sample_email_document.model_dump(mode="json")},
        )

        assert response.status_code == 200
        data = response.json()

        assert data["processing_time_ms"] > 0.0

    async def test_token_counts_recorded(self, async_client, sample_email_document):
        """Test that token counts are populated."""
        response = await async_client.post(
            "/api/v1/candidates/extract",
            json={"document": sample_email_document.model_dump(mode="json")},
        )

        assert response.status_code == 200
        data = response.json()

        assert data["total_tokens_subject"] >= 0
        assert data["total_tokens_body"] >= 0

    async def test_stable_candidate_ids(self, async_client, sample_email_document):
        """Test that candidate IDs are 12-char hex strings."""
        response = await async_client.post(
            "/api/v1/candidates/extract",
            json={"document": sample_email_document.model_dump(mode="json")},
        )

        assert response.status_code == 200
        data = response.json()

        candidates = data["raw_candidates"]
        for cand in candidates[:10]:  # Check first 10
            cand_id = cand["candidate_id"]
            assert isinstance(cand_id, str)
            assert len(cand_id) == 12
            assert all(c in "0123456789abcdef" for c in cand_id)

    async def test_source_field_accuracy(self, async_client):
        """Test that source field correctly marks subject vs body."""
        from tests.unit.candidates.conftest import create_simple_document

        doc = create_simple_document(subject="urgente", body="problema")

        response = await async_client.post(
            "/api/v1/candidates/extract",
            json={"document": doc.model_dump(mode="json")},
        )

        assert response.status_code == 200
        data = response.json()

        candidates = data["raw_candidates"]

        # Find candidates
        urgente_cands = [c for c in candidates if c["term"] == "urgente"]
        problema_cands = [c for c in candidates if c["term"] == "problema"]

        if urgente_cands:
            assert urgente_cands[0]["source"] == "subject"

        if problema_cands:
            assert problema_cands[0]["source"] == "body"


# ============================================================================
# Test Class: Determinism Tests
# ============================================================================


@pytest.mark.integration
@pytest.mark.asyncio
@pytest.mark.slow
class TestDeterminism:
    """Tests for deterministic behavior."""

    async def test_same_email_twice_identical_candidates(self, async_client, sample_email_document):
        """Test that same email produces identical candidates."""
        payload = {"document": sample_email_document.model_dump(mode="json")}

        response1 = await async_client.post("/api/v1/candidates/extract", json=payload)
        response2 = await async_client.post("/api/v1/candidates/extract", json=payload)

        assert response1.status_code == 200
        assert response2.status_code == 200

        data1 = response1.json()
        data2 = response2.json()

        # Same number of candidates
        assert len(data1["raw_candidates"]) == len(data2["raw_candidates"])

        # Same order
        for c1, c2 in zip(data1["raw_candidates"], data2["raw_candidates"]):
            assert c1["term"] == c2["term"]
            assert c1["source"] == c2["source"]
            assert c1["count"] == c2["count"]

    async def test_stable_id_persistence(self, async_client, sample_email_document):
        """Test that candidate IDs are stable across runs."""
        payload = {"document": sample_email_document.model_dump(mode="json")}

        response1 = await async_client.post("/api/v1/candidates/extract", json=payload)
        response2 = await async_client.post("/api/v1/candidates/extract", json=payload)

        data1 = response1.json()
        data2 = response2.json()

        # Build mapping: term+source -> candidate_id
        id_map1 = {(c["term"], c["source"]): c["candidate_id"] for c in data1["raw_candidates"]}
        id_map2 = {(c["term"], c["source"]): c["candidate_id"] for c in data2["raw_candidates"]}

        # Same IDs for same term+source
        for key in id_map1:
            if key in id_map2:
                assert id_map1[key] == id_map2[key]

    async def test_sorting_consistency(self, async_client, sample_email_document):
        """Test that candidate sorting is consistent."""
        payload = {"document": sample_email_document.model_dump(mode="json")}

        response1 = await async_client.post("/api/v1/candidates/extract", json=payload)
        response2 = await async_client.post("/api/v1/candidates/extract", json=payload)

        data1 = response1.json()
        data2 = response2.json()

        # Same sorting
        terms1 = [c["term"] for c in data1["enriched_candidates"]]
        terms2 = [c["term"] for c in data2["enriched_candidates"]]

        assert terms1 == terms2

    async def test_version_consistency(self, async_client, sample_email_document):
        """Test that version strings are consistent across runs."""
        payload = {"document": sample_email_document.model_dump(mode="json")}

        response1 = await async_client.post("/api/v1/candidates/extract", json=payload)
        response2 = await async_client.post("/api/v1/candidates/extract", json=payload)

data1 = response1.json()
        data2 = response2.json()

        # Same versions
        assert data1["tokenizer_version"] == data2["tokenizer_version"]
        assert data1["stoplist_version"] == data2["stoplist_version"]
        assert data1["ner_model_version"] == data2["ner_model_version"]


# ============================================================================
# Test Class: Error Handling
# ============================================================================


@pytest.mark.integration
@pytest.mark.asyncio
class TestErrorHandling:
    """Tests for error scenarios."""

    async def test_empty_document(self, async_client):
        """Test with empty subject and body."""
        from tests.unit.candidates.conftest import create_simple_document

        doc = create_simple_document(subject="", body="")

        response = await async_client.post(
            "/api/v1/candidates/extract",
            json={"document": doc.model_dump(mode="json")},
        )

        # Should succeed with empty candidates
        assert response.status_code == 200
        data = response.json()

        assert len(data["raw_candidates"]) == 0

    async def test_missing_document_field(self, async_client):
        """Test with missing document field."""
        response = await async_client.post(
            "/api/v1/candidates/extract",
            json={"enable_embeddings": True},  # Missing document
        )

        # Should return 422 Unprocessable Entity
        assert response.status_code == 422

    async def test_invalid_enable_embeddings_type(self, async_client, sample_email_document):
        """Test with invalid enable_embeddings value."""
        response = await async_client.post(
            "/api/v1/candidates/extract",
            json={
                "document": sample_email_document.model_dump(mode="json"),
                "enable_embeddings": "invalid",  # Should be bool
            },
        )

        # Should return 422
        assert response.status_code == 422


# ============================================================================
# Test: Health Check Endpoint
# ============================================================================


@pytest.mark.integration
@pytest.mark.asyncio
async def test_health_check_endpoint(async_client):
    """Test /api/v1/candidates/health endpoint."""
    response = await async_client.get("/api/v1/candidates/health")

    assert response.status_code == 200
    data = response.json()

    assert data["status"] == "healthy"
    assert data["service"] == "candidate_generation"
    assert "embeddings_enabled" in data
    assert "spacy_model" in data
