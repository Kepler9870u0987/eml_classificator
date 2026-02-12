"""
Global pytest fixtures and configuration for test suite.

This module provides reusable fixtures for:
- HTTP clients
- Mock settings/configuration
- Sample email data
- Temporary files
"""

import os
import tempfile
from datetime import datetime
from typing import AsyncGenerator, Generator

import pytest
from httpx import ASGITransport, AsyncClient

from eml_classificator.api.app import app
from eml_classificator.config import Settings
from .fixtures.emails import SAMPLE_EMAILS


@pytest.fixture
async def async_client() -> AsyncGenerator[AsyncClient, None]:
    """
    Create async HTTP client for testing FastAPI endpoints.

    Yields:
        AsyncClient instance
    """
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        yield client


@pytest.fixture
def mock_settings() -> Settings:
    """
    Create mock settings for testing with safe defaults.

    Returns:
        Settings instance with test configuration
    """
    return Settings(
        pii_salt="test-salt-not-for-production",
        max_email_size_mb=25,
        max_attachments=50,
        log_level="INFO",
        log_json=False,  # Easier to read in tests
    )


@pytest.fixture
def sample_eml_bytes() -> bytes:
    """
    Get simple plain text email bytes for basic tests.

    Returns:
        bytes of a simple .eml file
    """
    return SAMPLE_EMAILS["simple_plain_text"]


@pytest.fixture
def italian_signature_eml() -> bytes:
    """
    Get email with Italian signature for signature removal tests.

    Returns:
        bytes of email with "Cordiali saluti" signature
    """
    return SAMPLE_EMAILS["italian_signature"]


@pytest.fixture
def pii_email_bytes() -> bytes:
    """
    Get email containing PII (emails, phone numbers) for redaction tests.

    Returns:
        bytes of email with Italian PII
    """
    return SAMPLE_EMAILS["pii_email"]


@pytest.fixture
def multipart_html_eml() -> bytes:
    """
    Get multipart email with both HTML and plain text.

    Returns:
        bytes of multipart/alternative email
    """
    return SAMPLE_EMAILS["multipart_html"]


@pytest.fixture
def html_only_eml() -> bytes:
    """
    Get email with only HTML content (no plain text).

    Returns:
        bytes of HTML-only email
    """
    return SAMPLE_EMAILS["html_only"]


@pytest.fixture
def attachment_eml() -> bytes:
    """
    Get email with single attachment.

    Returns:
        bytes of email with PDF attachment
    """
    return SAMPLE_EMAILS["attachment"]


@pytest.fixture
def reply_chain_eml() -> bytes:
    """
    Get email with quoted reply chain.

    Returns:
        bytes of email with > quoted text
    """
    return SAMPLE_EMAILS["reply_chain"]


@pytest.fixture
def malformed_eml() -> bytes:
    """
    Get malformed email for error handling tests.

    Returns:
        bytes of invalid RFC5322 data
    """
    return SAMPLE_EMAILS["malformed"]


@pytest.fixture
def tmp_eml_file(tmp_path) -> Generator[str, None, None]:
    """
    Create temporary .eml file for file-based tests.

    Args:
        tmp_path: pytest's temporary directory fixture

    Yields:
        Path to temporary .eml file
    """
    eml_path = tmp_path / "test_email.eml"
    eml_path.write_bytes(SAMPLE_EMAILS["simple_plain_text"])
    yield str(eml_path)
    # Cleanup is automatic with tmp_path


@pytest.fixture
def fixed_timestamp() -> datetime:
    """
    Provide fixed timestamp for deterministic testing.

    Returns:
        Fixed datetime for reproducible tests
    """
    return datetime(2026, 2, 12, 10, 30, 0)


@pytest.fixture(autouse=True)
def reset_env_vars():
    """
    Reset environment variables before each test.

    This prevents test pollution from env var changes.
    """
    original_env = os.environ.copy()
    yield
    os.environ.clear()
    os.environ.update(original_env)


@pytest.fixture
def test_pii_salt() -> str:
    """
    Provide test PII salt for hashing tests.

    Returns:
        Test salt string (NOT for production)
    """
    return "test-salt-for-hashing-pii-data"


# Configuration for pytest-asyncio
def pytest_configure(config):
    """
    Configure pytest with custom markers and settings.
    """
    config.addinivalue_line(
        "markers", "unit: Unit tests (fast, no external dependencies)"
    )
    config.addinivalue_line(
        "markers", "integration: Integration tests (API, end-to-end)"
    )
    config.addinivalue_line(
        "markers", "slow: Slow tests that may take longer to run"
    )
