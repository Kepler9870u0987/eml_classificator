"""
Application configuration management.

This module handles configuration from environment variables using Pydantic Settings.
"""

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """
    Application configuration from environment variables.

    All settings can be overridden via environment variables with the same name.
    """

    # API settings
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    api_workers: int = 4

    # Logging
    log_level: str = "INFO"
    log_json: bool = True

    # PII redaction
    pii_salt: str = "default-salt-change-in-production"
    default_pii_level: str = "standard"

    # Processing limits
    max_email_size_mb: int = 25
    max_attachments: int = 50

    # Monitoring
    enable_metrics: bool = True
    metrics_port: int = 9090

    model_config = {
        "env_file": ".env",
        "env_file_encoding": "utf-8",
    }


# Global settings instance
settings = Settings()
