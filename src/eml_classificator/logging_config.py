"""
Structured logging configuration using structlog.

This module sets up structlog for JSON-based structured logging with context binding.
"""

import logging
import structlog

from .config import settings


def setup_logging() -> None:
    """
    Configure structlog for structured JSON logging.

    Sets up processors for:
    - Context variable merging
    - Log level addition
    - Exception info rendering
    - Timestamp addition
    - JSON or console rendering based on settings
    """
    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.processors.add_log_level,
            structlog.processors.StackInfoRenderer(),
            structlog.dev.set_exc_info,
            structlog.processors.TimeStamper(fmt="iso"),
            (
                structlog.processors.JSONRenderer()
                if settings.log_json
                else structlog.dev.ConsoleRenderer()
            ),
        ],
        wrapper_class=structlog.make_filtering_bound_logger(
            logging.getLevelName(settings.log_level)
        ),
        context_class=dict,
        logger_factory=structlog.PrintLoggerFactory(),
        cache_logger_on_first_use=True,
    )


def get_logger(name: str) -> structlog.BoundLogger:
    """
    Get a logger instance with the given name.

    Args:
        name: Logger name (usually __name__)

    Returns:
        structlog BoundLogger instance
    """
    return structlog.get_logger(name)
