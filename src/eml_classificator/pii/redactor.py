"""
PII redaction module with audit logging.

This module handles Personal Identifiable Information (PII) redaction from email text
with complete audit trail for GDPR compliance.
"""

import re
from enum import Enum
from typing import Any, Dict, List, Tuple

from .hash_utils import hash_with_salt

# Version constant
PII_REDACTION_VERSION = "pii-redact-1.0.0"


class RedactionLevel(str, Enum):
    """PII redaction aggressiveness levels."""

    NONE = "none"
    MINIMAL = "minimal"  # Only emails in body
    STANDARD = "standard"  # Emails + phone numbers
    AGGRESSIVE = "aggressive"  # Emails + phones + person names (requires NER)


class PIIRedactor:
    """
    Handles PII redaction with audit logging.

    Detects and replaces PII (emails, phone numbers) with placeholder tokens.
    Each redaction is logged with position, hash, and version information for audit.
    """

    def __init__(self, level: RedactionLevel = RedactionLevel.STANDARD, salt: str = ""):
        """
        Initialize PII redactor.

        Args:
            level: Redaction aggressiveness level
            salt: Salt for hashing (for potential reversibility)
        """
        self.level = level
        self.salt = salt

        # Email pattern (RFC 5322 simplified)
        self._email_pattern = re.compile(
            r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b"
        )

        # Italian phone number patterns
        # Matches: +39 ..., 0XX ..., mobile formats, etc.
        self._phone_pattern = re.compile(
            r"\b(\+39\s?)?((0\d{1,4}[\s\-]?\d{4,8})|([3]\d{2}[\s\-]?\d{6,7}))\b"
        )

    def redact_text(self, text: str) -> Tuple[str, List[Dict[str, Any]]]:
        """
        Redact PII from text and return audit log.

        Args:
            text: Input text to redact

        Returns:
            Tuple of (redacted_text, redaction_log)

        redaction_log format:
        [
            {
                "type": "email",
                "position": [start, end],
                "original_hash": "sha256...",
                "placeholder": "[EMAIL_1]",
                "pii_redaction_version": "pii-redact-1.0.0",
                "pii_redaction_level": "standard"
            },
            ...
        ]
        """
        if self.level == RedactionLevel.NONE:
            return text, []

        redaction_log: List[Dict[str, Any]] = []
        redacted_text = text

        # Redact emails (for levels: minimal, standard, aggressive)
        if self.level in [RedactionLevel.MINIMAL, RedactionLevel.STANDARD, RedactionLevel.AGGRESSIVE]:
            redacted_text, email_log = self.redact_emails(redacted_text)
            redaction_log.extend(email_log)

        # Redact phone numbers (for levels: standard, aggressive)
        if self.level in [RedactionLevel.STANDARD, RedactionLevel.AGGRESSIVE]:
            redacted_text, phone_log = self.redact_phone_numbers(redacted_text)
            redaction_log.extend(phone_log)

        # TODO: Redact person names with NER (for level: aggressive)
        # This requires spaCy NER model and will be implemented in later phase

        return redacted_text, redaction_log

    def redact_emails(self, text: str) -> Tuple[str, List[Dict[str, Any]]]:
        """
        Redact email addresses from text.

        Args:
            text: Input text

        Returns:
            Tuple of (redacted_text, log_entries)
        """
        log_entries = []
        offset = 0  # Track offset due to replacements
        result = text

        for i, match in enumerate(self._email_pattern.finditer(text), start=1):
            email = match.group(0)
            start, end = match.span()

            # Adjust positions for accumulated offset
            adjusted_start = start + offset
            adjusted_end = end + offset

            # Create placeholder
            placeholder = f"[EMAIL_{i}]"

            # Hash the original email
            email_hash = self._hash_pii(email)

            # Log this redaction WITH version and level as requested
            log_entries.append(
                {
                    "type": "email",
                    "position": [start, end],  # Original position before offset
                    "original_hash": email_hash,
                    "placeholder": placeholder,
                    "pii_redaction_version": PII_REDACTION_VERSION,
                    "pii_redaction_level": self.level.value,
                }
            )

            # Replace in result text
            result = result[:adjusted_start] + placeholder + result[adjusted_end:]

            # Update offset
            offset += len(placeholder) - len(email)

        return result, log_entries

    def redact_phone_numbers(self, text: str) -> Tuple[str, List[Dict[str, Any]]]:
        """
        Redact phone numbers from text.

        Args:
            text: Input text

        Returns:
            Tuple of (redacted_text, log_entries)
        """
        log_entries = []
        offset = 0
        result = text

        for i, match in enumerate(self._phone_pattern.finditer(text), start=1):
            phone = match.group(0)
            start, end = match.span()

            # Adjust positions
            adjusted_start = start + offset
            adjusted_end = end + offset

            # Create placeholder
            placeholder = f"[PHONE_{i}]"

            # Hash the original phone
            phone_hash = self._hash_pii(phone)

            # Log this redaction WITH version and level
            log_entries.append(
                {
                    "type": "phone",
                    "position": [start, end],
                    "original_hash": phone_hash,
                    "placeholder": placeholder,
                    "pii_redaction_version": PII_REDACTION_VERSION,
                    "pii_redaction_level": self.level.value,
                }
            )

            # Replace
            result = result[:adjusted_start] + placeholder + result[adjusted_end:]

            # Update offset
            offset += len(placeholder) - len(phone)

        return result, log_entries

    def _hash_pii(self, pii_value: str) -> str:
        """
        Hash PII value for potential reversibility.

        Args:
            pii_value: The PII value to hash

        Returns:
            SHA-256 hash with salt
        """
        return hash_with_salt(pii_value, self.salt)
