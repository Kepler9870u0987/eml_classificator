"""
Hashing utilities for PII redaction.

This module provides cryptographic hashing functions for PII values
to enable potential reversibility while maintaining security.
"""

import hashlib
import secrets


def hash_with_salt(value: str, salt: str) -> str:
    """
    Create salted hash of value.

    Uses SHA-256 with salt for secure but reversible (with salt) hashing.

    Args:
        value: Value to hash
        salt: Salt string

    Returns:
        Hex-encoded SHA-256 hash with prefix
    """
    combined = f"{salt}{value}".encode("utf-8")
    hash_obj = hashlib.sha256(combined)
    return f"sha256-{hash_obj.hexdigest()}"


def create_salt(length: int = 32) -> str:
    """
    Generate random salt for PII hashing.

    Args:
        length: Length of salt in bytes

    Returns:
        Random salt string (hex-encoded)
    """
    return secrets.token_hex(length)


def verify_hash(value: str, hash_str: str, salt: str) -> bool:
    """
    Verify if a value matches a hash.

    Args:
        value: Value to verify
        hash_str: Hash string to check against (with "sha256-" prefix)
        salt: Salt used for original hashing

    Returns:
        True if value matches hash, False otherwise
    """
    computed_hash = hash_with_salt(value, salt)
    return computed_hash == hash_str
