"""
Text canonicalization and normalization module.

This module handles deterministic text normalization including whitespace handling,
signature removal (with Italian patterns), and reply chain detection.
"""

import re
from typing import List, Tuple

from ..models.canonicalization import RemovedSection

# Version constant for audit trail
CANONICALIZATION_VERSION = "canon-1.2.0"

# Pattern definitions with types for intelligent merging
# Each pattern is a tuple of (pattern, type, description)
SIGNATURE_PATTERNS = [
    (r"^--+\s*$", "signature", "Standard signature delimiter"),
    (r"^___+\s*$", "signature", "Alternative signature delimiter"),
    (
        r"^\s*(Cordiali saluti|Distinti saluti|Saluti|Best regards|Regards|Grazie)",
        "signature",
        "Italian/English closings",
    ),
    (
        r"^Sent from my (iPhone|iPad|Android|Samsung|Huawei)",
        "signature",
        "Mobile device signatures",
    ),
    (r"^Inviato da", "signature", "Italian 'sent from'"),
]

# Disclaimer patterns (from spec)
DISCLAIMER_PATTERNS = [
    (
        r"(?is)\n_{10,}.*$",
        "disclaimer",
        "Long underscore disclaimers",
    ),  # NEW from spec
]

# Reply chain patterns with types
REPLY_PATTERNS = [
    (r"^>+\s", "quote", "Quoted lines"),
    (r"^\|+\s", "quote", "Alternative quote style"),
    (r"^On .+ wrote:", "reply_header", "English reply header"),
    (r"^Il .+ ha scritto:", "reply_header", "Italian reply header"),
    (
        r"(?is)\nIl giorno .+ ha scritto:.+",
        "reply_header",
        "Italian reply with context",
    ),  # NEW from spec
    (
        r"^-+\s*Original Message\s*-+",
        "reply_header",
        "English forwarded message",
    ),
    (
        r"^-+\s*Messaggio originale\s*-+",
        "reply_header",
        "Italian forwarded message",
    ),
]


def canonicalize_text(
    text: str, remove_signatures: bool = True, keep_audit_info: bool = True
) -> Tuple[str, List[RemovedSection]]:
    """
    Canonicalize email text for deterministic processing.

    Steps:
    1. Normalize line endings to \\n
    2. Remove excessive whitespace
    3. Remove email signatures (if enabled)
    4. Remove quote/reply chains
    5. Remove disclaimers
    6. Convert to lowercase
    7. Strip leading/trailing whitespace

    Args:
        text: Raw email text
        remove_signatures: Whether to attempt signature removal
        keep_audit_info: Whether to track removed sections (default True for GDPR compliance)

    Returns:
        Tuple of (canonicalized_text, list_of_removed_sections)
        If keep_audit_info=False, returns empty list for removed_sections
    """
    if not text:
        return "", []

    removed_sections: List[RemovedSection] = []

    # 1. Normalize line endings BEFORE tracking positions
    text = text.replace("\r\n", "\n").replace("\r", "\n")

    # 2. Normalize whitespace
    text = normalize_whitespace(text)

    # 3. Remove disclaimers first (typically at end)
    text, disclaimer_sections = remove_disclaimers(
        text, keep_audit_info=keep_audit_info
    )
    if keep_audit_info:
        removed_sections.extend(disclaimer_sections)

    # 4. Remove signatures
    if remove_signatures:
        text, signature_sections = remove_email_signatures(
            text, keep_audit_info=keep_audit_info
        )
        if keep_audit_info:
            removed_sections.extend(signature_sections)

    # 5. Remove reply chains
    text, reply_sections = remove_reply_chains(text, keep_audit_info=keep_audit_info)
    if keep_audit_info:
        removed_sections.extend(reply_sections)

    # 6. Convert to lowercase (after tracking - positions still map back)
    text = text.lower()

    # 7. Strip whitespace
    text = text.strip()

    return text, removed_sections


def normalize_whitespace(text: str) -> str:
    """
    Normalize whitespace in text.

    - Convert multiple spaces to single space (except at line starts)
    - Remove trailing whitespace from lines
    - Normalize line endings to \\n
    - Collapse multiple blank lines to max 2

    Args:
        text: Input text

    Returns:
        Text with normalized whitespace
    """
    # Normalize line endings
    text = text.replace("\r\n", "\n").replace("\r", "\n")

    # Process line by line
    lines = []
    for line in text.split("\n"):
        # Remove trailing whitespace
        line = line.rstrip()
        if not line:
            lines.append(line)
            continue
        # Preserve leading indentation while collapsing internal whitespace
        match = re.match(r"^([ \t]*)(.*)$", line)
        leading = match.group(1)
        content = re.sub(r"[ \t]+", " ", match.group(2))
        lines.append(f"{leading}{content}")

    # Join lines and collapse multiple blank lines
    text = "\n".join(lines)
    text = re.sub(r"\n{3,}", "\n\n", text)

    return text


def remove_urls(text: str, replacement: str = "[URL]") -> str:
    """
    Remove or replace URLs in text.

    Args:
        text: Input text
        replacement: What to replace URLs with

    Returns:
        Text with URLs replaced
    """
    url_pattern = r"https?://[^\s]+"
    return re.sub(url_pattern, replacement, text)


def remove_email_signatures(text: str) -> Tuple[str, bool]:
    """
    Detect and remove email signatures.

    Common patterns:
    - Lines starting with "--" or "___"
    - "Sent from my iPhone" type footers
    - "Cordiali saluti" / "Distinti saluti" / "Best regards" followed by content
    - Italian: "Cordiali saluti", "Distinti saluti", "Saluti"

    Args:
        text: Email body text

    Returns:
        Tuple of (text_without_signature, signature_was_detected)
    """
    lines = text.split("\n")
    signature_detected = False
    result_lines = []

    for i, line in enumerate(lines):
        is_signature = False

        # Check against signature patterns
        for pattern in SIGNATURE_PATTERNS:
            if re.match(pattern, line.strip(), re.IGNORECASE):
                is_signature = True
                signature_detected = True
                break

        if is_signature:
            # Found signature start - discard this and remaining lines
            break

        result_lines.append(line)

    return "\n".join(result_lines), signature_detected


def remove_reply_chains(text: str) -> Tuple[str, bool]:
    """
    Detect and remove quoted reply chains.

    Patterns:
    - Lines starting with ">" or "|"
    - "On [date], [person] wrote:" blocks
    - "Il [date], [person] ha scritto:" blocks
    - "From: ... Sent: ..." forwarded headers
    - "---Original Message---" blocks

    Args:
        text: Email body text

    Returns:
        Tuple of (text_without_replies, replies_were_detected)
    """
    lines = text.split("\n")
    replies_detected = False
    result_lines = []
    in_quoted_block = False

    for line in lines:
        is_quoted = False

        # Check if line is quoted
        for pattern in REPLY_PATTERNS:
            if re.match(pattern, line.strip(), re.IGNORECASE):
                is_quoted = True
                replies_detected = True
                in_quoted_block = True
                break

        # Also consider lines starting with > as quoted
        if line.strip().startswith(">"):
            is_quoted = True
            replies_detected = True
            in_quoted_block = True

        if not is_quoted:
            # Check if we should exit quoted block (non-empty line after quotes)
            if in_quoted_block and line.strip():
                # This could be content after quotes, so reset
                in_quoted_block = False
            result_lines.append(line)

    return "\n".join(result_lines), replies_detected


def remove_excessive_newlines(text: str, max_consecutive: int = 2) -> str:
    """
    Remove excessive consecutive newlines.

    Args:
        text: Input text
        max_consecutive: Maximum number of consecutive newlines to keep

    Returns:
        Text with excessive newlines removed
    """
    pattern = r"\n{" + str(max_consecutive + 1) + r",}"
    replacement = "\n" * max_consecutive
    return re.sub(pattern, replacement, text)
