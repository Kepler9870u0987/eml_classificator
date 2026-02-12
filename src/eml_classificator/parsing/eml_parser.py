"""
Email parser for .eml files (RFC5322/MIME format).

This module handles parsing of email files using Python's standard library email module.
Extracts headers, body content, and attachment metadata deterministically.
"""

import hashlib
from datetime import datetime
from email import message_from_bytes
from email.message import Message
from email.utils import parsedate_to_datetime
from typing import List, Optional, Tuple

import charset_normalizer

from ..models.email_document import AttachmentMetadata, EmailHeaders


def parse_eml_bytes(eml_bytes: bytes) -> Message:
    """
    Parse .eml bytes into email.Message object.

    Args:
        eml_bytes: Raw .eml file bytes

    Returns:
        Parsed email.Message object

    Raises:
        ValueError: If bytes are not valid RFC5322 format
    """
    try:
        msg = message_from_bytes(eml_bytes)
        return msg
    except Exception as e:
        raise ValueError(f"Failed to parse .eml file: {str(e)}") from e


def parse_eml_file(eml_path: str) -> Message:
    """
    Parse .eml file into email.Message object.

    Args:
        eml_path: Path to .eml file

    Returns:
        Parsed email.Message object

    Raises:
        FileNotFoundError: If file doesn't exist
        ValueError: If file is not valid RFC5322 format
    """
    with open(eml_path, "rb") as f:
        eml_bytes = f.read()
    return parse_eml_bytes(eml_bytes)


def extract_headers(msg: Message) -> EmailHeaders:
    """
    Extract and parse email headers into structured format.

    Args:
        msg: Parsed email.Message object

    Returns:
        EmailHeaders with parsed header information
    """
    # Parse date header
    date_str = msg.get("Date")
    date = None
    if date_str:
        try:
            date = parsedate_to_datetime(date_str)
        except (TypeError, ValueError):
            pass  # Keep as None if parsing fails

    # Parse To addresses
    to_addresses = []
    to_header = msg.get("To", "")
    if to_header:
        # Simple split for now - could be enhanced with email.utils.getaddresses
        to_addresses = [addr.strip() for addr in to_header.split(",") if addr.strip()]

    # Parse CC addresses
    cc_addresses = []
    cc_header = msg.get("Cc", "")
    if cc_header:
        cc_addresses = [addr.strip() for addr in cc_header.split(",") if addr.strip()]

    # Parse References (for threading)
    references = []
    ref_header = msg.get("References", "")
    if ref_header:
        references = [ref.strip() for ref in ref_header.split() if ref.strip()]

    return EmailHeaders(
        from_address=msg.get("From", ""),
        to_addresses=to_addresses,
        cc_addresses=cc_addresses,
        subject=msg.get("Subject", ""),
        date=date,
        message_id=msg.get("Message-ID"),
        in_reply_to=msg.get("In-Reply-To"),
        references=references,
        extra_headers={},  # Could be extended to capture other headers
    )


def extract_body(msg: Message) -> Tuple[str, Optional[str]]:
    """
    Extract plain text and HTML body from email message.
    Handles multipart messages and walking the MIME tree.

    Args:
        msg: Parsed email.Message object

    Returns:
        Tuple of (plain_text_body, html_body)
        If no plain text found, returns empty string
        If no HTML found, returns None
    """
    plain_text = ""
    html_body = None

    if msg.is_multipart():
        # Walk through all parts of multipart message
        for part in msg.walk():
            content_type = part.get_content_type()
            content_disposition = str(part.get("Content-Disposition", ""))

            # Skip attachments
            if "attachment" in content_disposition:
                continue

            try:
                payload = part.get_payload(decode=True)
                if payload is None:
                    continue

                # Decode with charset
                charset = part.get_content_charset() or "utf-8"
                try:
                    text = payload.decode(charset, errors="replace")
                except (UnicodeDecodeError, LookupError):
                    # Fallback to charset detection
                    detected = charset_normalizer.from_bytes(payload).best()
                    text = str(detected) if detected else payload.decode("utf-8", errors="replace")

                if content_type == "text/plain":
                    plain_text += text
                elif content_type == "text/html" and html_body is None:
                    html_body = text

            except Exception:
                # Skip parts that can't be decoded
                continue

    else:
        # Non-multipart message
        content_type = msg.get_content_type()
        try:
            payload = msg.get_payload(decode=True)
            if payload:
                charset = msg.get_content_charset() or "utf-8"
                try:
                    text = payload.decode(charset, errors="replace")
                except (UnicodeDecodeError, LookupError):
                    detected = charset_normalizer.from_bytes(payload).best()
                    text = str(detected) if detected else payload.decode("utf-8", errors="replace")

                if content_type == "text/plain":
                    plain_text = text
                elif content_type == "text/html":
                    html_body = text
        except Exception:
            pass

    return plain_text, html_body


def extract_attachments(msg: Message) -> List[AttachmentMetadata]:
    """
    Extract attachment metadata (without content) from email.

    Args:
        msg: Parsed email.Message object

    Returns:
        List of AttachmentMetadata objects
    """
    attachments = []

    if msg.is_multipart():
        for part in msg.walk():
            content_disposition = str(part.get("Content-Disposition", ""))

            if "attachment" in content_disposition or part.get_filename():
                filename = part.get_filename()
                content_type = part.get_content_type()
                content_id = part.get("Content-ID")

                # Get size
                payload = part.get_payload(decode=False)
                size_bytes = len(str(payload).encode()) if payload else 0

                attachments.append(
                    AttachmentMetadata(
                        filename=filename,
                        content_type=content_type,
                        size_bytes=size_bytes,
                        content_id=content_id,
                    )
                )

    return attachments


def compute_document_id(eml_bytes: bytes, timestamp: datetime) -> str:
    """
    Compute deterministic document ID from email content.
    Uses SHA-256 hash of raw bytes + timestamp for uniqueness.

    Args:
        eml_bytes: Raw .eml file bytes
        timestamp: Ingestion timestamp

    Returns:
        Hex-encoded SHA-256 hash prefixed with 'eml-'
    """
    # Combine email content with timestamp for deterministic but unique ID
    content = eml_bytes + timestamp.isoformat().encode()
    hash_obj = hashlib.sha256(content)
    return f"eml-{hash_obj.hexdigest()[:32]}"


def detect_encoding(msg: Message) -> str:
    """
    Detect character encoding using charset-normalizer.
    Falls back to charset declared in headers.

    Args:
        msg: Parsed email.Message object

    Returns:
        Detected encoding name (e.g., 'utf-8', 'iso-8859-1')
    """
    # Try header charset first
    charset = msg.get_content_charset()
    if charset:
        return charset

    # Try to detect from payload
    payload = msg.get_payload(decode=True)
    if payload:
        detected = charset_normalizer.from_bytes(payload).best()
        if detected:
            return detected.encoding

    return "utf-8"  # Default fallback
