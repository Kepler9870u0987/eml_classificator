"""
MIME utility functions for handling multipart email messages.

This module provides helper functions for working with MIME messages.
"""

from email.message import Message
from typing import Iterator

import charset_normalizer


def walk_message_parts(msg: Message) -> Iterator[Message]:
    """
    Walk through all parts of a multipart message.

    Args:
        msg: Email message (potentially multipart)

    Yields:
        Individual message parts
    """
    if msg.is_multipart():
        for part in msg.walk():
            yield part
    else:
        yield msg


def decode_payload(part: Message) -> str:
    """
    Decode message part payload handling various encodings.

    Args:
        part: Message part to decode

    Returns:
        Decoded string content

    Raises:
        UnicodeDecodeError: If decoding fails with all strategies
    """
    payload = part.get_payload(decode=True)
    if payload is None:
        return ""

    # Try declared charset first
    charset = part.get_content_charset()
    if charset:
        try:
            return payload.decode(charset)
        except (UnicodeDecodeError, LookupError):
            pass

    # Try charset detection
    detected = charset_normalizer.from_bytes(payload).best()
    if detected:
        return str(detected)

    # Final fallback
    return payload.decode("utf-8", errors="replace")


def is_attachment(part: Message) -> bool:
    """
    Determine if message part is an attachment.

    Args:
        part: Message part to check

    Returns:
        True if part is an attachment, False otherwise
    """
    content_disposition = str(part.get("Content-Disposition", ""))
    return "attachment" in content_disposition or part.get_filename() is not None


def get_text_parts(msg: Message) -> list[str]:
    """
    Extract all text/plain parts from a message.

    Args:
        msg: Email message

    Returns:
        List of decoded text parts
    """
    texts = []

    for part in walk_message_parts(msg):
        if part.get_content_type() == "text/plain" and not is_attachment(part):
            try:
                text = decode_payload(part)
                texts.append(text)
            except Exception:
                continue

    return texts


def get_html_parts(msg: Message) -> list[str]:
    """
    Extract all text/html parts from a message.

    Args:
        msg: Email message

    Returns:
        List of decoded HTML parts
    """
    htmls = []

    for part in walk_message_parts(msg):
        if part.get_content_type() == "text/html" and not is_attachment(part):
            try:
                html = decode_payload(part)
                htmls.append(html)
            except Exception:
                continue

    return htmls
