# Email parsing module

from .eml_parser import (
    compute_document_id,
    detect_encoding,
    extract_attachments,
    extract_body,
    extract_headers,
    parse_eml_bytes,
    parse_eml_file,
)
from .mime_utils import (
    decode_payload,
    get_html_parts,
    get_text_parts,
    is_attachment,
    walk_message_parts,
)

__all__ = [
    "parse_eml_bytes",
    "parse_eml_file",
    "extract_headers",
    "extract_body",
    "extract_attachments",
    "compute_document_id",
    "detect_encoding",
    "walk_message_parts",
    "decode_payload",
    "is_attachment",
    "get_text_parts",
    "get_html_parts",
]
