"""
Customer status determination via mock CRM lookup.

Implements deterministic customer status classification as specified
in brainstorming v2 section 4.1.

IMPORTANT: Customer status MUST be deterministic, not LLM-based.
This module overrides any LLM guesses with actual CRM lookup results.

Mock implementation supports:
- Exact email match (known customers file)
- Domain pattern matching (known domains list)
- Text signal detection (fallback for unknowns)
"""

import re
from typing import Optional, Set, List
from dataclasses import dataclass
from pathlib import Path

import structlog

from eml_classificator.classification.schemas import CustomerStatus, CustomerStatusValue
from eml_classificator.config import settings


logger = structlog.get_logger(__name__)


# ============================================================================
# MOCK CRM DATABASE
# ============================================================================

@dataclass
class MockCRMDatabase:
    """
    Mock CRM database with known customers.

    In production, replace with actual CRM API integration.
    """
    known_emails: Set[str]  # Exact email addresses
    known_domains: Set[str]  # Domain patterns (@company.com)

    @classmethod
    def load_from_config(cls) -> "MockCRMDatabase":
        """Load mock CRM from configuration."""
        # Parse known domains from config
        domains_str = settings.crm_mock_known_domains
        known_domains = {d.strip() for d in domains_str.split(",") if d.strip()}

        # Load known emails from file (if exists)
        known_emails = set()
        emails_file = Path(settings.crm_mock_known_emails_file)

        if emails_file.exists():
            try:
                with open(emails_file, 'r', encoding='utf-8') as f:
                    for line in f:
                        email = line.strip().lower()
                        if email and '@' in email:
                            known_emails.add(email)

                logger.info(
                    "mock_crm_loaded_from_file",
                    emails_count=len(known_emails),
                    file=str(emails_file)
                )
            except Exception as e:
                logger.error(
                    "mock_crm_file_load_failed",
                    file=str(emails_file),
                    error=str(e)
                )

        logger.info(
            "mock_crm_initialized",
            known_emails_count=len(known_emails),
            known_domains=list(known_domains)
        )

        return cls(
            known_emails=known_emails,
            known_domains=known_domains
        )

    def lookup_email(self, email: str) -> tuple[str, float, str]:
        """
        Lookup email in mock CRM.

        Args:
            email: Customer email address

        Returns:
            (match_type, confidence, source) tuple:
            - match_type: "exact" | "domain" | "none"
            - confidence: float 0.0-1.0
            - source: description of match
        """
        email_lower = email.lower().strip()

        # Exact email match
        if email_lower in self.known_emails:
            return ("exact", 1.0, "crm_exact_match")

        # Domain match
        if '@' in email_lower:
            domain = email_lower.split('@')[1]
            if domain in self.known_domains:
                return ("domain", 0.7, "crm_domain_match")

        # No match
        return ("none", 0.0, "no_crm_match")


# ============================================================================
# TEXT SIGNAL DETECTION
# ============================================================================

# Italian text signals indicating existing customer
EXISTING_CUSTOMER_SIGNALS = [
    r"(?i)\b(ho già|già|abbiamo già)\b.*\b(contratto|account|servizio|abbonamento)\b",
    r"(?i)\b(cliente|abbonato)\b.*\b(dal|da|esistente)\b",
    r"(?i)\bvostro cliente\b",
    r"(?i)\bmio.*\b(contratto|account|codice cliente)\b",
    r"(?i)\bnumero cliente\b",
    r"(?i)\bcodice fiscale.*\bregistrato\b",
]

# Italian text signals indicating new customer
NEW_CUSTOMER_SIGNALS = [
    r"(?i)\bprimo contatto\b",
    r"(?i)\bvorrei diventare cliente\b",
    r"(?i)\bprima volta\b.*\b(scrivo|contatto)\b",
    r"(?i)\bnon sono ancora cliente\b",
]


def detect_text_signals(text: str) -> Optional[tuple[str, List[str]]]:
    """
    Detect customer status signals in email text.

    Args:
        text: Email body canonicalized

    Returns:
        (status, matched_signals) or None if no clear signal found
        - status: "existing" | "new"
        - matched_signals: List of matched signal patterns
    """
    text_lower = text.lower()

    # Check existing customer signals
    existing_matches = []
    for pattern in EXISTING_CUSTOMER_SIGNALS:
        if re.search(pattern, text):
            existing_matches.append(f"existing_signal: {pattern[:50]}")

    # Check new customer signals
    new_matches = []
    for pattern in NEW_CUSTOMER_SIGNALS:
        if re.search(pattern, text):
            new_matches.append(f"new_signal: {pattern[:50]}")

    # Decide based on signal strength
    if len(existing_matches) > len(new_matches):
        return ("existing", existing_matches)
    elif len(new_matches) > len(existing_matches):
        return ("new", new_matches)
    else:
        return None


# ============================================================================
# CUSTOMER STATUS COMPUTATION
# ============================================================================

def compute_customer_status(
    from_email: str,
    text_body: str = "",
    crm_db: Optional[MockCRMDatabase] = None
) -> CustomerStatus:
    """
    Compute customer status deterministically.

    Decision hierarchy:
    1. CRM exact match → existing (confidence=1.0)
    2. CRM domain match → existing (confidence=0.7)
    3. Text signals (existing) → existing (confidence=0.5)
    4. Text signals (new) → new (confidence=0.6)
    5. No CRM, no signals → new (confidence=0.8, assume new unless proven otherwise)

    Args:
        from_email: Customer email address
        text_body: Email body canonical text
        crm_db: Mock CRM database (default: load from config)

    Returns:
        CustomerStatus with value, confidence, and source
    """
    if crm_db is None:
        if not settings.enable_crm_lookup:
            # CRM lookup disabled, return unknown
            return CustomerStatus(
                value=CustomerStatusValue.UNKNOWN,
                confidence=0.5,
                source="crm_disabled"
            )

        crm_db = MockCRMDatabase.load_from_config()

    logger.debug("computing_customer_status", from_email=from_email)

    # Step 1: CRM lookup
    match_type, crm_confidence, crm_source = crm_db.lookup_email(from_email)

    if match_type == "exact":
        return CustomerStatus(
            value=CustomerStatusValue.EXISTING,
            confidence=1.0,
            source=crm_source
        )

    if match_type == "domain":
        return CustomerStatus(
            value=CustomerStatusValue.EXISTING,
            confidence=0.7,
            source=crm_source
        )

    # Step 2: Text signal detection
    signal_result = detect_text_signals(text_body)

    if signal_result:
        status_value, matched_signals = signal_result

        logger.debug(
            "customer_status_text_signals_detected",
            status=status_value,
            signals_count=len(matched_signals)
        )

        if status_value == "existing":
            return CustomerStatus(
                value=CustomerStatusValue.EXISTING,
                confidence=0.5,
                source="text_signal"
            )
        else:  # "new"
            return CustomerStatus(
                value=CustomerStatusValue.NEW,
                confidence=0.6,
                source="text_signal"
            )

    # Step 3: Default to new (no CRM match, no text signals)
    # Rationale: better to treat unknown as new than incorrectly assume existing
    return CustomerStatus(
        value=CustomerStatusValue.NEW,
        confidence=0.8,
        source="no_crm_no_signal"
    )


# ============================================================================
# OVERRIDE LLM CUSTOMER STATUS
# ============================================================================

def override_customer_status_with_crm(
    llm_customer_status: CustomerStatus,
    from_email: str,
    text_body: str = "",
    crm_db: Optional[MockCRMDatabase] = None
) -> CustomerStatus:
    """
    Override LLM-guessed customer status with deterministic CRM lookup.

    This is the main entry point for post-processing classification results.

    Args:
        llm_customer_status: Customer status guessed by LLM (ignored)
        from_email: Customer email address
        text_body: Email body for text signal detection
        crm_db: Mock CRM database

    Returns:
        Deterministic CustomerStatus from CRM lookup
    """
    # Log LLM guess for comparison
    logger.debug(
        "overriding_llm_customer_status",
        llm_value=llm_customer_status.value,
        llm_confidence=llm_customer_status.confidence,
        llm_source=getattr(llm_customer_status, 'source', 'llm_guess')
    )

    # Compute deterministic status
    crm_status = compute_customer_status(
        from_email=from_email,
        text_body=text_body,
        crm_db=crm_db
    )

    # Log override if different
    if crm_status.value != llm_customer_status.value:
        logger.info(
            "customer_status_overridden",
            llm_value=llm_customer_status.value,
            crm_value=crm_status.value,
            crm_confidence=crm_status.confidence,
            crm_source=crm_status.source
        )

    return crm_status


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def add_email_to_crm(email: str, append_to_file: bool = True):
    """
    Add email to mock CRM database.

    Args:
        email: Email address to add
        append_to_file: If True, append to known_customers.txt file
    """
    if append_to_file:
        emails_file = Path(settings.crm_mock_known_emails_file)
        emails_file.parent.mkdir(parents=True, exist_ok=True)

        try:
            with open(emails_file, 'a', encoding='utf-8') as f:
                f.write(f"{email.lower().strip()}\n")

            logger.info("email_added_to_mock_crm", email=email)
        except Exception as e:
            logger.error("failed_to_add_email_to_crm", email=email, error=str(e))


def check_crm_availability() -> bool:
    """
    Check if CRM lookup is configured and available.

    Returns:
        True if CRM is ready, False otherwise
    """
    if not settings.enable_crm_lookup:
        return False

    # Check if at least one CRM source is configured
    has_domains = bool(settings.crm_mock_known_domains.strip())
    emails_file = Path(settings.crm_mock_known_emails_file)
    has_emails_file = emails_file.exists()

    return has_domains or has_emails_file
