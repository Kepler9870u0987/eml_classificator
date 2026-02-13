"""
Prompt management for LLM classification.

Provides versioned prompt templates with:
- Clear instructions for multi-label classification
- Italian topics taxonomy
- Strict keyword selection rules
- Examples for guidance
- Structured output format

Based on brainstorming v2 section 3.3 (prompting "vincolato")
and v3 specifications.
"""

from typing import Dict, Any, List, Optional
from dataclasses import dataclass

from eml_classificator.config import settings


# ============================================================================
# PROMPT VERSIONS
# ============================================================================

CURRENT_PROMPT_VERSION = "v2.2"


# ============================================================================
# SYSTEM PROMPT
# ============================================================================

SYSTEM_PROMPT_V2_2 = """You are a precise email triage classifier for Italian customer service.

Your task is to analyze customer service emails and classify them using ONLY the structured function calling format.

CRITICAL RULES:
1. **Topics (multi-label)**: Select 1-5 topic labels from the allowed taxonomy. If content is unclear or doesn't fit existing labels, use UNKNOWN_TOPIC.

2. **Keywords**: Select keywords ONLY from the 'candidate_keywords' list provided in the user message. Each keyword MUST have a valid candidate_id. DO NOT invent new keywords not in the candidate list.

3. **Evidence**: Provide 1-2 short quotes (max 200 chars each) from the email that justify each topic assignment.

4. **Sentiment**: Classify as:
   - positive: customer expresses satisfaction, gratitude, positive tone
   - neutral: informational, no emotional charge
   - negative: complaints, frustration, dissatisfaction, anger

5. **Priority**: Assess urgency as:
   - urgent: immediate action required (SLA breach, blocking issue, legal threat)
   - high: same-day response needed
   - medium: 1-2 days acceptable
   - low: routine, no time pressure
   Provide 'signals' with phrases from the email justifying the priority level.

6. **Customer Status**: Provide your best guess based on text signals only:
   - new: appears to be first contact
   - existing: mentions having a contract, being a client, etc.
   - unknown: cannot determine from text
   This will be verified against CRM data in post-processing.

OUTPUT FORMAT:
- Return valid JSON via the classify_email function call
- All fields are required
- Confidence values must be between 0.0 and 1.0
- At least one topic, one keyword per topic, one evidence per topic

LANGUAGE:
- Email content is in Italian
- Analyze the Italian text carefully
- Topics are in Italian (FATTURAZIONE, ASSISTENZA_TECNICA, etc.)
"""


# ============================================================================
# EXAMPLES
# ============================================================================

EXAMPLES = [
    {
        "subject": "Problema con fattura di dicembre",
        "body": "Buongiorno, ho ricevuto la fattura n. 12345 per il mese di dicembre ma l'importo non è corretto. Risulta 150€ invece di 120€ come da contratto. Chiedo gentilmente una verifica urgente.",
        "expected_topics": ["FATTURAZIONE", "RECLAMO"],
        "expected_sentiment": "negative",
        "expected_priority": "high",
        "reasoning": "Billing issue with incorrect amount, polite but urgent request"
    },
    {
        "subject": "Richiesta informazioni commerciali",
        "body": "Buonasera, vorrei avere maggiori informazioni sui vostri piani aziendali per 50 dipendenti. Potete inviarmi un preventivo? Grazie",
        "expected_topics": ["INFO_COMMERCIALI"],
        "expected_sentiment": "neutral",
        "expected_priority": "medium",
        "reasoning": "Commercial inquiry, neutral tone, standard priority"
    },
    {
        "subject": "Assistenza urgente - servizio non funzionante",
        "body": "Il sistema è completamente bloccato da stamattina e non riusciamo a lavorare. Abbiamo bisogno di supporto immediato! Questa situazione sta causando gravi perdite.",
        "expected_topics": ["ASSISTENZA_TECNICA", "RECLAMO"],
        "expected_sentiment": "negative",
        "expected_priority": "urgent",
        "reasoning": "Blocking technical issue, business impact, requires immediate action"
    },
]


def format_examples_for_prompt() -> str:
    """
    Format examples for inclusion in prompt.

    Returns formatted string with examples showing expected classification.
    """
    if not settings.prompt_include_examples:
        return ""

    examples_text = "\n\nEXAMPLES OF GOOD CLASSIFICATION:\n"

    for i, ex in enumerate(EXAMPLES, 1):
        examples_text += f"""
Example {i}:
Subject: "{ex['subject']}"
Body: "{ex['body']}"

Expected:
- Topics: {ex['expected_topics']}
- Sentiment: {ex['expected_sentiment']}
- Priority: {ex['expected_priority']}
- Reasoning: {ex['reasoning']}
"""

    return examples_text


# ============================================================================
# USER PROMPT BUILDER
# ============================================================================

@dataclass
class EmailToClassify:
    """Email data for classification."""
    message_id: str
    subject: str
    from_address: str
    body_canonical: str
    dictionary_version: int


def build_user_prompt(
    email: EmailToClassify,
    candidates: List[Dict[str, Any]],
    max_candidates: int = 100,
    max_body_length: int = 8000
) -> str:
    """
    Build user prompt from email document and candidate keywords.

    Args:
        email: EmailToClassify with email data
        candidates: List of candidate keywords with scores
        max_candidates: Maximum number of candidates to include (top N by score)
        max_body_length: Maximum body length (truncate if longer)

    Returns:
        Formatted user prompt string
    """
    # Truncate body if too long
    body = email.body_canonical
    if len(body) > max_body_length:
        body = body[:max_body_length] + "\n\n[... testo troncato per lunghezza ...]"

    # Sort candidates by composite_score (descending) and take top N
    sorted_candidates = sorted(
        candidates,
        key=lambda c: c.get("composite_score", 0.0),
        reverse=True
    )[:max_candidates]

    # Format candidates for prompt
    candidates_formatted = []
    for c in sorted_candidates:
        candidates_formatted.append({
            "candidate_id": c["candidate_id"],
            "term": c["term"],
            "lemma": c["lemma"],
            "count": c["count"],
            "source": c["source"],
            "score": round(c.get("composite_score", 0.0), 3)
        })

    # Build prompt
    prompt = f"""Classify the following Italian customer service email.

DICTIONARY VERSION: {email.dictionary_version}

EMAIL HEADERS:
- From: {email.from_address}
- Subject: {email.subject}

EMAIL BODY:
{body}

ALLOWED TOPICS (select 1-5):
- FATTURAZIONE: invoices, billing, payments, pricing
- ASSISTENZA_TECNICA: technical support, troubleshooting, bugs, system issues
- RECLAMO: complaints, dissatisfaction, problems, claims
- INFO_COMMERCIALI: commercial information, quotes, product inquiries
- DOCUMENTI: document requests, contracts, paperwork
- APPUNTAMENTO: appointments, meetings, scheduling
- CONTRATTO: contract management, renewals, modifications
- GARANZIA: warranty, guarantees, covered issues
- SPEDIZIONE: shipping, delivery, logistics
- UNKNOWN_TOPIC: use this if email doesn't fit other categories clearly

CANDIDATE KEYWORDS (select keywords ONLY from this list):
{format_candidates_json(candidates_formatted)}

INSTRUCTIONS:
1. Analyze the email carefully
2. Select 1-5 relevant topics from the allowed list
3. For each topic, select supporting keywords from the candidate list (by candidate_id)
4. Provide 1-2 short evidence quotes per topic
5. Classify sentiment (positive/neutral/negative)
6. Assess priority (low/medium/high/urgent) with justification signals
7. Guess customer status from text (new/existing/unknown)
8. Return results via the classify_email function call

Remember:
- Keywords MUST come from the candidate list above (reference by candidate_id)
- Provide at least 1 keyword and 1 evidence per topic
- Confidence scores between 0.0-1.0
- Evidence quotes max 200 characters each
"""

    # Add examples if enabled
    if settings.prompt_include_examples:
        prompt += format_examples_for_prompt()

    return prompt


def format_candidates_json(candidates: List[Dict[str, Any]]) -> str:
    """
    Format candidates as compact JSON for prompt.

    Args:
        candidates: List of candidate dicts

    Returns:
        Formatted JSON string
    """
    import json
    # Format compactly
    return json.dumps(candidates, ensure_ascii=False, indent=2)


# ============================================================================
# PROMPT RETRIEVAL
# ============================================================================

def get_system_prompt(version: str = CURRENT_PROMPT_VERSION) -> str:
    """
    Get system prompt by version.

    Args:
        version: Prompt version (default: current)

    Returns:
        System prompt string

    Raises:
        ValueError: If version not found
    """
    if version == "v2.2":
        return SYSTEM_PROMPT_V2_2
    else:
        raise ValueError(f"Unknown prompt version: {version}")


def build_classification_prompt(
    email: EmailToClassify,
    candidates: List[Dict[str, Any]],
    prompt_version: Optional[str] = None
) -> tuple[str, str]:
    """
    Build complete classification prompt (system + user).

    Args:
        email: Email to classify
        candidates: Candidate keywords
        prompt_version: Prompt version to use (default: from settings)

    Returns:
        (system_prompt, user_prompt) tuple
    """
    version = prompt_version or settings.prompt_version

    system_prompt = get_system_prompt(version)

    user_prompt = build_user_prompt(
        email=email,
        candidates=candidates,
        max_candidates=settings.classification_max_candidates_in_prompt,
        max_body_length=settings.prompt_max_body_length
    )

    return system_prompt, user_prompt


# ============================================================================
# PROMPT TEMPLATES FOR DIFFERENT USE CASES
# ============================================================================

def build_simple_prompt(subject: str, body: str) -> str:
    """
    Build simplified prompt for quick testing (no candidates).

    Not for production use - just for testing LLM connectivity.
    """
    return f"""Classify this Italian email:

Subject: {subject}
Body: {body}

Select topics, sentiment, and priority. Use the classify_email function."""


# ============================================================================
# VALIDATION REMINDERS
# ============================================================================

VALIDATION_REMINDER = """
VALIDATION CHECKLIST:
✓ At least 1 topic assigned
✓ Each topic has 1-15 keywords
✓ All keywords have valid candidate_id from input
✓ Each topic has 1-2 evidence quotes
✓ All confidence scores in range [0.0, 1.0]
✓ Priority has signals/justification
✓ Output is valid JSON matching the schema
"""


def add_validation_reminder_to_prompt(prompt: str) -> str:
    """
    Add validation reminder to prompt (optional, for models that struggle).

    Args:
        prompt: Original prompt

    Returns:
        Prompt with validation reminder appended
    """
    return prompt + "\n\n" + VALIDATION_REMINDER
