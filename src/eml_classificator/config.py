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

    # Phase 2: Candidate generation
    enable_embeddings: bool = True
    keybert_top_n: int = 50
    keybert_diversity: float = 0.7
    candidate_min_term_length: int = 3
    candidate_max_ngram: int = 3
    spacy_model_name: str = "it_core_news_lg"

    # Composite scoring weights
    score_weight_count: float = 0.3
    score_weight_embedding: float = 0.5
    score_weight_subject_bonus: float = 0.2

    # Phase 3: LLM Classification
    # LLM Provider Configuration
    llm_provider: str = "ollama"  # "ollama" | "openai" | "deepseek" | "openrouter"
    llm_model: str = "deepseek-r1:7b"  # Model name (provider-specific)
    llm_api_key: str = ""  # Optional for Ollama, required for cloud providers
    llm_api_base_url: str = "http://localhost:11434"  # Ollama default
    llm_temperature: float = 0.1  # Low for determinism
    llm_max_tokens: int = 2048
    llm_timeout_seconds: int = 120  # Higher for reasoning models
    llm_max_retries: int = 3
    llm_retry_delay_seconds: float = 1.0

    # Model-specific configuration
    llm_supports_tool_calling: bool = True  # Auto-detect based on provider/model
    llm_tool_calling_format: str = "openai"  # "openai" | "anthropic" | "none"

    # Classification settings
    enable_tool_calling: bool = True
    classification_confidence_threshold: float = 0.3
    classification_max_topics: int = 5
    classification_max_keywords_per_topic: int = 15
    classification_max_evidence_per_topic: int = 2
    classification_max_candidates_in_prompt: int = 100  # Top N by composite score

    # Priority scoring weights (rule-based)
    priority_weight_urgent_terms: float = 3.0
    priority_weight_high_terms: float = 1.5
    priority_weight_sentiment_negative: float = 2.0
    priority_weight_customer_new: float = 1.0
    priority_weight_deadline: float = 2.0
    priority_weight_vip: float = 2.5

    # Priority bucketing thresholds
    priority_threshold_urgent: float = 7.0  # score >= 7.0 → urgent
    priority_threshold_high: float = 4.0    # score >= 4.0 → high
    priority_threshold_medium: float = 2.0  # score >= 2.0 → medium

    # Confidence adjustment weights
    confidence_weight_llm: float = 0.3
    confidence_weight_keywords: float = 0.4
    confidence_weight_evidence: float = 0.2
    confidence_weight_collision: float = 0.1

    # Customer status (Mock CRM)
    enable_crm_lookup: bool = True
    crm_mock_known_domains: str = "company.com,cliente.it"  # Comma-separated
    crm_mock_known_emails_file: str = "data/known_customers.txt"  # Path to email list
    crm_lookup_timeout_seconds: float = 1.0

    # Prompt configuration
    prompt_version: str = "v2.2"
    prompt_include_examples: bool = True
    prompt_max_body_length: int = 8000  # Truncate long emails

    # Phase 3: Dictionary database settings
    dictionary_db_url: str = "postgresql://user:password@localhost:5432/eml_classificator"
    dictionary_db_pool_size: int = 10
    dictionary_db_echo_sql: bool = False

    # Dictionary version (incremented when dictionaries are updated)
    current_dictionary_version: int = 1

    # Keyword promoter settings (deterministic promotion rules)
    promoter_regex_min_doc_freq: int = 3  # Min unique documents for regex promotion
    promoter_regex_min_embedding_score: float = 0.35  # Min avg KeyBERT score for regex
    promoter_ner_min_doc_freq: int = 2  # Min unique documents for NER promotion
    promoter_ner_min_embedding_score: float = 0.25  # Min avg KeyBERT score for NER
    promoter_max_collision_labels: int = 2  # Keyword in >2 labels = quarantine
    promoter_min_total_count: int = 5  # Cumulative count minimum

    # Entity extraction settings
    entity_extraction_enable_regex: bool = True
    entity_extraction_enable_ner: bool = True
    entity_extraction_enable_lexicon: bool = True

    model_config = {
        "env_file": ".env",
        "env_file_encoding": "utf-8",
    }


# Global settings instance
settings = Settings()
