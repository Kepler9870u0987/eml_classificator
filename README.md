# Email Classification Pipeline - Ingestion Layer

Deterministic email parsing, canonicalization, and PII redaction service for the email classification pipeline.

## Overview

The Ingestion Layer is the first component of a deterministic, auditable email triage pipeline. It provides:

- **RFC5322/MIME parsing** of .eml files
- **Text canonicalization** with Italian signature detection
- **PII redaction** with complete audit trail
- **REST API** via FastAPI
- **Statistical determinism** - same input + version = same output

### Key Principle: Statistical Determinism

A parità di `PipelineVersion`, lo stesso input produce lo stesso output. Tracciabilità completa per audit e backtesting.

## Features

✅ Parse .eml files (RFC5322/MIME compliant)
✅ Extract headers, body, attachments metadata
✅ Convert HTML emails to plain text
✅ Canonicalize text (normalization, signatures, quotes)
✅ Redact PII (email addresses, phone numbers) with audit log
✅ Track 10 component versions for reproducibility
✅ REST API with FastAPI
✅ Structured logging (JSON) with structlog
✅ GDPR-compliant PII handling

## Architecture

```
Ingestion Pipeline:
1. Upload .eml file
2. Parse RFC5322/MIME → headers, body, attachments
3. Extract text (plain or HTML→text)
4. Canonicalize (normalize, signatures, quotes, lowercase)
5. PII Redaction (emails, phones with placeholders)
6. Create EmailDocument + PipelineVersion
7. Return JSON via API
```

### Project Structure

```
eml_classificator/
├── src/eml_classificator/
│   ├── models/               # Data models (Pydantic)
│   ├── parsing/              # Email parsing (RFC5322/MIME)
│   ├── canonicalization/     # Text normalization
│   ├── pii/                  # PII redaction + audit
│   ├── api/                  # FastAPI application
│   ├── version.py            # Version constants
│   ├── config.py             # Configuration (env vars)
│   └── logging_config.py     # Structured logging
├── tests/                    # Tests (unit + integration)
├── pyproject.toml            # Poetry dependencies
└── README.md                 # This file
```

## Installation

### Requirements

- Python 3.11+
- Poetry (recommended) or pip

### Setup with Poetry

```bash
# Clone repository
git clone <repo-url>
cd eml_classificator

# Install dependencies
poetry install

# Copy environment file
cp .env.example .env

# Edit .env and set PII_SALT to a secure random value
# IMPORTANT: Change PII_SALT in production!

# Run API server (development mode with auto-reload)
poetry run uvicorn eml_classificator.api.app:app --reload --port 8000
```

### Setup with pip (alternative)

```bash
pip install -r requirements.txt  # (generate from poetry export)
```

## Configuration

Configuration via environment variables (see `.env.example`):

```bash
# API Configuration
API_HOST=0.0.0.0
API_PORT=8000

# Logging
LOG_LEVEL=INFO
LOG_JSON=true

# PII Redaction (CHANGE IN PRODUCTION!)
PII_SALT=change-this-to-random-value
DEFAULT_PII_LEVEL=standard

# Processing Limits
MAX_EMAIL_SIZE_MB=25
MAX_ATTACHMENTS=50
```

**⚠️ IMPORTANT**: Change `PII_SALT` to a cryptographically secure random value in production!

```bash
# Generate secure salt
python -c "import secrets; print(secrets.token_hex(32))"
```

## API Usage

### Start the Server

```bash
# Development mode (auto-reload)
poetry run uvicorn eml_classificator.api.app:app --reload --port 8000

# Production mode (4 workers)
poetry run uvicorn eml_classificator.api.app:app \
  --host 0.0.0.0 \
  --port 8000 \
  --workers 4
```

### API Endpoints

#### 1. Health Check

```bash
curl http://localhost:8000/health
```

Response:
```json
{
  "status": "healthy",
  "version": "1.0.0",
  "uptime_seconds": 3600.5
}
```

#### 2. Get Version Info

```bash
curl http://localhost:8000/api/v1/version
```

Response:
```json
{
  "api_version": "1.0.0",
  "pipeline_version": {
    "dictionary_version": 1,
    "model_version": "deepseek-r1-2026-01",
    "parser_version": "eml-parser-1.0.0",
    "canonicalization_version": "canon-1.0.0",
    "pii_redaction_version": "pii-redact-1.0.0",
    "pii_redaction_level": "standard",
    ...
  }
}
```

#### 3. Ingest Email

```bash
curl -X POST http://localhost:8000/api/v1/ingest/eml \
  -F "file=@path/to/email.eml" \
  -F "pii_redaction_level=standard" \
  -F "store_html=false"
```

Response:
```json
{
  "success": true,
  "document": {
    "document_id": "eml-sha256-abc123...",
    "headers": {
      "from_address": "customer@example.com",
      "to_addresses": ["support@company.com"],
      "subject": "Problema con ordine #12345",
      "date": "2026-02-12T10:30:00Z"
    },
    "canonical_text": "buongiorno problema ordine 12345...",
    "pii_redacted": true,
    "pii_redaction_log": [
      {
        "type": "email",
        "position": [45, 65],
        "original_hash": "sha256-...",
        "placeholder": "[EMAIL_1]",
        "pii_redaction_version": "pii-redact-1.0.0",
        "pii_redaction_level": "standard"
      }
    ],
    "pipeline_version": { ... },
    "processing_time_ms": 125.3
  }
}
```

### PII Redaction Levels

- `none` - No PII redaction
- `minimal` - Only email addresses in body
- `standard` - Email addresses + phone numbers (default)
- `aggressive` - Emails + phones + person names (requires NER - not yet implemented)

### OpenAPI Documentation

Interactive API documentation available at:

- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc
- **OpenAPI JSON**: http://localhost:8000/openapi.json

## Data Models

### PipelineVersion

Immutable version contract tracking 10 component versions:

- `dictionary_version` - Dynamic dictionaries (RegEx/NER)
- `model_version` - LLM model identifier
- `parser_version` - Email parser version
- `stoplist_version` - Italian stopwords version
- `ner_model_version` - spaCy NER model version
- `schema_version` - JSON schema version
- `tool_calling_version` - LLM tool calling API version
- `canonicalization_version` - Text canonicalization version
- `pii_redaction_version` - PII redaction algorithm version
- `pii_redaction_level` - PII redaction aggressiveness

### EmailDocument

Canonical output after ingestion:

- `document_id` - SHA-256 hash (deterministic)
- `headers` - From, To, Subject, Date, Message-ID, References
- `body_text` - Extracted plain text
- `body_html` - Original HTML (optional)
- `canonical_text` - Canonicalized and PII-redacted text
- `attachments` - Metadata only (filename, type, size)
- `pii_redaction_log` - Audit log of all redactions
- `pipeline_version` - Complete version tracking
- `processing_time_ms` - Performance metric

## Text Canonicalization

Deterministic normalization pipeline:

1. **Normalize line endings** (`\r\n` → `\n`)
2. **Normalize whitespace** (multi-space → single)
3. **Remove signatures** (Italian patterns: "Cordiali saluti", "--", etc.)
4. **Remove reply chains** (">", "On ... wrote:", etc.)
5. **Lowercase**
6. **Strip whitespace**

### Italian Signature Patterns

- `Cordiali saluti`
- `Distinti saluti`
- `Saluti`
- `--` / `___` (delimiter)
- `Sent from my iPhone` / `Inviato da`

## PII Redaction

### Email Addresses

Regex: `\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b`

Replaced with: `[EMAIL_1]`, `[EMAIL_2]`, etc.

### Phone Numbers (Italian)

Regex: `\b(\+39\s?)?((0\d{1,4}[\s\-]?\d{4,8})|([3]\d{2}[\s\-]?\d{6,7}))\b`

Matches:
- `+39 02 12345678`
- `0XX XXXXXXX` (landline)
- `3XX XXXXXXX` (mobile)

Replaced with: `[PHONE_1]`, `[PHONE_2]`, etc.

### Audit Log

Each redaction logged with:
- `type`: "email" or "phone"
- `position`: [start, end] in original text
- `original_hash`: SHA-256 hash (salted, reversible with salt)
- `placeholder`: Replacement token
- `pii_redaction_version`: Algorithm version
- `pii_redaction_level`: Redaction level used

**CRITICAL**: Audit log includes `pii_redaction_version` and `pii_redaction_level` for GDPR compliance and backtesting.

## Development

### Run Tests

```bash
# Run all tests
poetry run pytest

# Run with coverage
poetry run pytest --cov=src/eml_classificator --cov-report=html

# Run specific test file
poetry run pytest tests/unit/test_eml_parser.py -v
```

### Code Quality

```bash
# Format code
poetry run black src/ tests/

# Lint code
poetry run ruff check src/ tests/

# Type checking
poetry run mypy src/
```

### Project Structure Conventions

- All modules export public API via `__init__.py` with `__all__`
- Type hints on all public functions
- Docstrings (Google style) on all public functions
- Structured logging with context binding
- Immutable data models where appropriate

## Deployment

### Pre-Production Checklist

- [ ] Change `PII_SALT` to cryptographically secure value
- [ ] Configure CORS origins (not `*` in production)
- [ ] Set up log aggregation (ELK, CloudWatch, etc.)
- [ ] Configure Prometheus metrics scraping
- [ ] Set up alerts (error rate, processing time)
- [ ] Test with real email samples
- [ ] Document version upgrade procedures
- [ ] Set up CI/CD pipeline
- [ ] Configure rate limiting
- [ ] Health check monitoring
- [ ] GDPR compliance audit

### Docker

```bash
# Build image
docker build -t eml-classificator:latest .

# Run container
docker run -d -p 8000:8000 \
  -e PII_SALT=<secure-value> \
  eml-classificator:latest
```

### Production Deployment

```bash
# Run with Uvicorn (4 workers)
uvicorn eml_classificator.api.app:app \
  --host 0.0.0.0 \
  --port 8000 \
  --workers 4 \
  --log-config logging_config.py
```

## Monitoring & Observability

### Structured Logging

All logs output as JSON (when `LOG_JSON=true`):

```json
{
  "event": "Email ingested successfully",
  "level": "info",
  "timestamp": "2026-02-12T10:30:00.000Z",
  "document_id": "eml-sha256-abc...",
  "subject": "Problema con ordine",
  "processing_time_ms": 125.3,
  "pii_redactions": 2
}
```

### Metrics

Key metrics to monitor:

- **Processing time** (`processing_time_ms`)
- **Error rate** (failed ingestions)
- **PII redaction count** (per email)
- **Email size** (`raw_size_bytes`)
- **Attachment count**

## Troubleshooting

### Common Issues

**Issue**: "File must be .eml format"
- **Solution**: Ensure file has `.eml` extension

**Issue**: "File size exceeds maximum"
- **Solution**: Increase `MAX_EMAIL_SIZE_MB` in `.env`

**Issue**: PII not redacted
- **Solution**: Check `pii_redaction_level` is not `none`

**Issue**: HTML email body is empty
- **Solution**: Check `store_html=true` parameter if you need HTML

**Issue**: Signature not removed
- **Solution**: Add custom patterns to `SIGNATURE_PATTERNS` in `text_normalizer.py`

## Contributing

1. Fork the repository
2. Create feature branch (`git checkout -b feature/amazing-feature`)
3. Make changes with tests
4. Run tests and linting (`poetry run pytest && poetry run ruff check`)
5. Commit (`git commit -m 'Add amazing feature'`)
6. Push (`git push origin feature/amazing-feature`)
7. Open Pull Request

## License

[Your License Here]

## References

- **RFC 5322**: Internet Message Format
- **MIME Standard**: RFC 2045-2049
- **FastAPI**: https://fastapi.tiangolo.com
- **Pydantic**: https://docs.pydantic.dev
- **structlog**: https://www.structlog.org

## Contact

[Your Contact Info]

---

**Version**: 1.0.0
**Last Updated**: 2026-02-12
**Status**: Production-ready Ingestion Layer
