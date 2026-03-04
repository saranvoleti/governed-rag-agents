"""
core/config.py — System Identity Object

Every agent receives a AnalysisConfig. Nothing is hardcoded anywhere else.
This is the single source of truth for a session.

Lesson from v1: DatasetConfig flowing through all agents was the best
design decision in the old system. Keeping it, simplifying it.
"""

from dataclasses import dataclass, field
from typing import Optional
from datetime import datetime, timezone
import uuid


@dataclass(frozen=True)
class AnalysisConfig:
    """
    Identity object for one analysis session.
    Created when user uploads a document or starts a query.
    Immutable after creation — agents read, never write.
    """

    # Session identity
    session_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    created_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )

    # Domain — what kind of document is this?
    # Agents use this to select the right policy signals and thresholds
    domain: str = "general"          # general | finance | healthcare | legal | ops

    # Document info — filled by Ingestion Agent after upload
    document_name: str = ""
    document_type: str = ""          # pdf | docx | csv | txt
    chunk_count: int = 0

    # Retrieval settings
    # How many chunks to retrieve per query — more = more context, more cost
    top_k_chunks: int = 5

    # Reflection settings
    # How many times the Reflection Agent may attempt rewrite before fallback
    max_reflection_attempts: int = 2

    # Escalation settings
    escalation_threshold: float = 0.6

    # LLM settings
    # Temperature 0.1 — minimal creativity, maximum fidelity to retrieved context
    llm_temperature: float = 0.1
    llm_max_tokens: int = 1000

    # Schema version — increment when this contract changes
    schema_version: str = "1.0"

    def to_dict(self) -> dict:
        return {
            "session_id": self.session_id,
            "created_at": self.created_at,
            "domain": self.domain,
            "document_name": self.document_name,
            "document_type": self.document_type,
            "chunk_count": self.chunk_count,
            "top_k_chunks": self.top_k_chunks,
            "max_reflection_attempts": self.max_reflection_attempts,
            "escalation_threshold": self.escalation_threshold,
            "llm_temperature": self.llm_temperature,
            "llm_max_tokens": self.llm_max_tokens,
            "schema_version": self.schema_version,
        }


# ── Domain policy registry ────────────────────────────────────────────────────
# Each domain has different escalation language and policy signals.
# Add a new domain here — zero code changes anywhere else.

DOMAIN_POLICY = {
    "general": {
        "escalation_label": "Review Required",
        "confidence_label": "Confidence",
        "signals": ["verify_sources", "check_recency", "flag_assumptions"],
    },
    "finance": {
        "escalation_label": "Compliance Review Required",
        "confidence_label": "Regulatory Confidence",
        "signals": ["sec_compliance", "material_disclosure", "forward_looking_statements"],
    },
    "healthcare": {
        "escalation_label": "Clinical Review Required",
        "confidence_label": "Clinical Confidence",
        "signals": ["contraindication_check", "dosage_flag", "off_label_use"],
    },
    "legal": {
        "escalation_label": "Legal Review Required",
        "confidence_label": "Legal Confidence",
        "signals": ["jurisdiction_check", "precedent_flag", "liability_exposure"],
    },
    "ops": {
        "escalation_label": "Operations Review Required",
        "confidence_label": "Operational Confidence",
        "signals": ["sla_breach", "capacity_threshold", "vendor_dependency"],
    },
}


def get_policy(domain: str) -> dict:
    """Returns policy signals for a domain. Falls back to general."""
    return DOMAIN_POLICY.get(domain, DOMAIN_POLICY["general"])
