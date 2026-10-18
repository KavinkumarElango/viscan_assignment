# app/models.py
"""
Pydantic schemas enforcing strict input/output data types for
intent classification, clinical dictation drafts, and API payloads.
"""

from typing import Literal
from pydantic import BaseModel, ConfigDict, Field


class IntentResult(BaseModel):
    """Schema for LLM intent classification in Feature A & C."""
    model_config = ConfigDict(extra="forbid")

    intent: Literal[
        "retrieve_report",
        "select_patient",
        "summarize_report",
        "question_report",
        "clinical_dictation",
        "unknown",
    ]
    patient_name: str | None = None
    patient_id: str | None = None
    report_type: str | None = None
    latest: bool = True
    question: str | None = None


class ClinicalFinding(BaseModel):
    """Schema for a single extracted clinical observation in Feature B."""
    model_config = ConfigDict(extra="forbid")

    observation: str
    anatomical_site: str | None = None
    tooth_number_fdi: str | None = None
    certainty: Literal["stated", "uncertain"]
    source_text: str


class DraftReport(BaseModel):
    """Schema for the validated clinical draft report in Feature B."""
    model_config = ConfigDict(extra="forbid")

    findings: list[ClinicalFinding]
    draft_report: str
    status: Literal["draft_pending_clinician_review"] = "draft_pending_clinician_review"
    warnings: list[str] = Field(default_factory=list)