# tests/test_dictation.py
"""
Automated unit tests for Feature B clinical dictation safety validator.
Tests rejection of unsupported claims, preservation of uncertainty,
negation flip detection, and FDI tooth notation validation.
"""

from app.models import ClinicalFinding, DraftReport
from app.dictation import validate_draft_report


def test_valid_dictation_and_fdi_mapping():
    """Verify valid dictation preserves stated findings, maps FDI 48, and sets review status."""
    transcript = "Lower right third molar is horizontally impacted."
    mock_llm_draft = DraftReport(
        findings=[
            ClinicalFinding(
                observation="Horizontally impacted third molar",
                anatomical_site="Lower right third molar",
                tooth_number_fdi="48",
                certainty="stated",
                source_text="Lower right third molar is horizontally impacted.",
            )
        ],
        draft_report="The lower right third molar is horizontally impacted.",
    )

    validated = validate_draft_report(transcript, mock_llm_draft)
    assert validated.status == "draft_pending_clinician_review"
    assert len(validated.findings) == 1
    assert validated.findings[0].tooth_number_fdi == "48"
    assert validated.findings[0].certainty == "stated"


def test_rejects_unsupported_clinical_claims():
    """
    Verify that if an LLM invents an unsupported claim (e.g., 'abscess' or 'infection')
    not present in the transcript, the validator rejects the finding and flags a warning.
    """
    transcript = "Lower right third molar is horizontally impacted."
    mock_hallucinated_draft = DraftReport(
        findings=[
            ClinicalFinding(
                observation="Horizontally impacted third molar",
                anatomical_site="Lower right third molar",
                tooth_number_fdi="48",
                certainty="stated",
                source_text="Lower right third molar is horizontally impacted.",
            ),
            ClinicalFinding(
                observation="Periapical abscess and infection detected",
                anatomical_site="Lower right third molar",
                tooth_number_fdi="48",
                certainty="stated",
                source_text="Lower right third molar is horizontally impacted.",
            ),
        ],
        draft_report="Horizontally impacted lower right third molar with periapical abscess.",
    )

    validated = validate_draft_report(transcript, mock_hallucinated_draft)
    # Only the legitimate finding should remain
    assert len(validated.findings) == 1
    assert "abscess" not in validated.findings[0].observation.casefold()
    assert "abscess" not in validated.draft_report.casefold()
    assert any("abscess" in w.casefold() for w in validated.warnings)


def test_preserves_uncertainty_and_blocks_certainty_upgrade():
    """
    Verify that 'Possible caries' cannot be upgraded by the LLM to 'Confirmed caries'
    or certainty='stated'.
    """
    transcript = "Possible caries in upper left first molar."
    mock_upgraded_draft = DraftReport(
        findings=[
            ClinicalFinding(
                observation="Confirmed caries in upper left first molar",
                anatomical_site="Upper left first molar",
                tooth_number_fdi="26",
                certainty="stated",  # Wrongly marked 'stated' by mock LLM
                source_text="Possible caries in upper left first molar.",
            )
        ],
        draft_report="Confirmed caries noted in upper left first molar.",
    )

    validated = validate_draft_report(transcript, mock_upgraded_draft)
    assert len(validated.findings) == 1
    assert validated.findings[0].certainty == "uncertain"
    assert "confirmed" not in validated.findings[0].observation.casefold()
    assert "confirmed" not in validated.draft_report.casefold()
    assert len(validated.warnings) >= 1


def test_rejects_negation_flip_and_laterality_mismatch():
    """
    Verify that negated statements ('No bone loss') or flipped sides ('left' vs 'right')
    are rejected by the validator.
    """
    transcript = "No bone loss observed on the lower right first molar."
    mock_bad_draft = DraftReport(
        findings=[
            ClinicalFinding(
                observation="Bone loss observed",
                anatomical_site="Lower right first molar",
                tooth_number_fdi="46",
                certainty="stated",
                source_text="No bone loss observed on the lower right first molar.",
            ),
            ClinicalFinding(
                observation="finding on lower left first molar",
                anatomical_site="Lower left first molar",
                tooth_number_fdi="36",
                certainty="stated",
                source_text="No bone loss observed on the lower right first molar.",
            ),
        ],
        draft_report="Bone loss observed.",
    )

    validated = validate_draft_report(transcript, mock_bad_draft)
    assert len(validated.findings) == 0
    assert len(validated.warnings) >= 2