# app/dictation.py
"""
Deterministic post-LLM clinical validation rules for Feature B.
Verifies source text grounding, controlled vocabulary, uncertainty preservation,
negation consistency, laterality, and FDI tooth notation.
"""

import re
from app.models import ClinicalFinding, DraftReport

# 1. Controlled vocabulary of supported demo clinical concepts
CONTROLLED_OBSERVATION_TERMS = {
    "impacted",
    "impaction",
    "caries",
    "carious",
    "cavity",
    "radiolucency",
    "radiolucent",
    "pain",
    "restoration",
    "filling",
    "root canal",
    "missing",
    "edentulous",
    "bone loss",
    "calculus",
    "mobility",
}

# 2. High-risk clinical claims that must NEVER appear unless explicitly in source_text
RESTRICTED_CLINICAL_CLAIMS = {
    "abscess",
    "infection",
    "cyst",
    "tumor",
    "malignancy",
    "fracture",
    "necrosis",
    "granuloma",
    "osteomyelitis",
    "deep caries",
    "bone loss",
    "periapical lesion",
}

# 3. Words indicating diagnostic uncertainty
UNCERTAINTY_MARKERS = {
    "possible",
    "possibly",
    "suspected",
    "suspect",
    "probable",
    "likely",
    "questionable",
    "rule out",
    "may be",
    "suggestive of",
}

# 4. Words that falsely upgrade an uncertain finding to definite
CERTAINTY_UPGRADE_WORDS = {
    "confirmed",
    "definite",
    "definitely",
    "definitive",
    "proven",
    "established",
    "severe",
    "acute",
}

# 5. Negation markers
NEGATION_MARKERS = {
    "no ",
    "not ",
    "without ",
    "denies ",
    "absence of ",
    "negative for ",
    "free of ",
}

# 6. Controlled FDI mapping for common anatomical tooth descriptions
ANATOMY_TO_FDI = {
    "upper right central incisor": "11",
    "upper right lateral incisor": "12",
    "upper right canine": "13",
    "upper right first premolar": "14",
    "upper right second premolar": "15",
    "upper right first molar": "16",
    "upper right second molar": "17",
    "upper right third molar": "18",
    "upper left central incisor": "21",
    "upper left lateral incisor": "22",
    "upper left canine": "23",
    "upper left first premolar": "24",
    "upper left second premolar": "25",
    "upper left first molar": "26",
    "upper left second molar": "27",
    "upper left third molar": "28",
    "lower left central incisor": "31",
    "lower left lateral incisor": "32",
    "lower left canine": "33",
    "lower left first premolar": "34",
    "lower left second premolar": "35",
    "lower left first molar": "36",
    "lower left second molar": "37",
    "lower left third molar": "38",
    "lower right central incisor": "41",
    "lower right lateral incisor": "42",
    "lower right canine": "43",
    "lower right first premolar": "44",
    "lower right second premolar": "45",
    "lower right first molar": "46",
    "lower right second molar": "47",
    "lower right third molar": "48",
}

# FDI Quadrant to jaw/side rules
FDI_QUADRANT_RULES = {
    "1": ("upper", "right"),
    "2": ("upper", "left"),
    "3": ("lower", "left"),
    "4": ("lower", "right"),
}


def _contains_word_or_phrase(text: str, phrase: str) -> bool:
    """Case-insensitive check for a word or phrase using word boundaries."""
    pattern = r"\b" + re.escape(phrase.strip().casefold()) + r"\b"
    return bool(re.search(pattern, text.casefold()))


def infer_fdi_from_site(anatomical_site: str | None) -> str | None:
    """Lookup FDI tooth number from an anatomical site description if known."""
    if not anatomical_site:
        return None
    site_clean = anatomical_site.strip().casefold()
    for desc, fdi in ANATOMY_TO_FDI.items():
        if desc in site_clean:
            return fdi
    return None


def validate_draft_report(
    transcript: str, raw_draft: DraftReport
) -> DraftReport:
    """
    Validate an LLM-generated DraftReport against the clinician's original transcript.
    Rejects unsupported/hallucinated findings, preserves uncertainty, checks negation
    and laterality, and flags any discrepancies in `warnings`.
    """
    transcript_clean = transcript.strip()
    transcript_lower = transcript_clean.casefold()

    accepted_findings: list[ClinicalFinding] = []
    warnings: list[str] = list(raw_draft.warnings)

    if not transcript_clean:
        return DraftReport(
            findings=[],
            draft_report="",
            status="draft_pending_clinician_review",
            warnings=["Empty dictation transcript provided."],
        )

    for finding in raw_draft.findings:
        obs_lower = finding.observation.casefold()
        src_clean = finding.source_text.strip()
        src_lower = src_clean.casefold()
        site_lower = (finding.anatomical_site or "").casefold()

        # Check 1: Verbatim Source Grounding
        if not src_clean or src_lower not in transcript_lower:
            warnings.append(
                f"Rejected finding '{finding.observation}': source_text "
                f"'{finding.source_text}' is not a verbatim span of the transcript."
            )
            continue

        # Check 2: Restricted / Hallucinated Clinical Claims
        unsupported_claim_found = False
        for claim in RESTRICTED_CLINICAL_CLAIMS:
            if _contains_word_or_phrase(obs_lower, claim) and not _contains_word_or_phrase(src_lower, claim):
                warnings.append(
                    f"Rejected unsupported clinical claim '{claim}' in finding "
                    f"'{finding.observation}' (not stated in source text)."
                )
                unsupported_claim_found = True
                break
        if unsupported_claim_found:
            continue

        # Check 3: Controlled Observation Vocabulary Check
        in_controlled_set = any(
            _contains_word_or_phrase(obs_lower, term) or _contains_word_or_phrase(src_lower, term)
            for term in CONTROLLED_OBSERVATION_TERMS
        )
        if not in_controlled_set:
            warnings.append(
                f"Flagged for manual review: finding '{finding.observation}' falls outside "
                "the controlled demo observation vocabulary."
            )

        # Check 4: Uncertainty Preservation & Blocking False Certainty Upgrades
        source_is_uncertain = any(
            _contains_word_or_phrase(src_lower, marker)
            for marker in UNCERTAINTY_MARKERS
        )
        has_upgrade_word = any(
            _contains_word_or_phrase(obs_lower, upg) and not _contains_word_or_phrase(src_lower, upg)
            for upg in CERTAINTY_UPGRADE_WORDS
        )

        certainty = finding.certainty
        observation_text = finding.observation

        if source_is_uncertain:
            if certainty != "uncertain":
                warnings.append(
                    f"Corrected certainty from 'stated' to 'uncertain' for '{finding.observation}' "
                    f"based on hedging language in source ('{finding.source_text}')."
                )
                certainty = "uncertain"
            if has_upgrade_word:
                warnings.append(
                    f"Rejected certainty upgrade in '{finding.observation}': source text "
                    f"('{finding.source_text}') is uncertain."
                )
                # Strip upgrade words from observation so it cannot mislead
                for upg in CERTAINTY_UPGRADE_WORDS:
                    observation_text = re.sub(
                        r"\b" + re.escape(upg) + r"\b",
                        "possible",
                        observation_text,
                        flags=re.IGNORECASE,
                    )
        elif has_upgrade_word:
            warnings.append(
                f"Flagged unverified qualifier in '{finding.observation}' not present in source text."
            )

        # Check 5: Negation Flip Detection
        source_has_negation = any(neg in f" {src_lower} " for neg in NEGATION_MARKERS)
        obs_has_negation = any(neg in f" {obs_lower} " for neg in NEGATION_MARKERS)
        if source_has_negation and not obs_has_negation:
            warnings.append(
                f"Rejected finding '{finding.observation}': source text contains a negation "
                f"('{finding.source_text}') that was omitted in the finding."
            )
            continue

        # Check 6: Laterality (left/right, upper/lower) consistency
        laterality_error = False
        for side, opposite in [("left", "right"), ("right", "left"), ("upper", "lower"), ("lower", "upper")]:
            if (
                (_contains_word_or_phrase(obs_lower, side) or _contains_word_or_phrase(site_lower, side))
                and _contains_word_or_phrase(src_lower, opposite)
                and not _contains_word_or_phrase(src_lower, side)
            ):
                warnings.append(
                    f"Rejected finding '{finding.observation}': laterality mismatch between "
                    f"finding ('{side}') and source text ('{opposite}')."
                )
                laterality_error = True
                break
        if laterality_error:
            continue

        # Check 7: FDI Tooth Number Verification
        fdi = finding.tooth_number_fdi
        expected_fdi = infer_fdi_from_site(finding.anatomical_site) or infer_fdi_from_site(finding.source_text)
        if expected_fdi and fdi and fdi != expected_fdi:
            warnings.append(
                f"Corrected FDI tooth number from '{fdi}' to '{expected_fdi}' to match "
                f"anatomical site '{finding.anatomical_site or finding.source_text}'."
            )
            fdi = expected_fdi
        elif expected_fdi and not fdi:
            fdi = expected_fdi
        elif fdi and len(fdi) == 2 and fdi[0] in FDI_QUADRANT_RULES:
            jaw, side = FDI_QUADRANT_RULES[fdi[0]]
            combined_site = f"{site_lower} {src_lower}"
            if (
                ("upper" in combined_site or "lower" in combined_site)
                and jaw not in combined_site
            ) or (
                ("left" in combined_site or "right" in combined_site)
                and side not in combined_site
            ):
                warnings.append(
                    f"Cleared mismatched FDI tooth number '{fdi}' (quadrant implies {jaw} {side})."
                )
                fdi = None

        accepted_findings.append(
            ClinicalFinding(
                observation=observation_text.strip(),
                anatomical_site=finding.anatomical_site,
                tooth_number_fdi=fdi,
                certainty=certainty,
                source_text=src_clean,
            )
        )

    # Check 8: Sanitize the readable draft_report text against restricted claims & upgrades
    draft_text = raw_draft.draft_report.strip()
    draft_lower = draft_text.casefold()

    for claim in RESTRICTED_CLINICAL_CLAIMS:
        if _contains_word_or_phrase(draft_lower, claim) and not _contains_word_or_phrase(transcript_lower, claim):
            warnings.append(
                f"Removed unsupported claim '{claim}' from draft report text."
            )
            draft_text = ""
            break

    for upg in CERTAINTY_UPGRADE_WORDS:
        if _contains_word_or_phrase(draft_lower, upg) and not _contains_word_or_phrase(transcript_lower, upg):
            warnings.append(
                f"Removed unverified certainty upgrade '{upg}' from draft report text."
            )
            draft_text = ""
            break

    # Rebuild a safe, deterministic draft sentence from accepted_findings if needed
    if not draft_text and accepted_findings:
        sentences = []
        for f in accepted_findings:
            prefix = "Possible finding (uncertain): " if f.certainty == "uncertain" else ""
            site_info = f" at {f.anatomical_site}" if f.anatomical_site and f.anatomical_site.casefold() not in f.observation.casefold() else ""
            fdi_info = f" (FDI tooth {f.tooth_number_fdi})" if f.tooth_number_fdi else ""
            sentences.append(f"{prefix}{f.observation}{site_info}{fdi_info}.")
        draft_text = " ".join(sentences)
    elif not accepted_findings:
        draft_text = "No validated clinical findings could be extracted. Manual clinician entry required."

    return DraftReport(
        findings=accepted_findings,
        draft_report=draft_text,
        status="draft_pending_clinician_review",
        warnings=warnings,
    )