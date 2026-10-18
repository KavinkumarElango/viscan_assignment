# app/tools.py
"""
Deterministic Python functions for loading synthetic data, searching patients,
and retrieving reports without relying on LLM guesswork.
"""

import json
from pathlib import Path

# Path to our data folder
DATA_DIR = Path(__file__).resolve().parent.parent / "data"


def load_patients() -> list[dict]:
    """Load the synthetic patients table from data/patients.json."""
    with open(DATA_DIR / "patients.json", "r", encoding="utf-8") as f:
        return json.load(f)


def load_reports() -> list[dict]:
    """Load the synthetic reports table from data/reports.json."""
    with open(DATA_DIR / "reports.json", "r", encoding="utf-8") as f:
        return json.load(f)


def find_patients(name_or_id: str, patients: list[dict]) -> list[dict]:
    """
    Search patients by exact patient_id (e.g. 'P001') OR case-insensitive
    partial name match (e.g. 'Priya' matches 'Priya Sharma' and 'Priya Verma').
    """
    query = name_or_id.strip().casefold()
    if not query:
        return []

    # 1. Check if the user typed an exact Patient ID like "P001"
    exact_id_matches = [
        p for p in patients if p["patient_id"].casefold() == query
    ]
    if exact_id_matches:
        return exact_id_matches

    # 2. Otherwise search by partial name match
    return [p for p in patients if query in p["name"].casefold()]


def get_patient_by_id(patient_id: str, patients: list[dict]) -> dict | None:
    """Get a single patient dictionary by their exact patient_id."""
    for p in patients:
        if p["patient_id"].casefold() == patient_id.strip().casefold():
            return p
    return None


def get_report_by_id(report_id: str, reports: list[dict]) -> dict | None:
    """Get a single report dictionary by its exact report_id."""
    for r in reports:
        if r["report_id"].casefold() == report_id.strip().casefold():
            return r
    return None


def get_patient_reports(
    patient_id: str, reports: list[dict], report_type: str | None = None
) -> list[dict]:
    """Get all reports for a patient, optionally filtered by report_type (e.g. OPG)."""
    matches = [r for r in reports if r["patient_id"] == patient_id]
    if report_type:
        matches = [
            r for r in matches
            if r["report_type"].casefold() == report_type.strip().casefold()
        ]
    # Sort newest first by report_date, then report_id
    return sorted(
        matches, key=lambda r: (r["report_date"], r["report_id"]), reverse=True
    )


def get_latest_report(
    patient_id: str, report_type: str | None, reports: list[dict]
) -> dict | None:
    """
    Return the most recent report for a patient (matching the student guide logic).
    If report_type is None, returns the latest report of any type.
    """
    matches = [
        r for r in reports
        if r["patient_id"] == patient_id
        and (
            not report_type
            or r["report_type"].casefold() == report_type.strip().casefold()
        )
    ]
    if not matches:
        return None
    return max(matches, key=lambda r: (r["report_date"], r["report_id"]))


def retrieve_report_workflow(
    patient_query: str,
    report_type: str | None = None,
    patients: list[dict] | None = None,
    reports: list[dict] | None = None,
) -> dict:
    """
    Complete deterministic business-rule workflow for Feature A:
    Handles not_found, ambiguous_patient, report_not_found, and success.
    """
    if patients is None:
        patients = load_patients()
    if reports is None:
        reports = load_reports()

    matched_patients = find_patients(patient_query, patients)

    # Case 1: No matching patients
    if not matched_patients:
        return {
            "status": "patient_not_found",
            "message": f"No patient found matching '{patient_query}'.",
            "candidates": [],
            "patient": None,
            "report": None,
        }

    # Case 2: Ambiguous match (2 or more patients, e.g. "Priya")
    if len(matched_patients) > 1:
        candidates = [
            {"patient_id": p["patient_id"], "name": p["name"], "age": p.get("age")}
            for p in matched_patients
        ]
        candidate_str = ", ".join(
            f"{c['patient_id']} ({c['name']})" for c in candidates
        )
        return {
            "status": "ambiguous_patient",
            "message": (
                f"Multiple patients match '{patient_query}': {candidate_str}. "
                "Please specify the full name or Patient ID."
            ),
            "candidates": candidates,
            "patient": None,
            "report": None,
        }

    # Case 3: Exactly one patient matched -> look up their report
    patient = matched_patients[0]
    report = get_latest_report(patient["patient_id"], report_type, reports)

    if not report:
        type_label = f" {report_type.upper()}" if report_type else ""
        return {
            "status": "report_not_found",
            "message": (
                f"Patient {patient['name']} ({patient['patient_id']}) was found, "
                f"but no{type_label} reports are on file."
            ),
            "candidates": [],
            "patient": patient,
            "report": None,
        }

    # Case 4: Success!
    return {
        "status": "success",
        "message": (
            f"Retrieved latest {report['report_type']} report ({report['report_id']}) "
            f"for {patient['name']} ({patient['patient_id']}) dated {report['report_date']}."
        ),
        "candidates": [],
        "patient": patient,
        "report": report,
    }

def get_all_reports_for_patient(patient_id: str) -> list:
    """Return all diagnostic reports associated with a specific patient ID."""
    # Assuming MOCK_REPORTS is your dictionary/list of reports in tools.py
    patient_reports = []
    for rep_id, rep_data in MOCK_REPORTS.items():
        if rep_data.get("patient_id") == patient_id:
            patient_reports.append(rep_data)
    return patient_reports