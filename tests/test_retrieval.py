# tests/test_retrieval.py
"""
Automated unit tests for Feature A (patient search, ambiguity resolution,
latest report selection, and missing report handling) and Feature C (session memory).
"""

from app.tools import find_patients, get_latest_report, retrieve_report_workflow
from app.memory import set_context, get_context, clear_context


def test_latest_report_selection():
    """Verify get_latest_report selects the newest report by ISO date."""
    reports = [
        {
            "report_id": "R001",
            "patient_id": "P001",
            "report_type": "OPG",
            "report_date": "2026-01-10",
            "findings": ["Initial impaction noted."],
        },
        {
            "report_id": "R002",
            "patient_id": "P001",
            "report_type": "OPG",
            "report_date": "2026-06-15",
            "findings": ["Follow-up OPG shows persistent horizontal impaction."],
        },
    ]
    result = get_latest_report("P001", "OPG", reports)
    assert result is not None
    assert result["report_id"] == "R002"


def test_ambiguous_patient_search():
    """Verify searching 'Priya' matches both Priya Sharma and Priya Verma and asks for clarification."""
    patients = [
        {"patient_id": "P001", "name": "Priya Sharma", "age": 29},
        {"patient_id": "P002", "name": "Priya Verma", "age": 35},
        {"patient_id": "P003", "name": "Rahul Mehta", "age": 42},
    ]
    matches = find_patients("Priya", patients)
    assert len(matches) == 2

    workflow_result = retrieve_report_workflow("Priya", "OPG", patients=patients, reports=[])
    assert workflow_result["status"] == "ambiguous_patient"
    assert len(workflow_result["candidates"]) == 2
    assert workflow_result["report"] is None


def test_missing_report_and_unknown_patient():
    """Verify safe handling when a patient has no reports or does not exist."""
    patients = [{"patient_id": "P004", "name": "Aisha Khan", "age": 24}]
    reports = []

    missing_report_res = retrieve_report_workflow("Aisha Khan", "OPG", patients=patients, reports=reports)
    assert missing_report_res["status"] == "report_not_found"
    assert missing_report_res["patient"]["patient_id"] == "P004"
    assert missing_report_res["report"] is None

    unknown_patient_res = retrieve_report_workflow("Nonexistent Person", "OPG", patients=patients, reports=reports)
    assert unknown_patient_res["status"] == "patient_not_found"


def test_session_context_isolation():
    """Verify session memory isolates active patient/report per session_id."""
    clear_context("session_a")
    clear_context("session_b")

    set_context("session_a", "P001", "R002")
    set_context("session_b", "P003", "R005")

    assert get_context("session_a") == {"patient_id": "P001", "report_id": "R002"}
    assert get_context("session_b") == {"patient_id": "P003", "report_id": "R005"}