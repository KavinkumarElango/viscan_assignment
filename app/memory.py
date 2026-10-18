# app/memory.py
"""
Simple in-memory session store to track which patient and report
a clinician is currently viewing during a conversation.
"""

sessions: dict[str, dict[str, str | None]] = {}


def set_context(session_id: str, patient_id: str, report_id: str | None = None) -> dict:
    """Save the active patient_id and report_id for a specific session."""
    sessions[session_id] = {
        "patient_id": patient_id,
        "report_id": report_id,
    }
    return sessions[session_id]


def get_context(session_id: str) -> dict | None:
    """Retrieve the active patient_id and report_id for a specific session."""
    return sessions.get(session_id)


def clear_context(session_id: str) -> None:
    """Clear session memory (useful for testing or resetting chat)."""
    sessions.pop(session_id, None)