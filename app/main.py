# app/main.py
"""
FastAPI application routes for the Viscan Dental AI Assistant.
Exposes endpoints for health checks, intent processing, patient search,
report retrieval, clinical dictation validation, and report summarization/Q&A.
"""

from fastapi import FastAPI, HTTPException, status
from pydantic import BaseModel, Field

from app.tools import (
    load_patients,
    load_reports,
    retrieve_report_workflow,
    get_patient_reports,
    get_report_by_id,
)
from app.llm import classify_intent, extract_clinical_draft, answer_or_summarize_report
from app.memory import set_context, get_context, clear_context
from app.models import DraftReport

app = FastAPI(
    title="Viscan Dental AI Assistant",
    description="Voice-first dental assistant API for patient report retrieval and clinical dictation drafts.",
    version="1.0.0",
)


# --- API REQUEST / RESPONSE MODELS ---

class AssistantRequest(BaseModel):
    session_id: str = Field(min_length=1, description="Unique session identifier for context tracking.")
    message: str = Field(min_length=1, description="Typed or spoken clinician message.")


class PatientSearchRequest(BaseModel):
    query: str = Field(min_length=1, description="Patient name or ID.")
    report_type: str | None = Field(default=None, description="Optional report filter (e.g. OPG, CBCT).")


class DictationRequest(BaseModel):
    transcript: str = Field(min_length=1, description="Spoken or typed clinical dictation transcript.")


class SummarizeRequest(BaseModel):
    session_id: str = Field(min_length=1, description="Session identifier.")
    question: str | None = Field(default=None, description="Optional specific question about the report.")


# --- ENDPOINTS ---

@app.get("/health", status_code=status.HTTP_200_OK)
def health_check():
    """Return service status."""
    return {"status": "ok", "service": "viscan-dental-assistant"}


@app.post("/patients/search")
def search_patients_endpoint(payload: PatientSearchRequest):
    """Search for patient candidates or execute report retrieval workflow."""
    result = retrieve_report_workflow(payload.query, payload.report_type)
    return result


@app.get("/patients/{patient_id}/reports")
def get_reports_endpoint(patient_id: str, report_type: str | None = None):
    """Retrieve all reports for a specific patient ID."""
    patients = load_patients()
    reports = load_reports()
    
    patient = next((p for p in patients if p["patient_id"].casefold() == patient_id.casefold()), None)
    if not patient:
        raise HTTPException(status_code=404, detail=f"Patient ID '{patient_id}' not found.")
        
    patient_reports = get_patient_reports(patient_id, reports, report_type)
    return {
        "patient": patient,
        "count": len(patient_reports),
        "reports": patient_reports,
    }


@app.post("/dictation")
def dictation_endpoint(payload: DictationRequest):
    """Convert spoken/typed clinical dictation into structured JSON and an editable draft report."""
    draft = extract_clinical_draft(payload.transcript)
    return draft


@app.post("/reports/summarize")
def summarize_endpoint(payload: SummarizeRequest):
    """Summarize the active report or answer a grounded question using session context."""
    context = get_context(payload.session_id)
    if not context or not context.get("report_id"):
        raise HTTPException(
            status_code=400,
            detail="No active report selected in this session. Please open a patient report first."
        )
        
    reports = load_reports()
    patients = load_patients()
    
    report = get_report_by_id(context["report_id"], reports)
    if not report:
        raise HTTPException(status_code=404, detail="Selected report ID no longer exists.")
        
    patient = next((p for p in patients if p["patient_id"] == report["patient_id"]), {"name": "Unknown Patient"})
    
    response_text = answer_or_summarize_report(report, patient["name"], payload.question)
    return {
        "session_id": payload.session_id,
        "patient_id": patient["patient_id"],
        "patient_name": patient["name"],
        "report_id": report["report_id"],
        "report_date": report["report_date"],
        "response": response_text,
    }


@app.post("/assistant")
def assistant_orchestrator(payload: AssistantRequest):
    """
    Central orchestration endpoint for natural language requests:
    Handles intent classification, retrieval workflows, ambiguity,
    dictation routing, and session-aware follow-up Q&A.
    """
    session_id = payload.session_id
    message = payload.message.strip()
    
    # -------------------------------------------------------------------------
    # MULTI-REPORT SESSION CHECK: Handle requests for other/previous reports
    # -------------------------------------------------------------------------
    session_ctx = get_context(session_id)
    if session_ctx and session_ctx.get("patient_id") and any(phrase in message.casefold() for phrase in ["other report", "previous report", "past report", "any other", "more reports"]):
        pat_id = session_ctx["patient_id"]
        patients = load_patients()
        reports = load_reports()
        
        patient = next((p for p in patients if p["patient_id"] == pat_id), None)
        # Fix: reports is a list, so we iterate over it directly instead of calling .values()
        patient_reports = [r for r in reports if r.get("patient_id") == pat_id]
        
        if patient_reports:
            reps_info = []
            for r in patient_reports:
                reps_info.append(
                    f"- **{r.get('report_type')}** (ID: `{r.get('report_id')}`, Date: {r.get('report_date')})\n"
                    f"  Findings: {', '.join(r.get('findings', []))}"
                )
            return {
                "status": "success",
                "intent": "list_patient_reports",
                "patient": patient,
                "message": f"📋 **All Available Reports for {patient.get('name', pat_id)}:**\n\n" + "\n".join(reps_info)
            }
    # -------------------------------------------------------------------------

    # 1. Classify intent using LLM or deterministic fallback
    intent_result = classify_intent(message)
    
    # 2. Handle patient selection intent (e.g. user chose "P001" after an ambiguity prompt)
    if intent_result.intent == "select_patient" and intent_result.patient_id:
        patients = load_patients()
        reports = load_reports()
        patient = next((p for p in patients if p["patient_id"].casefold() == intent_result.patient_id.casefold()), None)
        
        if not patient:
            return {"status": "error", "message": f"Patient ID '{intent_result.patient_id}' not found."}
            
        report = retrieve_report_workflow(patient["patient_id"], intent_result.report_type, patients, reports)["report"]
        if report:
            set_context(session_id, patient["patient_id"], report["report_id"])
            return {
                "status": "success",
                "intent": "select_patient",
                "message": f"Selected {patient['name']} ({patient['patient_id']}). Active report set to {report['report_id']} ({report['report_type']}, {report['report_date']}).",
                "patient": patient,
                "report": report,
            }
        else:
            set_context(session_id, patient["patient_id"], None)
            return {
                "status": "report_not_found",
                "message": f"Selected {patient['name']}, but no matching reports were found.",
                "patient": patient,
                "report": None,
            }

    # 3. Handle report retrieval intent (e.g. "Open Priya's OPG")
    if intent_result.intent == "retrieve_report":
        patients = load_patients()
        reports = load_reports()
        
        # Safely extract the query attribute regardless of what IntentResult named it
        query_val = getattr(intent_result, "patient_query", None) or getattr(intent_result, "patient_name", None) or getattr(intent_result, "query", "")
        
        result = retrieve_report_workflow(query_val, intent_result.report_type, patients, reports)
        
        status = result["status"]
        if status == "ambiguous_patient":
            return {
                "status": "ambiguous_patient",
                "intent": "retrieve_report",
                "message": result["message"],
                "candidates": result["candidates"]
            }
        elif status == "patient_not_found":
            return {
                "status": "patient_not_found",
                "intent": "retrieve_report",
                "message": result["message"]
            }
        elif status == "report_not_found":
            return {
                "status": "report_not_found",
                "intent": "retrieve_report",
                "message": result["message"],
                "patient": result.get("patient")
            }
        elif status == "success":
            pat = result["patient"]
            rep = result["report"]
            set_context(session_id, pat["patient_id"], rep["report_id"])
            return {
                "status": "success",
                "intent": "retrieve_report",
                "message": f"Successfully retrieved report {rep['report_id']} for {pat['name']}.",
                "patient": pat,
                "report": rep
            }
    # 4. Handle session-aware follow-up Q&A or summarization
    if intent_result.intent in ("summarize_report", "question_report"):
        session_ctx = get_context(session_id)
        if not session_ctx or not session_ctx.get("report_id"):
            return {
                "status": "error",
                "intent": intent_result.intent,
                "message": "No active report loaded in session. Please open a patient report first (e.g., 'Open Priya Sharma OPG')."
            }
        
        reports = load_reports()
        patients = load_patients()
        
        # ---------------------------------------------------------------------
        # MULTI-REPORT SUMMARY CHECK: If user asks for "all reports" or "both reports"
        # ---------------------------------------------------------------------
        message_lower = message.casefold()
        if any(p in message_lower for p in ["all report", "every report", "both report", "summarize all"]) and session_ctx.get("patient_id"):
            pat_id = session_ctx["patient_id"]
            patient = next((p for p in patients if p["patient_id"] == pat_id), None)
            patient_reports = [r for r in reports if r.get("patient_id") == pat_id]
            
            if patient_reports:
                reps_summary = []
                for r in patient_reports:
                    reps_summary.append(
                        f"📁 **Report ID: {r.get('report_id')}** ({r.get('report_type')} - {r.get('report_date')}):\n" +
                        "".join(f"  • {f}\n" for f in r.get('findings', []))
                    )
                return {
                    "status": "success",
                    "intent": intent_result.intent,
                    "message": f"📊 **Comprehensive Summary of All Reports for {patient.get('name', pat_id)}:**\n\n" + "\n\n".join(reps_summary)
                }
        # ---------------------------------------------------------------------

        # Otherwise, standard single-report summarization/Q&A
        active_report = next((r for r in reports if r.get("report_id") == session_ctx["report_id"]), None)
        if not active_report:
            return {
                "status": "report_not_found",
                "intent": intent_result.intent,
                "message": "The active session report could not be found."
            }
            
        answer = answer_or_summarize_report(active_report, message, intent_result.intent)
        return {
            "status": "success",
            "intent": intent_result.intent,
            "report_id": active_report["report_id"],
            "message": answer
        }

    # Fallback for unrecognized requests
    return {
        "status": "error",
        "intent": "unknown",
        "message": f"I could not interpret '{message}'. Try asking to retrieve a report (e.g., 'Open Priya Sharma OPG'), summarize a report ('Summarize her report'), or check other reports."
    }