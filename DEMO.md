# DEMO.md: Viscan Dental AI Assistant Walkthrough Guide

This document outlines the step-by-step demonstration script for reviewing the system's core capabilities.

---

## Scenario 1: Patient Report Retrieval & Ambiguity Handling
* **Clinician Input:** *"Open Priya's OPG report"*
* **Backend Behavior:** The intent classifier detects `retrieve_report`. The deterministic search matches both *Priya Sharma* (P001) and *Priya Verma* (P002).
* **Expected UI Output:**
  ```text
  ⚠️ Ambiguous Match: Multiple patients found matching 'Priya'.
  Candidates Found:
  - P001: Priya Sharma (Age: 29)
  - P002: Priya Verma (Age: 34)
  👉 Action Required: Type 'Select P001' or 'Choose Priya Sharma' to confirm.

Scenario 2: Session-Aware Follow-up Q&A
Clinician Input: "Summarize her report" (relying on active session context).

Backend Behavior: Identifies summarize_report, retrieves active patient/report ID from session memory, and summarizes findings.

Expected UI Output: Concise clinical summary highlighting bone levels and restorative status.

Scenario 3: Optional Voice-to-Text Input
Action: Click "Optional Voice Input (Microphone)", record speech saying "Open Priya Sharma OPG".

Backend Behavior: Audio is uploaded to Gemini STT (app/stt.py) and transcribed.

Expected UI Output: Transcribed text appears in the review text box for clinician verification before clicking Send Request.

Scenario 4: Clinical Dictation & Safety Validation
Clinician Dictation / Input: "Lower right third molar is horizontally impacted. Possible mild bone loss near tooth 46."

Backend Behavior: Parses findings, maps FDI tooth numbers (38 for lower right third molar, 46 for lower right first molar), and validates against safety rules.

Expected UI Output:

Editable Draft Report: Formatted clinical note ready for clinician sign-off.

Structured JSON: Validated JSON array of findings with evidence/laterality metadata.

Safety Status: ✅ Validation Passed: Findings fully grounded in source transcript.