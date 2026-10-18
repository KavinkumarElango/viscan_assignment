# Viscan Voice-First Dental AI Assistant

Proof-of-concept AI system for patient report retrieval, session-aware grounded Q&A, and clinical dictation with deterministic safety validation. Built for the Viscan AI Engineer technical assessment[cite: 1, 3].

---

## Features

1. **Patient Report Retrieval (Mandatory):** Deterministic Python lookup handling unambiguous patient matches, ambiguity clarification (e.g. searching "Priya" returns candidate options), and safe missing report messages[cite: 1, 8].
2. **Clinical Dictation to Draft Report (Mandatory):** Converts spoken or typed clinical observations into structured JSON and an editable draft, enforcing a strict post-processing safety validator that prevents hallucinated diagnoses, preserves uncertainty, and maps FDI tooth notation[cite: 1, 10, 11].
3. **Conversational Follow-Up & Grounded Q&A (Mandatory + Bonus):** Session-aware context memory allowing follow-ups like "Summarize her report" or "What tooth is impacted?" without asking for the patient name again, citing report evidence IDs[cite: 1, 4, 12, 16].
Optional Voice Input & Transcription: Record voice notes via Gradio microphone. Audio is transcribed using Gemini 2.5 Flash (reusing existing API key with zero added cost) and presented in a review box for clinician verification before dispatching to the backend.

---

## Tech Stack

- **Backend:** Python 3.12 + FastAPI + Uvicorn[cite: 5]
- **Validation:** Pydantic (`extra="forbid"`) + custom Python rule validators[cite: 9, 10]
- **AI Provider:** Google GenAI SDK (`gemini-2.5-flash`) with structured outputs + deterministic offline fallback[cite: 5, 8, 16]
- **UI:** Gradio (2-tab interface: Assistant Chat & Clinical Dictation)[cite: 3, 11, 14]
- **Testing:** `pytest` (8 automated test cases covering retrieval, ambiguity, and safety checks)[cite: 14, 15]

---

## Setup & Installation

1. **Clone repository & open in VS Code**[cite: 5]
2. **Create and activate virtual environment (PowerShell):**[cite: 5]
   ```powershell
   python -m venv .venv
   .\.venv\Scripts\Activate.ps1
   python -m pip install -r requirements.txt