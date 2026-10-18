# ARCHITECTURE.md: Viscan Dental AI Assistant

## 1. System Overview
The Viscan Dental AI Assistant is a modular, voice-first clinical decision support system designed for dental practices. It provides deterministic patient report retrieval, session-aware clinical Q&A, and a safety-validated clinical dictation pipeline.

---

## 2. Component Architecture & Data Flow
┌────────────────────────────────────────────────────────┐
│               Gradio UI Dashboard (ui.py)              │
│  - Tab 1: Assistant Chat & Report Lookup (w/ Voice STT)│
│  - Tab 2: Clinical Dictation & Safety Validator        │
└───────────────────────────┬────────────────────────────┘
│ HTTP POST (JSON / Multipart)
▼
┌────────────────────────────────────────────────────────┐
│            FastAPI Backend Orchestrator                │
│            (app/main.py & app/llm.py)                  │
└───┬───────────────────────────────────┬────────────────┘
│                                   │
▼ (Intent & Entity Parsing)         ▼ (Dictation Safety Check)
┌───────────────────────┐       ┌────────────────────────────────┐
│ Deterministic Tools   │       │ Safety & Validation Engine     │
│ (app/tools.py)        │       │ (app/dictation.py)             │
│ - Patient Lookup      │       │ - FDI Tooth Number Mapping     │
│ - Ambiguity Resolver  │       │ - Uncertainty & Negation Check │
│ - Report Retrieval    │       │ - Hallucination Blocker        │
└───────────────────────┘       └────────────────────────────────┘
---

## 3. Core Modules

- **`app/models.py`**: Pydantic data validation schemas for patients, diagnostic reports, chat requests, session contexts, and dictation outputs.
- **`app/tools.py`**: Deterministic Python mock database operations handling fuzzy name matching, ambiguity detection (`ambiguous_patient`), and report ID scoping.
- **`app/llm.py`**: Google GenAI integration (`gemini-3.8-flash`) for natural language intent classification, slot filling, and clinical summarization.
- **`app/stt.py`**: Multimodal Speech-to-Text module using Gemini Flash file uploads for verbatim audio transcription.
- **`app/dictation.py`**: NLP and rule-based validator ensuring generated clinical draft reports are strictly grounded in source observations without unsupported claims.
- **`app/main.py`**: FastAPI server exposing REST endpoints (`/assistant`, `/dictation`, `/health`).

---

## 4. Safety & Governance
- **Ambiguity Guard**: Never automatically resolves multi-patient matches (e.g., *Priya Sharma* vs *Priya Verma*); prompts the clinician for explicit candidate ID confirmation (`Select P001`).
- **Hallucination Prevention**: The dictation validator cross-checks all generated findings against the source transcript, rejecting unmentioned pathology or unverified severity claims.
- **Data Privacy**: Operates entirely on isolated synthetic mock records with stateless or session-scoped memory.