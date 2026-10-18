# ui.py
"""
Redesigned professional healthcare dashboard for the Viscan Dental AI Assistant.
Provides an intuitive, accessible, clean interface for patient report retrieval,
interactive session Q&A, and clinical dictation with safety validation.
"""

import uuid
import gradio as gr
import requests

API_BASE = "http://127.0.0.1:8000"

# Professional Healthcare Dashboard Custom CSS
CUSTOM_CSS = """
:root {
    --primary-teal: #0f4c5c;
    --secondary-teal: #1d7874;
    --accent-coral: #e36414;
    --bg-light: #f8f9fa;
    --card-bg: #ffffff;
    --text-main: #2b2d42;
    --border-color: #e2e8f0;
}
body {
    background-color: var(--bg-light);
    color: var(--text-main);
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
}
.gradio-container {
    max-width: 1200px !important;
    margin: auto;
    padding: 20px;
}
.header-box {
    background: linear-gradient(135deg, #0f4c5c 0%, #1d7874 100%);
    color: white;
    padding: 24px;
    border-radius: 12px;
    margin-bottom: 20px;
    box-shadow: 0 4px 6px rgba(0,0,0,0.05);
}
.header-box h1 {
    color: white !important;
    margin-bottom: 8px;
    font-size: 1.8rem;
}
.header-box p {
    color: #e2e8f0 !important;
    font-size: 0.95rem;
    margin: 0;
}
.action-card-btn {
    background-color: white !important;
    border: 1px solid var(--border-color) !important;
    border-radius: 8px !important;
    color: var(--primary-teal) !important;
    font-weight: 600 !important;
    text-align: left !important;
    transition: all 0.2s ease-in-out;
}
.action-card-btn:hover {
    border-color: var(--primary-teal) !important;
    background-color: #f0f7f7 !important;
    box-shadow: 0 2px 4px rgba(15, 76, 92, 0.1);
}
"""


def handle_audio_transcription(audio_path: str):
    """Transcribe recorded microphone audio and populate review box."""
    if not audio_path:
        return "", "⚠️ No audio recorded."
    
    try:
        from app.stt import transcribe_audio
        transcript = transcribe_audio(audio_path)
        if not transcript:
            return "", "⚠️ Transcription returned empty text. Please try again or type manually."
        return transcript, f"✅ Speech successfully transcribed ({len(transcript.split())} words). Review and edit below before sending."
    except Exception as exc:
        return "", f"❌ Voice transcription unavailable: {exc}. Please use text fallback."


def chat_with_assistant(message: str, history: list, session_id_state: str):
    """Send message to FastAPI /assistant orchestrator and format response."""
    if not message.strip():
        return history, session_id_state

    if not session_id_state:
        session_id_state = f"session_{uuid.uuid4().hex[:8]}"

    try:
        response = requests.post(
            f"{API_BASE}/assistant",
            json={"session_id": session_id_state, "message": message},
            timeout=15,
        )
        data = response.json()
    except Exception as exc:
        history.append({"role": "user", "content": message})
        history.append({"role": "assistant", "content": f"⚠️ **Communication Error:** Backend API unreachable ({exc})."})
        return history, session_id_state

    status = data.get("status")
    intent = data.get("intent")
    reply_text = ""

    if status == "ambiguous_patient":
        candidates = data.get("candidates", [])
        cand_list = "\n".join(f"- **{c['patient_id']}**: {c['name']} (Age: {c.get('age', 'N/A')})" for c in candidates)
        reply_text = (
            f"⚠️ **Ambiguous Match:** {data.get('message')}\n\n"
            f"**Candidates Found:**\n{cand_list}\n\n"
            f"👉 *Action Required: Type 'Select P001' or 'Choose Priya Sharma' to confirm patient identity.*"
        )
    elif status == "patient_not_found" or status == "report_not_found":
        reply_text = f"❌ **{data.get('message')}**"
    elif status == "success":
        if intent == "retrieve_report" or "report" in data:
            rep = data.get("report", {})
            pat = data.get("patient", {})
            reply_text = (
                f"✅ **Report Retrieved Successfully** (Evidence ID: `{rep.get('report_id')}`)\n\n"
                f"👤 **Patient:** {pat.get('name')} (`{pat.get('patient_id')}`)\n"
                f"📋 **Report Type:** {rep.get('report_type')} | **Date:** {rep.get('report_date')}\n\n"
                f"**Clinical Findings:**\n" + "".join(f"  • {f}\n" for f in rep.get('findings', []))
            )
        elif intent == "select_patient":
            rep = data.get("report", {})
            pat = data.get("patient", {})
            reply_text = (
                f"✅ **Patient Selected & Active Report Loaded**\n\n"
                f"👤 **Patient:** {pat.get('name')} (`{pat.get('patient_id')}`)\n"
                f"📁 **Active Report:** `{rep.get('report_id')}` ({rep.get('report_type')} - {rep.get('report_date')})"
            )
        elif intent in ("summarize_report", "question_report"):
            reply_text = (
                f"💬 **Report Analysis** (Session Report: `{data.get('report_id')}`)\n\n"
                f"{data.get('message')}"
            )
        else:
            reply_text = data.get("message") or str(data)
    else:
        reply_text = data.get("message") or str(data)

    history.append({"role": "user", "content": message})
    history.append({"role": "assistant", "content": reply_text})
    
    return history, session_id_state


def run_clinical_dictation(transcript: str):
    """Send dictation transcript to FastAPI /dictation endpoint."""
    if not transcript.strip():
        return "Please enter or dictate a clinical observation.", "[]", "⚠️ No transcription text provided."

    try:
        response = requests.post(
            f"{API_BASE}/dictation",
            json={"transcript": transcript},
            timeout=10,
        )
        data = response.json()
    except Exception as exc:
        return f"Error connecting to backend: {exc}", "[]", f"❌ Error: {exc}"

    import json
    formatted_json = json.dumps(data.get("findings", []), indent=2)
    editable_draft = data.get("draft_report", "")
    warnings = data.get("warnings", [])
    
    warning_text = "\n".join(f"⚠️ {w}" for w in warnings) if warnings else "✅ Validation Passed: Findings fully grounded in source transcript."
    status_label = f"Status: {data.get('status', 'draft_pending_clinician_review')}\n\n{warning_text}"

    return editable_draft, formatted_json, status_label


# --- REDESIGNED GRADIO LAYOUT (Gradio 6.0+ Compliant) ---
with gr.Blocks(title="Viscan Dental AI Assistant") as demo:
    session_id_box = gr.State(value="")

    # Professional Header Banner
    with gr.Row(elem_classes="header-box"):
        with gr.Column():
            gr.Markdown("# 🦷 Viscan Dental AI Assistant")
            gr.Markdown("Clinical Decision Support & Voice-First Diagnostic Report Retrieval Platform *(Synthetic Data)*")

    with gr.Tabs():
        # =====================================================================
        # TAB 1: ASSISTANT CHAT & REPORT LOOKUP
        # =====================================================================
        with gr.TabItem("💬 Assistant Chat & Report Lookup"):
            
            # Quick Action Cards Row
            gr.Markdown("### ⚡ Quick Clinical Actions")
            with gr.Row():
                quick_btn_1 = gr.Button("📂 Open Priya's Latest OPG", elem_classes="action-card-btn")
                quick_btn_2 = gr.Button("📊 Summarize Active Report", elem_classes="action-card-btn")
                quick_btn_3 = gr.Button("🦷 Check Impacted Tooth Q&A", elem_classes="action-card-btn")
                quick_btn_4 = gr.Button("🔍 Test Ambiguity (Search 'Priya')", elem_classes="action-card-btn")

            gr.Markdown("---")

            # Main Interface Layout (Input Section on Left, Guidance on Right)
            with gr.Row():
                with gr.Column(scale=3):
                    # 1. CLINICAL REQUEST & REVIEW SECTION PLACED ABOVE HISTORY
                    chat_input = gr.Textbox(
                        label="Clinician Request (Review transcribed speech or type request)",
                        placeholder="Type a request e.g., 'Open Priya Sharma OPG' or 'Summarize report'...",
                        lines=2
                    )
                    
                    with gr.Row():
                        chat_submit = gr.Button("Send Request", variant="primary", scale=2)
                        clear_btn = gr.Button("Clear Conversation", variant="secondary", scale=1)

                    # 2. SESSION HISTORY CHATBOT PLACED BELOW INPUT
                    chat_chatbot = gr.Chatbot(
                        label="Patient Session History",
                        height=380
                    )

                # Right Column: Expandable Voice Controls & Workflow Guidance
                with gr.Column(scale=2):
                    with gr.Accordion("🎙️ Optional Voice Input (Microphone)", open=False):
                        gr.Markdown("Record a voice query. Speech is transcribed into text for your verification before dispatching.")
                        audio_mic = gr.Audio(sources=["microphone"], type="filepath", label="Microphone Recording")
                        transcribe_btn = gr.Button("Transcribe Audio", variant="secondary")
                        transcribe_status = gr.Markdown(value="")

                    with gr.Accordion("📖 Clinician Workflow Guide", open=True):
                        gr.Markdown(
                            "**1. Patient Lookup:** Ask to open any patient report (e.g. *'Open Priya OPG'*).\n"
                            "**2. Ambiguity Safeguard:** If multiple patients match, select a candidate ID (e.g. *'Select P001'*).\n"
                            "**3. Session Memory:** Once loaded, ask follow-ups without repeating names (e.g. *'Summarize her report'* or *'Is there a mandibular fracture?'*)."
                        )

            # Quick Action Button Wiring
            quick_btn_1.click(fn=lambda h, s: chat_with_assistant("Open Priya's latest OPG report", h, s), inputs=[chat_chatbot, session_id_box], outputs=[chat_chatbot, session_id_box])
            quick_btn_2.click(fn=lambda h, s: chat_with_assistant("Summarize her report", h, s), inputs=[chat_chatbot, session_id_box], outputs=[chat_chatbot, session_id_box])
            quick_btn_3.click(fn=lambda h, s: chat_with_assistant("What tooth is impacted?", h, s), inputs=[chat_chatbot, session_id_box], outputs=[chat_chatbot, session_id_box])
            quick_btn_4.click(fn=lambda h, s: chat_with_assistant("Open Priya OPG", h, s), inputs=[chat_chatbot, session_id_box], outputs=[chat_chatbot, session_id_box])

            # Voice Transcription Wiring
            transcribe_btn.click(
                fn=handle_audio_transcription,
                inputs=[audio_mic],
                outputs=[chat_input, transcribe_status],
            )

            # Chat Submission Wiring
            chat_submit.click(
                fn=chat_with_assistant,
                inputs=[chat_input, chat_chatbot, session_id_box],
                outputs=[chat_chatbot, session_id_box],
            )
            chat_input.submit(
                fn=chat_with_assistant,
                inputs=[chat_input, chat_chatbot, session_id_box],
                outputs=[chat_chatbot, session_id_box],
            )

            # Clear Conversation Handler
            clear_btn.click(fn=lambda: ([], ""), inputs=None, outputs=[chat_chatbot, session_id_box])

        # =====================================================================
        # TAB 2: CLINICAL DICTATION & SAFETY VALIDATOR
        # =====================================================================
        with gr.TabItem("🎙️ Clinical Dictation & Safety Validator"):
            gr.Markdown(
                "### 📋 Deterministic Clinical Safety Validation Pipeline"
                "\nDictate or type clinical findings below. The safety validator extracts structured findings, "
                "maps FDI tooth numbers, preserves uncertainty, and blocks unsupported clinical claims."
            )

            with gr.Accordion("🎙️ Optional Voice Dictation Recording", open=False):
                with gr.Row():
                    dictation_mic = gr.Audio(sources=["microphone"], type="filepath", label="Microphone Input")
                    dictation_transcribe_btn = gr.Button("Transcribe Voice Note", variant="secondary")
                dictation_transcribe_status = gr.Markdown(value="")

            dictation_input = gr.Textbox(
                label="Clinical Observations Transcript (Review & Edit before submission)",
                placeholder="e.g. Lower right third molar is horizontally impacted. Possible caries in upper left first molar.",
                lines=4,
            )
            
            dictation_btn = gr.Button("Generate Safety-Validated Draft Report", variant="primary", size="lg")

            gr.Markdown("---")

            with gr.Row():
                with gr.Column():
                    draft_output = gr.Textbox(
                        label="Editable Draft Report (Clinician Sign-off Required)",
                        lines=6,
                    )
                with gr.Column():
                    json_output = gr.Code(
                        label="Structured JSON Findings (Auditable Metadata)",
                        language="json",
                    )

            warning_output = gr.Textbox(
                label="Safety & Compliance Status",
                lines=3,
            )

            # Wire up voice dictation transcription
            dictation_transcribe_btn.click(
                fn=handle_audio_transcription,
                inputs=[dictation_mic],
                outputs=[dictation_input, dictation_transcribe_status],
            )

            # Wire up structured dictation generation
            dictation_btn.click(
                fn=run_clinical_dictation,
                inputs=[dictation_input],
                outputs=[draft_output, json_output, warning_output],
            )

if __name__ == "__main__":
    demo.launch(server_name="127.0.0.1", server_port=7860, share=False, theme=gr.themes.Soft(), css=CUSTOM_CSS)