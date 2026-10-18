# app/stt.py
"""
Speech-to-Text (STT) module using Google GenAI SDK (Gemini 3.8 Flash).
Reuses the existing GEMINI_API_KEY with zero additional paid services.
"""

import os
from app.llm import _get_gemini_client


def transcribe_audio(audio_path: str) -> str:
    """
    Transcribe a local audio file into text using Gemini multimodal capabilities.
    Raises RuntimeError if API key is missing or transcription fails.
    """
    client = _get_gemini_client()
    if not client:
        raise RuntimeError(
            "Gemini API key is not configured. Voice transcription is unavailable. "
            "Please use the text input fallback."
        )

    if not audio_path or not os.path.exists(audio_path):
        raise ValueError("No valid audio file provided for transcription.")

    try:
        # Upload audio file to Gemini Files API
        audio_file = client.files.upload(file=audio_path)
        
        response = client.models.generate_content(
            model="gemini-3.8-flash",  # Updated to current active model
            contents=[
                audio_file,
                (
                    "Transcribe this audio clip verbatim into text. "
                    "Pay special attention to dental terminology, tooth numbers, and patient names. "
                    "Return ONLY the transcribed text without extra commentary."
                ),
            ],
        )
        
        transcript = response.text.strip() if response and response.text else ""
        return transcript
    except Exception as exc:
        raise RuntimeError(f"Speech transcription failed: {exc}")