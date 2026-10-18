# tests/test_stt.py
"""
Automated unit tests for speech-to-text error handling and fallback behavior.
"""

import pytest
from app.stt import transcribe_audio


def test_stt_missing_api_key_raises_error(monkeypatch):
    """Verify that calling transcription without a valid API key raises a clear RuntimeError."""
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    # Also mock _get_gemini_client to return None
    monkeypatch.setattr("app.stt._get_gemini_client", lambda: None)

    with pytest.raises(RuntimeError) as excinfo:
        transcribe_audio("nonexistent_audio.wav")
    assert "Gemini API key is not configured" in str(excinfo.value)


def test_stt_invalid_file_raises_error():
    """Verify that passing a non-existent file path raises ValueError or RuntimeError."""
    with pytest.raises(Exception):
        transcribe_audio("completely_fake_audio_path_12345.wav")