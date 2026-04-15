"""
voice_handler.py
Speech-to-Text and Text-to-Speech for Hindi and English.
PyAudio is optional — if not installed (e.g. Streamlit Cloud), STT is disabled gracefully.
"""

import os
import tempfile
from gtts import gTTS

# Try importing speech recognition — may fail on cloud (no microphone/pyaudio)
try:
    import speech_recognition as sr
    SR_AVAILABLE = True
except ImportError:
    SR_AVAILABLE = False

try:
    import pyaudio  # noqa — just checking availability
    PYAUDIO_AVAILABLE = True
except ImportError:
    PYAUDIO_AVAILABLE = False


# ─────────────────────────────────────────────
# Speech → Text
# ─────────────────────────────────────────────

def speech_to_text(language: str = "auto") -> str | None:
    """
    Record from microphone and transcribe.
    Returns None gracefully if PyAudio / microphone not available.
    """
    if not SR_AVAILABLE or not PYAUDIO_AVAILABLE:
        return None  # Voice input not available on this platform

    recognizer = sr.Recognizer()
    recognizer.energy_threshold = 300
    recognizer.pause_threshold  = 0.8

    try:
        with sr.Microphone() as source:
            recognizer.adjust_for_ambient_noise(source, duration=1)
            try:
                audio = recognizer.listen(source, timeout=8, phrase_time_limit=15)
            except sr.WaitTimeoutError:
                return None

        lang_order = ["hi-IN", "en-IN"] if language == "auto" else [language]
        for lang_code in lang_order:
            try:
                text = recognizer.recognize_google(audio, language=lang_code)
                if text:
                    return text
            except (sr.UnknownValueError, sr.RequestError):
                continue

    except Exception as e:
        print(f"[STT Error] {e}")

    return None


# ─────────────────────────────────────────────
# Text → Speech
# ─────────────────────────────────────────────

def text_to_speech(text: str, lang: str = "en", slow: bool = False) -> str | None:
    """
    Convert text to speech using gTTS.
    Returns path to generated MP3 file, or None on failure.
    """
    try:
        gtts_lang = "hi" if lang == "hi" else "en"
        tts = gTTS(text=text, lang=gtts_lang, slow=slow)
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3")
        tts.save(tmp.name)
        return tmp.name
    except Exception as e:
        print(f"[TTS Error] {e}")
        return None
