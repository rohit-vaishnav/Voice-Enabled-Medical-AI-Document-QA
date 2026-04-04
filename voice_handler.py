"""
voice_handler.py
Speech-to-Text and Text-to-Speech for Hindi and English.
"""

import os
import tempfile
import speech_recognition as sr
from gtts import gTTS


# ─────────────────────────────────────────────
# Speech → Text
# ─────────────────────────────────────────────

def speech_to_text(language: str = "auto") -> str | None:
    """
    Record from microphone and transcribe.

    Args:
        language: "en-IN", "hi-IN", or "auto" (tries both).

    Returns:
        Transcribed string or None if failed.
    """
    recognizer = sr.Recognizer()
    recognizer.energy_threshold = 300
    recognizer.pause_threshold = 0.8

    with sr.Microphone() as source:
        recognizer.adjust_for_ambient_noise(source, duration=1)
        try:
            audio = recognizer.listen(source, timeout=8, phrase_time_limit=15)
        except sr.WaitTimeoutError:
            return None

    # Try Hindi first, then English
    lang_order = ["hi-IN", "en-IN"] if language == "auto" else [language]

    for lang_code in lang_order:
        try:
            text = recognizer.recognize_google(audio, language=lang_code)
            if text:
                return text
        except (sr.UnknownValueError, sr.RequestError):
            continue

    return None


# ─────────────────────────────────────────────
# Text → Speech
# ─────────────────────────────────────────────

def text_to_speech(text: str, lang: str = "en", slow: bool = False) -> str | None:
    """
    Convert text to speech using gTTS.

    Args:
        text: Text to speak.
        lang: "en" for English, "hi" for Hindi.
        slow: Speak slowly if True.

    Returns:
        Path to generated MP3 file.
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
