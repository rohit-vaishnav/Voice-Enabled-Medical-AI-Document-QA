"""
language_detector.py
Detect Hindi vs English and perform basic translation using googletrans.
"""

from langdetect import detect, LangDetectException
from deep_translator import GoogleTranslator


def detect_language(text: str) -> str:
    """
    Detect whether text is Hindi ("hi") or English ("en").

    Args:
        text: Input string.

    Returns:
        "hi" or "en"
    """
    try:
        lang = detect(text)
        return "hi" if lang == "hi" else "en"
    except LangDetectException:
        return "en"


def translate_to_english(text: str, source_lang: str = "auto") -> str:
    """
    Translate text to English.

    Args:
        text: Text to translate.
        source_lang: Source language code or "auto".

    Returns:
        Translated English string.
    """
    if source_lang == "en":
        return text
    try:
        src = "auto" if source_lang == "auto" else source_lang
        translated = GoogleTranslator(source=src, target="en").translate(text)
        return translated or text
    except Exception as e:
        print(f"[Translation Error] {e}")
        return text


def translate_to_hindi(text: str) -> str:
    """
    Translate English text to Hindi.

    Args:
        text: English text.

    Returns:
        Hindi string.
    """
    try:
        translated = GoogleTranslator(source="en", target="hi").translate(text)
        return translated or text
    except Exception as e:
        print(f"[Translation Error] {e}")
        return text
