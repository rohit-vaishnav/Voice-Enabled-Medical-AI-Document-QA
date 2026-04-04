"""
medical_logic.py
Rule-based medical value range checker.
Parses extracted text for known lab parameters and raises alerts.
"""

import re

# ─────────────────────────────────────────────
# Normal Ranges (value: (min, max, unit))
# ─────────────────────────────────────────────
NORMAL_RANGES = {
    # Blood Sugar
    "glucose":          (70,   100,  "mg/dL"),
    "blood sugar":      (70,   100,  "mg/dL"),
    "fasting glucose":  (70,   100,  "mg/dL"),
    "hba1c":            (4.0,  5.6,  "%"),

    # Complete Blood Count
    "hemoglobin":       (12.0, 17.5, "g/dL"),
    "hgb":              (12.0, 17.5, "g/dL"),
    "wbc":              (4000, 11000,"cells/µL"),
    "platelets":        (150000, 400000, "cells/µL"),
    "rbc":              (4.5,  5.9,  "million/µL"),

    # Lipid Profile
    "cholesterol":      (0,    200,  "mg/dL"),
    "total cholesterol":(0,    200,  "mg/dL"),
    "ldl":              (0,    100,  "mg/dL"),
    "hdl":              (40,   60,   "mg/dL"),
    "triglycerides":    (0,    150,  "mg/dL"),

    # Kidney Function
    "creatinine":       (0.6,  1.2,  "mg/dL"),
    "urea":             (7,    20,   "mg/dL"),
    "bun":              (7,    20,   "mg/dL"),

    # Liver Function
    "sgpt":             (7,    56,   "U/L"),
    "alt":              (7,    56,   "U/L"),
    "sgot":             (10,   40,   "U/L"),
    "ast":              (10,   40,   "U/L"),
    "bilirubin":        (0.2,  1.2,  "mg/dL"),

    # Thyroid
    "tsh":              (0.4,  4.0,  "mIU/L"),
    "t3":               (80,   200,  "ng/dL"),
    "t4":               (5.0,  12.0, "µg/dL"),

    # Vitals
    "blood pressure":   (90,   120,  "mmHg"),
    "systolic":         (90,   120,  "mmHg"),
    "diastolic":        (60,   80,   "mmHg"),
    "heart rate":       (60,   100,  "bpm"),
    "pulse":            (60,   100,  "bpm"),
    "spo2":             (95,   100,  "%"),
    "oxygen":           (95,   100,  "%"),
}


def check_medical_values(text: str) -> list[str]:
    """
    Scan extracted text for medical values and compare against normal ranges.

    Args:
        text: Raw or processed text from the document.

    Returns:
        List of alert strings for out-of-range values.
    """
    alerts = []
    text_lower = text.lower()

    for param, (low, high, unit) in NORMAL_RANGES.items():
        # Find patterns like "Glucose: 130" or "Hemoglobin 9.5 g/dL"
        pattern = rf"{re.escape(param)}\s*[:\-=]?\s*(\d+\.?\d*)"
        matches = re.findall(pattern, text_lower)

        for match in matches:
            try:
                value = float(match)
                if value < low:
                    alerts.append(
                        f"⬇️ LOW {param.title()}: {value} {unit} "
                        f"(Normal: {low}–{high} {unit})"
                    )
                elif value > high:
                    alerts.append(
                        f"⬆️ HIGH {param.title()}: {value} {unit} "
                        f"(Normal: {low}–{high} {unit})"
                    )
            except ValueError:
                continue

    return alerts


def get_parameter_advice(param: str) -> str:
    """
    Return a brief clinical note for a given parameter.
    """
    advice_map = {
        "glucose": "High glucose may indicate diabetes. Consult your doctor.",
        "hemoglobin": "Low hemoglobin may indicate anemia. Ensure iron-rich diet.",
        "cholesterol": "High cholesterol increases heart disease risk. Consider dietary changes.",
        "creatinine": "High creatinine may indicate kidney dysfunction.",
        "tsh": "Abnormal TSH may indicate thyroid disorder.",
        "spo2": "Low SpO2 can be dangerous. Seek immediate medical attention.",
    }
    return advice_map.get(param.lower(), "Please consult a qualified physician.")
