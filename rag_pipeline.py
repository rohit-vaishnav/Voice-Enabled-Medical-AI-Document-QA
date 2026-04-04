"""
rag_pipeline.py
Two-stage Medical QA:
  Stage 1 — Direct regex extraction from PDF chunks
  Stage 2 — Claude AI enrichment (key read at call-time, not import-time)
  Stage 3 — Merged final answer
"""

import re
import os
import requests

ANTHROPIC_API_URL = "https://api.anthropic.com/v1/messages"
CLAUDE_MODEL      = "claude-haiku-4-5-20251001"

# ─────────────────────────────────────────────
# TEST REGISTRY  (search_terms, normal_display, unit, low, high)
# ─────────────────────────────────────────────
TEST_REGISTRY = {
    # CBC
    "hemoglobin":          (["hemoglobin"],              "13.0 – 16.5 g/dL",       "g/dL",       13.0, 16.5),
    "hb":                  (["hemoglobin"],              "13.0 – 16.5 g/dL",       "g/dL",       13.0, 16.5),
    "rbc":                 (["rbc count"],               "4.5 – 5.5 million/cmm",  "million",     4.5,  5.5),
    "rbc count":           (["rbc count"],               "4.5 – 5.5 million/cmm",  "million",     4.5,  5.5),
    "hematocrit":          (["hematocrit"],              "40 – 49 %",              "%",          40.0, 49.0),
    "mcv":                 (["mcv"],                     "83 – 101 fL",            "fL",         83.0,101.0),
    "mch":                 (["mch"],                     "27.1 – 32.5 pg",         "pg",         27.1, 32.5),
    "mchc":                (["mchc"],                    "32.5 – 36.7 g/dL",       "g/dL",       32.5, 36.7),
    "rdw":                 (["rdw"],                     "11.6 – 14 %",            "%",          11.6, 14.0),
    "wbc":                 (["wbc count"],               "4000 – 10000 /cmm",      "/cmm",       4000,10000),
    "wbc count":           (["wbc count"],               "4000 – 10000 /cmm",      "/cmm",       4000,10000),
    "white blood cell":    (["wbc count"],               "4000 – 10000 /cmm",      "/cmm",       4000,10000),
    "neutrophil":          (["neutrophils"],             "40 – 80 %",              "%",          40.0, 80.0),
    "lymphocyte":          (["lymphocytes"],             "20 – 40 %",              "%",          20.0, 40.0),
    "platelet":            (["platelet count"],          "150000 – 410000 /cmm",   "/cmm",     150000,410000),
    "platelet count":      (["platelet count"],          "150000 – 410000 /cmm",   "/cmm",     150000,410000),
    "mpv":                 (["mpv"],                     "7.5 – 10.3 fL",          "fL",          7.5, 10.3),
    "esr":                 (["esr"],                     "0 – 14 mm/1hr",          "mm/1hr",      0.0, 14.0),
    # Blood Sugar
    "glucose":             (["fasting blood sugar"],     "74 – 106 mg/dL",         "mg/dL",      74.0,106.0),
    "blood sugar":         (["fasting blood sugar"],     "74 – 106 mg/dL",         "mg/dL",      74.0,106.0),
    "fasting blood sugar": (["fasting blood sugar"],     "74 – 106 mg/dL",         "mg/dL",      74.0,106.0),
    "hba1c":               (["hba1c"],                   "< 5.7% Non-Diabetic | 5.7-6.4% Pre-Diabetic | >6.5% Diabetic", "%", 0.0, 5.6),
    "a1c":                 (["hba1c"],                   "< 5.7% Non-Diabetic | 5.7-6.4% Pre-Diabetic | >6.5% Diabetic", "%", 0.0, 5.6),
    # Lipid
    "cholesterol":         (["cholesterol"],             "< 200 mg/dL (Desirable)","mg/dL",       0.0,200.0),
    "triglyceride":        (["triglyceride"],            "< 150 mg/dL (Normal)",   "mg/dL",       0.0,150.0),
    "triglycerides":       (["triglyceride"],            "< 150 mg/dL (Normal)",   "mg/dL",       0.0,150.0),
    "hdl":                 (["hdl cholesterol"],         "40 – 60 mg/dL",          "mg/dL",      40.0, 60.0),
    "ldl":                 (["direct ldl"],              "< 100 mg/dL (Optimal)",  "mg/dL",       0.0,100.0),
    "vldl":                (["vldl"],                    "15 – 35 mg/dL",          "mg/dL",      15.0, 35.0),
    # Thyroid
    "tsh":                 (["tsh"],                     "0.35 – 4.94 microIU/mL", "microIU/mL",  0.35,4.94),
    "t3":                  (["t3"],                      "0.58 – 1.59 ng/mL",      "ng/mL",       0.58,1.59),
    "t4":                  (["t4"],                      "4.87 – 11.72 mg/mL",     "mg/mL",       4.87,11.72),
    # Kidney
    "creatinine":          (["creatinine"],              "0.66 – 1.25 mg/dL",      "mg/dL",       0.66,1.25),
    "urea":                (["urea"],                    "19.3 – 43.0 mg/dL",      "mg/dL",      19.3,43.0),
    "bun":                 (["blood urea nitrogen"],     "9.0 – 20.0 mg/dL",       "mg/dL",       9.0,20.0),
    "uric acid":           (["uric acid"],               "3.5 – 8.5 mg/dL",        "mg/dL",       3.5, 8.5),
    # Liver
    "sgpt":                (["sgpt"],                    "0 – 50 U/L",             "U/L",         0.0, 50.0),
    "alt":                 (["sgpt"],                    "0 – 50 U/L",             "U/L",         0.0, 50.0),
    "sgot":                (["sgot"],                    "17 – 59 U/L",            "U/L",        17.0, 59.0),
    "ast":                 (["sgot"],                    "17 – 59 U/L",            "U/L",        17.0, 59.0),
    "bilirubin":           (["total bilirubin"],         "0.2 – 1.3 mg/dL",        "mg/dL",       0.2,  1.3),
    "albumin":             (["albumin"],                 "3.5 – 5.0 g/dL",         "g/dL",        3.5,  5.0),
    "total protein":       (["total protein"],           "6.3 – 8.2 g/dL",         "g/dL",        6.3,  8.2),
    # Vitamins
    "vitamin d":           (["vitamin d","25(oh)"],      "30-100 ng/mL (Sufficient) | 10-30 (Insufficient) | <10 (Deficient)", "ng/mL", 30.0, 100.0),
    "vitamin b12":         (["vitamin b12"],             "187 – 833 pg/mL",        "pg/mL",     187.0,833.0),
    "b12":                 (["vitamin b12"],             "187 – 833 pg/mL",        "pg/mL",     187.0,833.0),
    # Others
    "homocysteine":        (["homocysteine"],            "6.0 – 14.8 micromol/L",  "micromol/L",  6.0,14.8),
    "ige":                 (["ige"],                     "0 – 87 IU/mL",           "IU/mL",       0.0, 87.0),
    "iron":                (["iron"],                    "49 – 181 micro g/dL",    "micro g/dL", 49.0,181.0),
    "psa":                 (["psa"],                     "0 – 4 ng/mL",            "ng/mL",       0.0,  4.0),
    "sodium":              (["sodium"],                  "136 – 145 mmol/L",       "mmol/L",    136.0,145.0),
    "potassium":           (["potassium"],               "3.5 – 5.1 mmol/L",       "mmol/L",      3.5,  5.1),
    "calcium":             (["calcium"],                 "8.4 – 10.2 mg/dL",       "mg/dL",       8.4,10.2),
    "microalbumin":        (["microalbumin"],            "< 16.7 mg/L",            "mg/L",        0.0,16.7),
}

EXPLANATIONS = {
    "hemoglobin":    "Hemoglobin carries oxygen in red blood cells. Low = anemia (fatigue, weakness). High = dehydration or blood disorder.",
    "wbc":           "WBC (White Blood Cells) fight infection. High = infection/inflammation. Low = weak immunity.",
    "wbc count":     "WBC (White Blood Cells) fight infection. High = infection/inflammation. Low = weak immunity.",
    "platelet":      "Platelets help blood clot. Low = easy bruising/bleeding. High = clot risk.",
    "platelet count":"Platelets help blood clot. Low = easy bruising/bleeding. High = clot risk.",
    "hba1c":         "HbA1c shows 3-month average blood sugar. >6.5% = diabetes. 5.7-6.4% = pre-diabetes. <5.7% = normal.",
    "glucose":       "Fasting Blood Sugar >106 mg/dL may indicate diabetes or pre-diabetes.",
    "blood sugar":   "Fasting Blood Sugar >106 mg/dL may indicate diabetes or pre-diabetes.",
    "cholesterol":   "Total Cholesterol >200 mg/dL raises heart disease risk. Diet and exercise help.",
    "ldl":           "LDL ('bad' cholesterol) builds in arteries. Keep below 100 mg/dL.",
    "hdl":           "HDL ('good' cholesterol) removes bad cholesterol. Higher is better.",
    "triglyceride":  "Triglycerides >150 increase heart and pancreas disease risk.",
    "triglycerides": "Triglycerides >150 increase heart and pancreas disease risk.",
    "tsh":           "High TSH = underactive thyroid. Low TSH = overactive thyroid.",
    "t3":            "T3 is the active thyroid hormone affecting metabolism, energy, and mood.",
    "t4":            "T4 is a thyroid hormone converted to active T3 in the body.",
    "creatinine":    "High creatinine indicates kidneys may not be filtering waste properly.",
    "urea":          "High urea = kidneys overworked or high protein diet. Low = poor nutrition or liver issue.",
    "sgpt":          "SGPT/ALT is a liver enzyme. High = liver cell damage (infection, alcohol, medication).",
    "sgot":          "SGOT/AST is found in liver and muscles. High = liver or muscle damage.",
    "bilirubin":     "High bilirubin causes jaundice. May indicate liver or bile duct issue.",
    "vitamin d":     "Vitamin D is essential for bones, immunity, and mood. <10 ng/mL = severe deficiency.",
    "vitamin b12":   "Vitamin B12 is essential for nerves and red blood cells. Low causes fatigue and numbness.",
    "b12":           "Vitamin B12 is essential for nerves and red blood cells. Low causes fatigue and numbness.",
    "homocysteine":  "High homocysteine (>14.8) is an independent risk factor for heart disease and stroke.",
    "ige":           "IgE >87 IU/mL strongly suggests allergies, asthma, or parasitic infection.",
    "iron":          "Iron is needed to make hemoglobin. Low iron leads to iron-deficiency anemia.",
    "psa":           "PSA >4 ng/mL may indicate prostate enlargement or cancer.",
    "esr":           "ESR measures inflammation. High = infection, autoimmune disease, or cancer.",
    "mpv":           "High MPV = large platelets, may suggest increased clotting activity.",
    "uric acid":     "High uric acid causes gout (painful joints). Reduce red meat and alcohol.",
    "sodium":        "Sodium controls fluid balance. Low = hyponatremia. High = dehydration or kidney issue.",
    "potassium":     "Potassium is vital for heart and muscle function. Abnormal levels need prompt attention.",
    "calcium":       "Calcium needed for bones and heart. High = hypercalcemia. Low = hypocalcemia.",
}


# ─────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────

def _get_api_key() -> str:
    """Read API key from env at call-time (NOT at import-time)."""
    return os.environ.get("ANTHROPIC_API_KEY", "").strip()


def _extract_value(search_terms: list, chunks: list):
    all_text   = " ".join(chunks)
    text_lower = all_text.lower()
    for term in search_terms:
        pattern = rf"{re.escape(term.lower())}\s*[:\-]?\s*(?:h|l)?\s*(?:<\s*)?(\d+\.?\d*)"
        m = re.search(pattern, text_lower)
        if m:
            try:
                return float(m.group(1))
            except ValueError:
                continue
    return None


def _status_label(value: float, low: float, high: float) -> str:
    if value < low:   return "⬇️ LOW"
    if value > high:  return "⬆️ HIGH"
    return "✅ NORMAL"


def _detect_intent(query: str) -> str:
    q = query.lower()
    if any(w in q for w in ["summary","all","overall","which test","abnormal","report","everything","full"]):
        return "summary"
    if any(w in q for w in ["what does","explain","why","meaning","dangerous","serious","what happen","tell me about"]):
        return "explain"
    if any(w in q for w in ["normal range","reference range","should be","healthy range"]):
        return "range"
    return "value"


def _find_test(query: str):
    q = query.lower()
    for key in sorted(TEST_REGISTRY.keys(), key=len, reverse=True):
        if key in q:
            return key
    return None


# ─────────────────────────────────────────────
# Stage 1 — PDF extraction
# ─────────────────────────────────────────────

def _pdf_extract_single(test_key: str, chunks: list) -> dict:
    search_terms, normal_display, unit, low, high = TEST_REGISTRY[test_key]
    value = _extract_value(search_terms, chunks)
    return {
        "test_key":       test_key,
        "value":          value,
        "unit":           unit,
        "normal_display": normal_display,
        "status":         _status_label(value, low, high) if value is not None else None,
        "explanation":    EXPLANATIONS.get(test_key, ""),
        "low": low, "high": high,
    }


def _pdf_extract_summary(chunks: list) -> dict:
    abnormal, normal = [], []
    checked = set()
    for test_key, (search_terms, normal_display, unit, low, high) in TEST_REGISTRY.items():
        canonical = search_terms[0]
        if canonical in checked:
            continue
        checked.add(canonical)
        value = _extract_value(search_terms, chunks)
        if value is not None:
            status = _status_label(value, low, high)
            entry  = {"test": canonical.title(), "value": value, "unit": unit, "status": status}
            (abnormal if ("HIGH" in status or "LOW" in status) else normal).append(entry)
    return {"abnormal": abnormal, "normal": normal}


# ─────────────────────────────────────────────
# Stage 2 — Claude AI enrichment
# ─────────────────────────────────────────────

def _call_claude(system_prompt: str, user_prompt: str) -> str:
    """Call Claude API — reads key fresh from os.environ every time."""
    api_key = _get_api_key()
    if not api_key:
        return ""
    try:
        headers = {
            "Content-Type":      "application/json",
            "x-api-key":         api_key,
            "anthropic-version": "2023-06-01",
        }
        payload = {
            "model":    CLAUDE_MODEL,
            "max_tokens": 512,
            "system":   system_prompt,
            "messages": [{"role": "user", "content": user_prompt}],
        }
        resp = requests.post(ANTHROPIC_API_URL, headers=headers, json=payload, timeout=20)
        resp.raise_for_status()
        data = resp.json()
        if "content" in data:
            return "".join(b.get("text", "") for b in data["content"]).strip()
        if "error" in data:
            print(f"[Claude API Error] {data['error']}")
    except requests.exceptions.HTTPError as e:
        print(f"[Claude HTTP {e.response.status_code}] {e.response.text[:300]}")
    except Exception as e:
        print(f"[Claude Error] {e}")
    return ""


def _ai_enrich_single(pdf_result: dict, query: str, chat_history: str, chunks: list) -> str:
    system = (
        "You are a helpful, empathetic medical AI assistant. "
        "You have a patient's lab result plus the raw text of their medical report sections.\n"
        "Your job:\n"
        "1. If a value was extracted, give 2-3 sentences of practical clinical advice — "
        "what it means for daily health, any risks, and what the patient should do.\n"
        "2. If the value was NOT found in the extracted sections, search the raw report text "
        "and try to find and interpret it yourself.\n"
        "3. Be empathetic and always recommend consulting a doctor.\n"
        "4. Do NOT repeat the numeric value or range — those are shown separately.\n"
        "5. Keep response under 100 words. Plain text, no markdown symbols."
    )
    raw_ctx    = "\n".join(chunks[:5])
    value_line = (
        f"Extracted — Test: {pdf_result['test_key'].title()}, "
        f"Value: {pdf_result['value']} {pdf_result['unit']}, "
        f"Status: {pdf_result['status']}, Range: {pdf_result['normal_display']}"
        if pdf_result["value"] is not None
        else f"Value NOT found in retrieved sections for: {pdf_result['test_key'].title()}"
    )
    history_part = f"\nConversation history:\n{chat_history}" if chat_history else ""
    user_prompt  = f"{value_line}\n\nRaw report text:\n{raw_ctx}{history_part}\n\nPatient question: {query}"
    return _call_claude(system, user_prompt)


def _ai_enrich_summary(pdf_summary: dict, query: str, chunks: list) -> str:
    system = (
        "You are a helpful medical AI assistant. "
        "Given a patient's complete lab summary and report context, write a clear overall "
        "health assessment in 4-5 sentences. Highlight the most urgent abnormal values first, "
        "then note what is healthy. Give one specific lifestyle recommendation. "
        "Always recommend seeing a doctor. Plain text only."
    )
    raw_ctx       = "\n".join(chunks[:6])
    abnormal_list = ", ".join(f"{e['test']} ({e['value']} {e['unit']} — {e['status']})" for e in pdf_summary["abnormal"]) or "None detected"
    normal_list   = ", ".join(e["test"] for e in pdf_summary["normal"]) or "None detected"
    user_prompt   = (
        f"Abnormal values: {abnormal_list}\n"
        f"Normal values: {normal_list}\n\n"
        f"Report sections:\n{raw_ctx}\n\n"
        f"Patient question: {query}"
    )
    return _call_claude(system, user_prompt)


def _ai_general_answer(query: str, chunks: list, chat_history: str) -> str:
    system = (
        "You are a knowledgeable medical AI assistant. "
        "Answer the patient's question using ONLY the medical report context provided. "
        "Be direct and accurate. Keep answer under 120 words. "
        "If the answer cannot be found in the context, say so honestly and suggest they ask their doctor. "
        "Plain text only."
    )
    context      = "\n".join(chunks[:5]) if chunks else "No report context available."
    history_part = f"\nConversation history:\n{chat_history}" if chat_history else ""
    user_prompt  = f"Medical report context:\n{context}{history_part}\n\nPatient question: {query}"
    return _call_claude(system, user_prompt)


# ─────────────────────────────────────────────
# Stage 3 — Merge & format
# ─────────────────────────────────────────────

def _format_single(pdf: dict, ai: str) -> str:
    lines = [f"🔬 **{pdf['test_key'].title()}**\n"]
    if pdf["value"] is not None:
        lines.append(f"📌 **Your Value (from report):** {pdf['value']} {pdf['unit']}")
        lines.append(f"📋 **Normal Range:** {pdf['normal_display']}")
        lines.append(f"📊 **Status:** {pdf['status']}")
    else:
        lines.append(f"📋 **Normal Range:** {pdf['normal_display']}")
        lines.append("⚠️ *Exact value not found in the retrieved report sections. Try increasing Top-K in settings.*")
    if pdf["explanation"]:
        lines.append(f"\n💡 **What this test measures:**\n{pdf['explanation']}")
    if ai:
        lines.append(f"\n🤖 **AI Health Advice:**\n{ai}")
    lines.append("\n\n*⚕️ Always consult a qualified physician for medical decisions.*")
    return "\n".join(lines)


def _format_summary(pdf: dict, ai: str) -> str:
    lines = ["📊 **Complete Report Summary**\n"]
    if pdf["abnormal"]:
        lines.append("🔴 **Values Needing Attention:**")
        for e in pdf["abnormal"]:
            lines.append(f"  • {e['test']}: {e['value']} {e['unit']} → {e['status']}")
    else:
        lines.append("✅ No abnormal values detected in retrieved sections.")
    if pdf["normal"]:
        lines.append("\n✅ **Normal Values:**")
        for e in pdf["normal"]:
            lines.append(f"  • {e['test']}: {e['value']} {e['unit']} → {e['status']}")
    if ai:
        lines.append(f"\n🤖 **AI Overall Assessment:**\n{ai}")
    lines.append("\n*⚕️ Please consult your doctor for full interpretation.*")
    return "\n".join(lines)


# ─────────────────────────────────────────────
# Public entry point
# ─────────────────────────────────────────────

def generate_answer(query: str, context_chunks: list, chat_history: str = "", max_new_tokens: int = 300) -> str:
    intent   = _detect_intent(query)
    test_key = _find_test(query)

    if intent == "summary":
        pdf = _pdf_extract_summary(context_chunks)
        ai  = _ai_enrich_summary(pdf, query, context_chunks)
        return _format_summary(pdf, ai)

    if test_key:
        pdf = _pdf_extract_single(test_key, context_chunks)
        ai  = _ai_enrich_single(pdf, query, chat_history, context_chunks)
        return _format_single(pdf, ai)

    ai_answer = _ai_general_answer(query, context_chunks, chat_history)
    if ai_answer:
        return (
            f"🤖 **AI Medical Assistant:**\n\n{ai_answer}\n\n"
            "💡 *For specific test values, try:*\n"
            "• 'What is my WBC count?'\n"
            "• 'Is my HbA1c normal?'\n"
            "• 'Give me a summary of my report'"
        )
    return (
        "❓ Could not identify a specific test in your question.\n\nTry asking:\n"
        "• 'What is my WBC count?'\n"
        "• 'Is my HbA1c normal?'\n"
        "• 'Explain my Vitamin D result'\n"
        "• 'Give me a summary of my report'"
    )
