"""
rag_pipeline.py
- Sends FULL document text to AI (not just top-K chunks) — fixes wrong values
- Uses Groq API (FREE, fast) as primary + Anthropic as fallback
- Supports: report Q&A, summary, medicine explainer, health chatbot
"""

import os
import re
import requests

# ── API Config ─────────────────────────────────────────────────────────────
# GROQ — Free API (get key at console.groq.com)
GROQ_API_URL  = "https://api.groq.com/openai/v1/chat/completions"
GROQ_MODEL    = "llama-3.3-70b-versatile"   # free, very fast, very smart

# Anthropic — fallback if Groq key not set
ANTHROPIC_API_URL = "https://api.anthropic.com/v1/messages"
ANTHROPIC_MODEL   = "claude-haiku-4-5-20251001"

# Full PDF text stored at module level so all functions can use it
_FULL_DOC_TEXT = ""


def set_full_document(text: str):
    """Called once after PDF is processed — stores full text for all queries."""
    global _FULL_DOC_TEXT
    _FULL_DOC_TEXT = text[:15000]   # ~15KB is enough for any lab report


def get_full_document() -> str:
    return _FULL_DOC_TEXT


# ── API Key helpers ─────────────────────────────────────────────────────────

def _get_groq_key() -> str:
    return os.environ.get("GROQ_API_KEY", "").strip()

def _get_anthropic_key() -> str:
    return os.environ.get("ANTHROPIC_API_KEY", "").strip()


# ── Core AI call — tries Groq first, falls back to Anthropic ────────────────

def _call_ai(system: str, user: str, max_tokens: int = 700) -> str:
    """
    Try Groq first (free), fall back to Anthropic (paid).
    Both keys are read fresh on every call.
    """
    groq_key      = _get_groq_key()
    anthropic_key = _get_anthropic_key()

    # ── Try Groq (FREE) ──
    if groq_key:
        try:
            resp = requests.post(
                GROQ_API_URL,
                headers={
                    "Content-Type":  "application/json",
                    "Authorization": f"Bearer {groq_key}",
                },
                json={
                    "model":       GROQ_MODEL,
                    "max_tokens":  max_tokens,
                    "messages": [
                        {"role": "system", "content": system},
                        {"role": "user",   "content": user},
                    ],
                },
                timeout=30,
            )
            resp.raise_for_status()
            data = resp.json()
            text = data["choices"][0]["message"]["content"].strip()
            if text:
                return text
        except requests.exceptions.HTTPError as e:
            print(f"[Groq HTTP {e.response.status_code}] {e.response.text[:200]}")
        except Exception as e:
            print(f"[Groq Error] {e}")

    # ── Fall back to Anthropic ──
    if anthropic_key:
        try:
            resp = requests.post(
                ANTHROPIC_API_URL,
                headers={
                    "Content-Type":      "application/json",
                    "x-api-key":         anthropic_key,
                    "anthropic-version": "2023-06-01",
                },
                json={
                    "model":      ANTHROPIC_MODEL,
                    "max_tokens": max_tokens,
                    "system":     system,
                    "messages":   [{"role": "user", "content": user}],
                },
                timeout=30,
            )
            resp.raise_for_status()
            data = resp.json()
            if "content" in data:
                return "".join(b.get("text", "") for b in data["content"]).strip()
        except requests.exceptions.HTTPError as e:
            print(f"[Anthropic HTTP {e.response.status_code}] {e.response.text[:200]}")
        except Exception as e:
            print(f"[Anthropic Error] {e}")

    return ""


# ── Intent detection ────────────────────────────────────────────────────────

# Lab test names — these should ALWAYS go to "report", never "medicine"
LAB_TEST_TERMS = {
    "wbc","rbc","hemoglobin","hb","hematocrit","mcv","mch","mchc","rdw",
    "platelet","mpv","esr","neutrophil","lymphocyte","monocyte","eosinophil",
    "basophil","leucocyte","tlc","dlc","cbc","blood count",
    "glucose","blood sugar","fasting","hba1c","a1c","sugar",
    "cholesterol","triglyceride","hdl","ldl","vldl","lipid",
    "tsh","t3","t4","thyroid",
    "creatinine","urea","bun","uric acid","kidney",
    "sgpt","sgot","alt","ast","bilirubin","albumin","liver","protein",
    "vitamin d","vitamin b12","b12","folate","ferritin","iron",
    "sodium","potassium","calcium","chloride","electrolyte",
    "ige","psa","homocysteine","microalbumin","crp","esr",
    "test","report","result","value","count","level","range",
    "normal","high","low","abnormal","my report","my test",
    "bleeding time","clotting time","pt","inr","aptt",
}

# Medicine drug names — these should ALWAYS go to "medicine"
MEDICINE_TERMS = {
    "tablet","capsule","syrup","injection","drop","ointment","cream","gel",
    "medicine","medication","drug","pill","dose","dosage","mg","ml",
    "prescribed","prescription","side effect","side effects",
    "paracetamol","ibuprofen","metformin","aspirin","amoxicillin","azithromycin",
    "cetirizine","dolo","crocin","combiflam","allegra","montair","telma",
    "pan 40","omeprazole","pantoprazole","amlodipine","atorvastatin",
    "losartan","ramipril","insulin","steroid","antibiotic","antifungal",
    "c-tin","when to take","how to take","can i take","should i take",
    "stop taking","continue taking","how many tablets","how many capsules",
}

def _detect_intent(query: str) -> str:
    q = query.lower().strip()

    # 1. Summary — always first
    if any(w in q for w in [
        "summary","all test","overall","abnormal","everything",
        "full report","complete report","all values","all results",
    ]):
        return "summary"

    # 2. Food/diet/lifestyle — check before lab terms
    #    "what food for low hemoglobin" should be health_chat not report
    FOOD_TERMS = [
        "food","eat","diet","drink","avoid","recover","recovery",
        "exercise","workout","yoga","lifestyle","sleep","stress",
        "how to improve","what should i","can i eat","what to eat",
        "what not to eat","which fruit","vegetable","recipe",
        "home remedy","natural","cure","nutrition","what can i eat",
        "which food","best food","good food","bad food",
    ]
    if any(w in q for w in FOOD_TERMS):
        return "health_chat"

    # 3. Lab test — if ANY lab term found, it is a report question
    if any(term in q for term in LAB_TEST_TERMS):
        return "report"

    # 4. Medicine — only if no lab/food term matched above
    if any(term in q for term in MEDICINE_TERMS):
        return "medicine"

    # 5. Short query (1-3 meaningful words) with no lab/food context
    #    → likely a medicine brand name like "C-Tin Plus" or "Dolo 650"
    words = [w for w in q.split() if len(w) > 2]
    if 1 <= len(words) <= 3:
        return "medicine"

    # 6. Default — treat as report question
    return "report"


# ── Answer: Report Q&A ──────────────────────────────────────────────────────

def _answer_report(query: str, chat_history: str) -> str:
    doc = get_full_document()

    system = """You are an expert medical AI assistant reading a patient's lab report.
You will be given the COMPLETE raw text extracted from their medical PDF.

STRICT RULES:
1. Scan the ENTIRE report text carefully. Find the EXACT numeric value for what the patient asks.
2. Lab reports have many formats — values may appear as: "14.5", "14.5 g/dL", "H 14.5", "14.5 H", "14.5*"
3. Always respond with this exact format:

📌 Your Value: [exact number] [unit]
📋 Normal Range: [range]
📊 Status: [HIGH / LOW / NORMAL] — explain briefly why
💡 What this means: [2-3 sentences in simple language]
🥗 Diet & Recovery Tips:
   • [specific food to eat]
   • [specific food to avoid]
   • [one lifestyle tip]
⚕️ See a doctor if: [specific warning sign]

4. If the test genuinely does not exist in the report, say exactly:
   "This test (X) was not found in your uploaded report."
5. NEVER guess or make up values. Only report what is in the text.
6. LANGUAGE RULE: Always respond in English. Only use Hindi if the patient wrote in Hindi script (Devanagari). Never use Chinese or any other language."""

    history_part = f"\nPrevious conversation:\n{chat_history}" if chat_history else ""
    user = f"""COMPLETE LAB REPORT TEXT:
{doc}
{history_part}

PATIENT QUESTION: {query}
[IMPORTANT: Always respond in English only]"""

    result = _call_ai(system, user, max_tokens=700)
    if result:
        return result
    return (
        "⚠️ No API key found.\n\n"
        "Add your **Groq API key** (free) or Anthropic API key in the sidebar.\n\n"
        "Get free Groq key at: https://console.groq.com\n\n"
        "**Preview of extracted report text:**\n```\n" + doc[:600] + "\n```"
    )


# ── Answer: Full Summary ────────────────────────────────────────────────────

def _answer_summary() -> str:
    doc = get_full_document()

    system = """You are an expert medical AI. Read this complete lab report and give a full health summary.

Format EXACTLY as:

📊 COMPLETE REPORT SUMMARY
━━━━━━━━━━━━━━━━━━━━━━━━

🔴 ABNORMAL VALUES (Need Attention):
[For each abnormal test:]
• Test Name: [value] [unit] → [HIGH/LOW]
  ↳ [one sentence what this means]

✅ NORMAL VALUES:
[For each normal test:]
• Test Name: [value] [unit] → NORMAL

🥗 TOP RECOMMENDATIONS:
• [diet tip 1 based on abnormal values]
• [diet tip 2]
• [lifestyle tip]

⚕️ SEE DOCTOR URGENTLY IF:
• [specific critical values that need immediate attention]

Keep language simple. Be thorough — include every test value you find.
IMPORTANT: Always respond in English only."""

    user = f"COMPLETE LAB REPORT:\n{doc}"

    result = _call_ai(system, user, max_tokens=1000)
    if result:
        return result
    return "⚠️ API key required. Add Groq (free) or Anthropic key in the sidebar."


# ── Answer: Medicine Explainer ──────────────────────────────────────────────

def _answer_medicine(query: str, chat_history: str) -> str:
    doc = get_full_document()

    system = """You are an expert clinical pharmacist AI. A patient is asking about a medicine.
Give a DETAILED, SPECIFIC, and CLEAR explanation. Be like a doctor explaining to a patient face-to-face.

IMPORTANT RULES:
1. Be SPECIFIC — give exact mg doses, exact timings, exact durations. Not vague like "as directed".
2. Use SIMPLE language — patient may have no medical knowledge.
3. Be THOROUGH — cover every section fully with real details.
4. If it is a combination medicine (like C-Tin Plus), identify ALL ingredients and explain each one.
5. PERSONALISE — if patient report is provided, relate the medicine to their specific test values.
6. Reply in Hindi if the patient wrote in Hindi.

USE THIS EXACT FORMAT WITH CLEAR SEPARATORS:

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
💊 MEDICINE: [Full medicine name]
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

🔬 WHAT IS THIS MEDICINE?
[2-3 sentences clearly explaining what this medicine is, what drug class it belongs to, 
and what health problem it is designed to solve. Be specific.]

🧪 ACTIVE INGREDIENTS:
[List every ingredient with its purpose]
• [Ingredient 1] ([amount]mg) — [what it does in the body]
• [Ingredient 2] ([amount]mg) — [what it does in the body]
• [Ingredient 3] ([amount]mg) — [what it does in the body]

✅ WHEN IS THIS MEDICINE GIVEN? (USES)
[List specific conditions it treats — be specific, not generic]
• [Condition 1] — [explain briefly why this medicine helps for this condition]
• [Condition 2] — [explain briefly]
• [Condition 3] — [explain briefly]

📏 HOW TO TAKE THIS MEDICINE (DOSAGE GUIDE)
┌─────────────────────────────────────────┐
  Adults      : [exact dose, e.g. 1 tablet twice daily]
  Children    : [dose by age/weight if applicable, or "not recommended under X years"]
  Timing      : [Before food / After food / With food — and WHY]
  Best time   : [Morning / Night / Specific time — and WHY]
  Duration    : [e.g. 5 days / 2 weeks / as prescribed — typical course]
  Maximum dose: [Do not exceed X tablets/day]
└─────────────────────────────────────────┘

💡 HOW TO TAKE IT CORRECTLY:
• [Step 1 — e.g. "Swallow whole with a full glass of water — do not crush or chew"]
• [Step 2 — e.g. "Take at same time every day for best effect"]
• [Step 3 — e.g. "If you miss a dose, take as soon as you remember unless next dose is in 2 hours"]
• [Step 4 — e.g. "Do not double the dose to make up for a missed one"]

⚠️ IMPORTANT WARNINGS & CAUTIONS:
• ⚠️ [Warning 1 — be specific and explain WHY it matters]
• ⚠️ [Warning 2]
• ⚠️ [Warning 3]
• ⚠️ [Warning 4 if applicable]

🚫 DO NOT TAKE THIS MEDICINE IF:
[Be specific — these are situations where this medicine is dangerous]
• ❌ [Contraindication 1] — [why it is dangerous]
• ❌ [Contraindication 2] — [why it is dangerous]
• ❌ [Contraindication 3] — [why it is dangerous]

😣 SIDE EFFECTS:
COMMON (may happen in 1 in 10 people):
• [Side effect] — [what to do if it happens]
• [Side effect] — [what to do if it happens]
• [Side effect] — [what to do if it happens]

RARE BUT SERIOUS (stop medicine and call doctor):
• [Serious side effect] — [symptoms to watch for]
• [Serious side effect] — [symptoms to watch for]

🥗 FOOD & DRINK RULES:
• ✅ TAKE WITH: [foods/drinks that help absorption or reduce side effects]
• 🚫 AVOID: [specific foods/drinks that interfere — and exactly why]
• 🚫 AVOID: [alcohol? specific juices? — explain effect]

💊 MEDICINE INTERACTIONS:
[Medicines that should NOT be taken with this one]
• ❌ [Medicine name] — [what happens if taken together]
• ❌ [Medicine name] — [what happens if taken together]
• ⚠️ [Medicine name] — [use with caution, tell your doctor]

🚨 GO TO HOSPITAL IMMEDIATELY IF YOU NOTICE:
• 🆘 [Emergency symptom 1]
• 🆘 [Emergency symptom 2]
• 🆘 [Emergency symptom 3]

🏥 STORAGE:
• Store at [temperature] in [conditions]
• Keep away from [light/moisture/children]
• Discard if [expiry/color change/etc.]

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
⚕️ DOCTOR'S REMINDER: This information is for education only. 
Always take medicines exactly as your doctor prescribed.
Never change your dose or stop taking without consulting your doctor.
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"""

    history_part = f"\nPrevious conversation:\n{chat_history}" if chat_history else ""
    report_context = f"\nPatient lab report (for personalised advice):\n{doc[:3000]}" if doc else ""
    user = f"{report_context}{history_part}\n\nPatient question: {query}\n\n[IMPORTANT: Respond in English only. Do not use Chinese, Spanish or any other language.]"

    result = _call_ai(system, user, max_tokens=1200)
    if result:
        return result
    return "⚠️ API key required. Add Groq (free) or Anthropic key in the sidebar."


# ── Answer: Health Chatbot ──────────────────────────────────────────────────

def _answer_health_chat(query: str, chat_history: str) -> str:
    doc = get_full_document()

    system = """You are a friendly, expert health and wellness AI assistant.
You help patients with diet, food, recovery, lifestyle, and general health questions.
You have the patient's lab report for personalised advice.

FORMAT:
🤖 HEALTH ASSISTANT

[Answer the question directly first in 1-2 sentences]

🥗 FOODS TO EAT:
• [specific food] — [why it helps]
• [specific food] — [why it helps]
• [specific food] — [why it helps]

🚫 FOODS TO AVOID:
• [food to avoid] — [why]
• [food to avoid] — [why]

💪 LIFESTYLE TIPS:
• [tip 1]
• [tip 2]

⚕️ Always consult your doctor before making major diet or lifestyle changes.

Be warm, encouraging, and specific. Personalise advice using report values if available.
LANGUAGE RULE:
- ALWAYS respond in English by default.
- ONLY switch to Hindi if the patient's question contains Hindi script (Devanagari characters like अ, ब, क etc.)
- NEVER respond in Chinese, Spanish, French or any other language.
- If unsure, always use English."""

    history_part = f"\nPrevious conversation:\n{chat_history}" if chat_history else ""
    report_context = f"\nPatient's report:\n{doc[:3000]}" if doc else ""
    user = f"{report_context}{history_part}\n\nQuestion: {query}"

    result = _call_ai(system, user, max_tokens=700)
    if result:
        return f"🤖 **Health Assistant:**\n\n{result}"
    return "⚠️ API key required. Add Groq (free) or Anthropic key in the sidebar."


# ── Debug helper ─────────────────────────────────────────────────────────────

def debug_chunks(chunks: list) -> str:
    doc = get_full_document()
    out = ["=== FULL DOCUMENT TEXT (first 2000 chars) ===\n"]
    out.append(doc[:2000])
    out.append("\n\n=== TOP RETRIEVED CHUNKS ===")
    for i, c in enumerate(chunks[:3]):
        out.append(f"\n--- Chunk {i+1} ---\n{c[:300]}")
    return "\n".join(out)


# ── Public entry point ────────────────────────────────────────────────────────

def generate_answer(
    query: str,
    context_chunks: list,
    chat_history: str = "",
    max_new_tokens: int = 300,
) -> str:
    intent = _detect_intent(query)

    if intent == "summary":
        return _answer_summary()
    if intent == "medicine":
        return _answer_medicine(query, chat_history)
    if intent == "health_chat":
        return _answer_health_chat(query, chat_history)

    return _answer_report(query, chat_history)
