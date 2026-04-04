# 🏥 Voice-Enabled Multilingual Medical Document QA System

> NLP + RAG + Claude AI powered system for querying medical reports via Hindi/English voice or text.
> **New: Two-stage answer generation — PDF extraction merged with Claude AI chatbot enrichment.**

---

## 📁 Project Structure

```
medical_rag/
├── app.py                  # Streamlit UI — 3D colorful premium interface
├── document_processor.py   # PDF extraction + text cleaning
├── chunker.py              # Text chunking (LangChain)
├── embedder.py             # Sentence embeddings (MiniLM)
├── vector_store.py         # FAISS index build + search
├── voice_handler.py        # STT (SpeechRecognition) + TTS (gTTS)
├── language_detector.py    # Language detection + translation
├── rag_pipeline.py         # ★ Two-stage: PDF extraction + Claude AI merge
├── medical_logic.py        # Rule-based medical range checker
├── chat_history.py         # Multi-turn conversation manager
├── qr_generator.py         # QR code generator
└── requirements.txt
```

---

## ⚙️ Installation

### 1. Create virtual environment
```bash
python -m venv venv
source venv/bin/activate        # Linux/Mac
venv\Scripts\activate           # Windows
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Install PyAudio (for microphone input)
```bash
# Ubuntu/Debian
sudo apt-get install portaudio19-dev
pip install pyaudio

# macOS
brew install portaudio
pip install pyaudio

# Windows
pip install pipwin
pipwin install pyaudio
```

### 4. Set Anthropic API Key (for AI chatbot enrichment)
```bash
# Option A: environment variable (recommended)
export ANTHROPIC_API_KEY="sk-ant-..."

# Option B: enter in the sidebar of the app
# Settings → AI Chatbot → Anthropic API Key field
```

---

## 🚀 Running the App

```bash
streamlit run app.py
```

Open `http://localhost:8501` in your browser.

---

## 🔄 Two-Stage Answer Architecture (NEW)

```
User Question
    │
    ▼
Stage 1: PDF Extraction
    ├─ FAISS semantic search → top-K chunks
    ├─ Keyword fallback retrieval
    ├─ Regex value extraction (e.g. "WBC Count: 10570 /cmm")
    └─ Status: NORMAL / HIGH / LOW

    │
    ▼
Stage 2: Claude AI Enrichment
    ├─ Receives: extracted value + status + normal range + patient question
    ├─ Generates: 2-3 sentences of clinical context & practical advice
    └─ Model: claude-haiku (fast, cost-effective)

    │
    ▼
Stage 3: Merged Final Answer
    ├─ 📄 PDF Extracted facts (value, range, status)
    ├─ 🤖 AI insight (explanation, lifestyle tips, urgency)
    └─ Displayed with source badges in chat bubble
```

---

## 📊 Full System Workflow

```
User → QR Scan → Web App
  → Upload PDF → Extract Text → Clean → Chunk → Embed → FAISS Index
  → Voice/Text Query
  → Language Detect → Translate to English
  → Embed Query → Retrieve Top-K Chunks (semantic + keyword)
  → Stage 1: Regex extract value → status label
  → Stage 2: Claude AI enrichment call
  → Stage 3: Merge both → final answer
  → Medical Logic (Range Checks → Alerts)
  → Translate to Hindi if needed
  → Text Display (chat bubble with PDF + AI badges) + Voice Output
```

---

## 🧠 Models Used

| Component         | Model / Tool                        |
|-------------------|-------------------------------------|
| Embeddings        | `all-MiniLM-L6-v2`                  |
| AI Enrichment     | `claude-haiku-4-5` (Anthropic API)  |
| Speech → Text     | Google Speech Recognition API       |
| Text → Speech     | gTTS (Google Text-to-Speech)        |
| Translation       | deep-translator (Google Translate)  |
| Vector Search     | FAISS (flat L2 index)               |

---

## ⚠️ Medical Alerts

Auto-detected out-of-range values for:
- Blood Sugar (Glucose, HbA1c)
- CBC (Hemoglobin, WBC, Platelets, RBC, MCV, MCH...)
- Lipid Profile (Cholesterol, LDL, HDL, Triglycerides)
- Kidney Function (Creatinine, Urea, BUN, Uric Acid)
- Liver Function (SGPT, SGOT, Bilirubin, Albumin)
- Thyroid (TSH, T3, T4)
- Vitamins (D, B12)
- Others (IgE, Homocysteine, Iron, PSA, Sodium, Potassium...)

---

## 🗨️ Sample Questions

- `What is my WBC count?`
- `Is my HbA1c normal?`
- `Explain my Vitamin D result`
- `Give me a summary of my report`
- `मेरा हीमोग्लोबिन कैसा है?` (Hindi)
- `What does high cholesterol mean?`
- `Which values are abnormal?`

---

## 📌 Notes

- AI enrichment requires an Anthropic API key — works without it (PDF extraction only)
- Internet required for Google STT, gTTS, translation, and Claude API
- This is a research/educational system — not a substitute for medical advice
