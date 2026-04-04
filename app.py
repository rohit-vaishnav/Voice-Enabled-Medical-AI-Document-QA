import streamlit as st
import os
import tempfile

from document_processor import extract_text_from_pdf, preprocess_text
from chunker import chunk_text
from embedder import generate_embeddings, embed_query
from vector_store import build_faiss_index, search_faiss
from voice_handler import speech_to_text, text_to_speech
from language_detector import detect_language, translate_to_english
from rag_pipeline import generate_answer
from medical_logic import check_medical_values
from chat_history import ChatHistory
from qr_generator import generate_qr_code

# ─────────────────────────────────────────────
# Page config
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="🏥 Medical QA System",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─────────────────────────────────────────────
# Session state
# ─────────────────────────────────────────────
defaults = {
    "chat_history":   ChatHistory(),
    "faiss_index":    None,
    "chunks":         [],
    "doc_processed":  False,
    "audio_files":    [],      # list of tts file paths, one per turn
    "last_query":     "",      # prevent re-processing same query on rerun
}
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

# ─────────────────────────────────────────────
# CSS
# ─────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

.hero-header {
    background: linear-gradient(135deg,#0a1628 0%,#0d2b4e 40%,#0a3d62 70%,#1a6b8a 100%);
    padding: 32px 28px; border-radius: 20px; text-align: center;
    margin-bottom: 24px; position: relative; overflow: hidden;
    box-shadow: 0 16px 48px rgba(0,100,200,0.3);
}
.hero-header h1 { font-size:2rem; font-weight:700; color:#fff !important; margin:0 0 6px; }
.hero-header h1 span { color:#22d3ee; }
.hero-header p { color:#94a3b8; font-size:0.92rem; margin:0 0 16px; }
.hero-badges { display:flex; gap:10px; justify-content:center; flex-wrap:wrap; }
.hero-badge {
    background:rgba(255,255,255,0.08); border:1px solid rgba(255,255,255,0.15);
    color:#7dd3fc; padding:5px 14px; border-radius:20px; font-size:0.75rem; font-weight:500;
}

.feat-grid { display:grid; grid-template-columns:repeat(4,1fr); gap:12px; margin-bottom:24px; }
.feat-card { border-radius:14px; padding:18px 14px; color:white; transition:transform .2s; }
.feat-card:hover { transform:translateY(-4px); }
.feat-card.blue   { background:linear-gradient(135deg,#0d2b4e,#1e40af); border:1px solid #3b82f6; box-shadow:0 6px 24px rgba(59,130,246,.25); }
.feat-card.teal   { background:linear-gradient(135deg,#042f2e,#0f6e56); border:1px solid #14b8a6; box-shadow:0 6px 24px rgba(20,184,166,.25); }
.feat-card.purple { background:linear-gradient(135deg,#1e1b4b,#4c1d95); border:1px solid #8b5cf6; box-shadow:0 6px 24px rgba(139,92,246,.25); }
.feat-card.rose   { background:linear-gradient(135deg,#4c0519,#9f1239); border:1px solid #f43f5e; box-shadow:0 6px 24px rgba(244,63,94,.25); }
.feat-card .ficon  { font-size:1.7rem; margin-bottom:8px; display:block; }
.feat-card .ftitle { font-size:0.9rem; font-weight:700; margin-bottom:3px; }
.feat-card .fdesc  { font-size:0.75rem; color:rgba(255,255,255,.65); line-height:1.5; }

.section-heading {
    font-size:0.95rem; font-weight:700; color:#1e40af;
    margin:18px 0 10px; padding-left:10px;
    border-left:4px solid #3b82f6; border-radius:0 4px 4px 0;
}

/* ── Chat area ── */
.chat-area {
    background:#0f172a; border-radius:16px;
    padding:18px; border:1px solid #1e3a5f;
    margin-bottom:16px;
}

/* User bubble */
.bubble-user { display:flex; justify-content:flex-end; margin-bottom:12px; }
.bubble-user-inner {
    background:linear-gradient(135deg,#1d4ed8,#7c3aed); color:#fff !important;
    border-radius:18px 18px 4px 18px; padding:11px 16px; max-width:72%;
    font-size:0.88rem; line-height:1.6; box-shadow:0 3px 12px rgba(99,102,241,.3);
}
.bubble-user-label { font-size:0.7rem; color:rgba(255,255,255,.65) !important; font-weight:600; margin-bottom:3px; text-align:right; }

/* Bot bubble */
.bubble-bot { display:flex; justify-content:flex-start; margin-bottom:4px; }
.bubble-bot-inner {
    background:#1e293b; border:1px solid #334155; border-top:3px solid #22d3ee;
    color:#e2e8f0 !important; border-radius:4px 18px 18px 18px;
    padding:13px 16px; max-width:84%; font-size:0.85rem; line-height:1.7;
    box-shadow:0 3px 12px rgba(0,0,0,.3);
}
.bubble-bot-label { font-size:0.7rem; color:#22d3ee !important; font-weight:700; margin-bottom:5px; letter-spacing:.04em; }
.source-pdf {
    display:inline-block; background:rgba(34,211,238,.15); border:1px solid rgba(34,211,238,.4);
    color:#22d3ee; font-size:0.68rem; padding:1px 7px; border-radius:10px; margin:0 3px 5px 0; font-weight:600;
}
.source-ai {
    display:inline-block; background:rgba(139,92,246,.15); border:1px solid rgba(139,92,246,.4);
    color:#a78bfa; font-size:0.68rem; padding:1px 7px; border-radius:10px; margin:0 3px 5px 0; font-weight:600;
}

/* Audio row under bot bubble */
.audio-row { margin:0 0 16px 0; padding-left:4px; }

/* Alert box */
.alert-box {
    background:linear-gradient(135deg,#431407,#78350f); color:#fed7aa !important;
    border-left:4px solid #f97316; padding:10px 14px; border-radius:8px;
    margin:5px 0; font-size:0.85rem;
}

/* Success box */
.success-box {
    background:linear-gradient(135deg,#14532d,#166534); color:#bbf7d0 !important;
    border-left:4px solid #22c55e; padding:10px 14px; border-radius:8px; font-size:0.85rem;
}

/* Upload zone */
.upload-zone {
    background:linear-gradient(135deg,#0d2b4e,#1e3a5f);
    border:2px dashed #3b82f6; border-radius:14px;
    padding:24px; text-align:center; margin-bottom:18px;
}
.upload-zone p { color:#94a3b8; font-size:0.85rem; margin-top:6px; }

/* Input row at bottom */
.input-section {
    background:#0d2b4e; border:1px solid #1e40af;
    border-radius:14px; padding:14px 16px; margin-top:8px;
}

/* Sidebar */
.sidebar-card {
    background:linear-gradient(135deg,#0d2b4e,#0f172a);
    border:1px solid #1e40af; border-radius:12px;
    padding:12px; margin-bottom:12px; color:#e2e8f0;
}

/* Streamlit overrides */
.stTextInput > div > div > input {
    background:#0f172a !important; border:1.5px solid #1e3a5f !important;
    color:#e2e8f0 !important; border-radius:10px !important; font-size:0.88rem !important;
}
.stTextInput > div > div > input:focus { border-color:#3b82f6 !important; }
.stButton > button {
    background:linear-gradient(135deg,#1e40af,#7c3aed) !important;
    color:white !important; border:none !important; border-radius:10px !important;
    font-weight:600 !important;
}
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# Hero
# ─────────────────────────────────────────────
st.markdown("""
<div class="hero-header">
    <h1>🏥 Voice-Enabled <span>Medical AI</span> Document QA</h1>
    <p>Upload your medical report · Ask in Hindi or English · Get AI-powered spoken answers</p>
    <div class="hero-badges">
        <span class="hero-badge">📄 PDF Extraction</span>
        <span class="hero-badge">🤖 Claude AI Chatbot</span>
        <span class="hero-badge">🎤 Voice I/O</span>
        <span class="hero-badge">🌐 Hindi + English</span>
        <span class="hero-badge">⚠️ Health Alerts</span>
        <span class="hero-badge">🔍 FAISS RAG</span>
    </div>
</div>
""", unsafe_allow_html=True)

# Feature cards
st.markdown("""
<div class="feat-grid">
    <div class="feat-card blue">
        <span class="ficon">📄</span>
        <div class="ftitle">Smart PDF Reading</div>
        <div class="fdesc">PyMuPDF + pdfplumber dual extraction</div>
    </div>
    <div class="feat-card teal">
        <span class="ficon">🎤</span>
        <div class="ftitle">Voice Q&A</div>
        <div class="fdesc">Speak in Hindi or English — answers spoken back</div>
    </div>
    <div class="feat-card purple">
        <span class="ficon">🤖</span>
        <div class="ftitle">AI Chatbot Merged</div>
        <div class="fdesc">PDF facts + Claude AI insights in every answer</div>
    </div>
    <div class="feat-card rose">
        <span class="ficon">⚠️</span>
        <div class="ftitle">Health Alerts</div>
        <div class="fdesc">Auto-detects 40+ out-of-range lab values</div>
    </div>
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown('<div class="sidebar-card">', unsafe_allow_html=True)
    st.header("⚙️ Settings")
    language_pref = st.selectbox("🌐 Language", ["Auto-Detect", "English", "Hindi"])
    top_k         = st.slider("🔍 Top-K Chunks", 2, 10, 5)
    voice_speed   = st.selectbox("🔊 Voice Speed", ["Normal", "Slow"])
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="sidebar-card">', unsafe_allow_html=True)
    st.header("🤖 Claude AI Key")
    api_key_input = st.text_input(
        "Anthropic API Key",
        type="password",
        placeholder="sk-ant-api03-...",
        help="Paste your key here. It will be used for AI enrichment in every answer."
    )
    if api_key_input:
        os.environ["ANTHROPIC_API_KEY"] = api_key_input.strip()
        st.success("✅ API key active")
    elif os.environ.get("ANTHROPIC_API_KEY"):
        st.success("✅ Key loaded from env")
    else:
        st.warning("⚠️ No key — PDF extraction only")
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="sidebar-card">', unsafe_allow_html=True)
    st.header("📱 QR Code")
    host_url = st.text_input("App URL", "http://localhost:8501")
    if st.button("Generate QR"):
        qr_path = generate_qr_code(host_url)
        st.image(qr_path, caption="Scan to Open", width=160)
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="sidebar-card">', unsafe_allow_html=True)
    st.header("📊 Session")
    st.write(f"💬 Messages: {len(st.session_state.chat_history.get_history())}")
    if st.session_state.doc_processed:
        st.success(f"✅ {len(st.session_state.chunks)} chunks loaded")
    if st.button("🗑️ Clear Chat"):
        st.session_state.chat_history.clear()
        st.session_state.audio_files = []
        st.session_state.last_query  = ""
        st.rerun()
    st.markdown('</div>', unsafe_allow_html=True)

# ─────────────────────────────────────────────
# Upload PDF
# ─────────────────────────────────────────────
st.markdown('<div class="section-heading">📄 Upload Medical Report</div>', unsafe_allow_html=True)
st.markdown('<div class="upload-zone">', unsafe_allow_html=True)
uploaded_file = st.file_uploader(
    "Drop PDF here", type=["pdf"], label_visibility="collapsed"
)
st.markdown('<p>Any standard lab report PDF · Hindi & English supported</p></div>',
            unsafe_allow_html=True)

if uploaded_file and not st.session_state.doc_processed:
    with st.spinner("🔄 Processing PDF — extracting · chunking · embedding..."):
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(uploaded_file.read())
            tmp_path = tmp.name
        raw_text    = extract_text_from_pdf(tmp_path)
        clean_text  = preprocess_text(raw_text)
        chunks      = chunk_text(clean_text)
        embeddings  = generate_embeddings(chunks)
        faiss_index = build_faiss_index(embeddings)
        st.session_state.chunks       = chunks
        st.session_state.faiss_index  = faiss_index
        st.session_state.doc_processed = True
        os.unlink(tmp_path)
    st.markdown('<div class="success-box">✅ Document processed! Ask your questions below.</div>',
                unsafe_allow_html=True)

# ─────────────────────────────────────────────
# Main Chat Interface
# ─────────────────────────────────────────────
if st.session_state.doc_processed:

    history = st.session_state.chat_history.get_history()
    audio_files = st.session_state.audio_files

    # ── Conversation (shown ABOVE the input box) ──
    if history:
        st.markdown('<div class="section-heading">🗨️ Conversation</div>', unsafe_allow_html=True)
        st.markdown('<div class="chat-area">', unsafe_allow_html=True)

        for i, turn in enumerate(history):
            # User bubble
            st.markdown(f"""
            <div class="bubble-user">
                <div class="bubble-user-inner">
                    <div class="bubble-user-label">👤 YOU</div>
                    {turn['user']}
                </div>
            </div>""", unsafe_allow_html=True)

            # Bot bubble
            answer_html = (
                turn['bot']
                .replace('\n', '<br>')
                .replace('**', '<b>', 1)
            )
            # Simple bold: replace remaining ** pairs
            import re as _re
            answer_html = _re.sub(r'\*\*(.*?)\*\*', r'<b>\1</b>', turn['bot'].replace('\n','<br>'))

            st.markdown(f"""
            <div class="bubble-bot">
                <div class="bubble-bot-inner">
                    <div class="bubble-bot-label">🏥 MEDICAL AI ASSISTANT</div>
                    <span class="source-pdf">📄 PDF Extracted</span>
                    <span class="source-ai">🤖 AI Enriched</span><br><br>
                    {answer_html}
                </div>
            </div>""", unsafe_allow_html=True)

            # 🔊 Individual audio player per turn
            if i < len(audio_files) and audio_files[i] and os.path.exists(audio_files[i]):
                st.markdown('<div class="audio-row">', unsafe_allow_html=True)
                st.audio(audio_files[i], format="audio/mp3")
                st.markdown('</div>', unsafe_allow_html=True)

        st.markdown('</div>', unsafe_allow_html=True)  # close chat-area

    # ── Alerts (shown after chat) ──
    # (Alerts stored in session from last query)
    if "last_alerts" in st.session_state and st.session_state.last_alerts:
        st.markdown('<div class="section-heading">⚠️ Medical Alerts</div>', unsafe_allow_html=True)
        for alert in st.session_state.last_alerts:
            st.markdown(f'<div class="alert-box">{alert}</div>', unsafe_allow_html=True)

    # ── Input at the BOTTOM ──
    st.markdown('<div class="section-heading">💬 Ask Your Question</div>', unsafe_allow_html=True)
    st.markdown('<div class="input-section">', unsafe_allow_html=True)

    col1, col2, col3 = st.columns([5, 1, 1])
    with col1:
        user_text = st.text_input(
            "q",
            label_visibility="collapsed",
            placeholder="Ask in Hindi or English:  'What is my WBC count?'  /  'मेरा हीमोग्लोबिन कैसा है?'",
            key="user_input"
        )
    with col2:
        voice_btn = st.button("🎤 Voice", use_container_width=True)
    with col3:
        send_btn  = st.button("📤 Send", use_container_width=True)

    st.markdown('</div>', unsafe_allow_html=True)

    # Determine query
    query = None
    if voice_btn:
        from voice_handler import SR_AVAILABLE, PYAUDIO_AVAILABLE
        if not SR_AVAILABLE or not PYAUDIO_AVAILABLE:
            st.warning("🎤 Voice input is not available on Streamlit Cloud (no microphone access). Please type your question.")
        else:
            with st.spinner("🎤 Listening..."):
                spoken = speech_to_text()
                if spoken:
                    query = spoken
                    st.info(f"🎤 You said: **{spoken}**")
                else:
                    st.warning("Could not detect speech. Please try again or type your question.")

    if send_btn and user_text.strip():
        query = user_text.strip()
    elif user_text.strip() and user_text.strip() != st.session_state.last_query:
        # Auto-submit on Enter (text changed since last submit)
        query = user_text.strip()

    # ── Process query ──
    if query and query != st.session_state.last_query:
        st.session_state.last_query = query

        with st.spinner("🧠 Extracting from PDF + consulting Claude AI..."):
            # Language
            lang = detect_language(query)
            if language_pref != "Auto-Detect":
                lang = "hi" if language_pref == "Hindi" else "en"

            # Translate to English for retrieval
            query_en = translate_to_english(query, lang)

            # Retrieve chunks
            q_emb      = embed_query(query_en)
            top_chunks = search_faiss(
                st.session_state.faiss_index,
                st.session_state.chunks,
                q_emb,
                top_k=top_k,
                query_text=query_en
            )

            # Medical alerts
            context = " ".join(top_chunks)
            alerts  = check_medical_values(context)
            st.session_state.last_alerts = alerts

            # Two-stage answer (PDF extract + Claude AI merge)
            history_text  = st.session_state.chat_history.get_context()
            answer_en     = generate_answer(query_en, top_chunks, history_text)

            # Translate back if Hindi
            if lang == "hi":
                from language_detector import translate_to_hindi
                answer_display = translate_to_hindi(answer_en)
            else:
                answer_display = answer_en

            # Save to history
            st.session_state.chat_history.add(query, answer_display)

            # TTS — generate audio for THIS turn
            tts_path = text_to_speech(answer_display, lang=lang, slow=(voice_speed == "Slow"))
            st.session_state.audio_files.append(tts_path)

        # Rerun so the new turn appears in the chat above the input box
        st.rerun()

else:
    # Not yet uploaded
    st.markdown("""
    <div style="
        background:linear-gradient(135deg,#0d2b4e,#0f172a);
        border:1px solid #1e3a5f; border-radius:16px;
        padding:36px; text-align:center; margin-top:20px;
    ">
        <p style="font-size:2.5rem;margin-bottom:10px;">📤</p>
        <p style="color:#7dd3fc;font-size:1.1rem;font-weight:600;">Upload a medical PDF to start</p>
        <p style="color:#475569;font-size:0.88rem;margin-top:6px;">
            Your report is processed locally · AI answers = PDF facts + Claude insights
        </p>
    </div>
    """, unsafe_allow_html=True)
