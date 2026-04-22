import streamlit as st
import os
import tempfile
import base64

from document_processor import extract_text_from_pdf, preprocess_text
from chunker import chunk_text
from embedder import generate_embeddings, embed_query
from vector_store import build_faiss_index, search_faiss
from voice_handler import speech_to_text, text_to_speech
from language_detector import detect_language, translate_to_english
from rag_pipeline import generate_answer, set_full_document, _call_ai
from medical_logic import check_medical_values
from chat_history import ChatHistory
from qr_generator import generate_qr_code

# ── Config ──────────────────────────────────────────────────────────────────
DEPLOY_URL = "https://voice-enabled-medical-ai-document-app-drwpwspgwlxapznqo3jumw.streamlit.app/"
# API keys are managed via sidebar — saved in session_state, restored on every rerun

# ─────────────────────────────────────────────
st.set_page_config(
    page_title="🏥 Medical AI",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ── Session state ─────────────────────────────────────────────────────────────
for k, v in {
    "chat_history":  ChatHistory(),
    "faiss_index":   None,
    "chunks":        [],
    "doc_processed": False,
    "audio_files":   [],
    "last_query":    "",
    "last_alerts":   [],
    "debug_mode":    False,
    "groq_key":      "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",
    "anthropic_key": "",
    "clean_text":    "",
}.items():
    if k not in st.session_state:
        st.session_state[k] = v

# Always restore API keys from session_state into os.environ on every rerun
if st.session_state.groq_key:
    os.environ["GROQ_API_KEY"] = st.session_state.groq_key
if st.session_state.anthropic_key:
    os.environ["ANTHROPIC_API_KEY"] = st.session_state.anthropic_key

# ── QR code (generated once, cached) ─────────────────────────────────────────
@st.cache_data
def get_qr_b64():
    path = generate_qr_code(DEPLOY_URL)
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode()

qr_b64 = get_qr_b64()

# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;700;900&family=Inter:wght@300;400;500;600&display=swap');

/* ── Global reset & space background ── */
html, body, [data-testid="stAppViewContainer"] {{
    font-family: 'Inter', sans-serif;
    background: #000510 !important;
}}
[data-testid="stAppViewContainer"] {{
    background:
        radial-gradient(ellipse at 20% 20%, rgba(0,80,180,0.18) 0%, transparent 55%),
        radial-gradient(ellipse at 80% 10%, rgba(100,0,200,0.13) 0%, transparent 50%),
        radial-gradient(ellipse at 60% 80%, rgba(0,150,200,0.10) 0%, transparent 50%),
        radial-gradient(ellipse at 10% 80%, rgba(180,0,120,0.08) 0%, transparent 40%),
        url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='800' height='800'%3E%3Crect fill='%23000510' width='800' height='800'/%3E%3Ccircle fill='white' cx='120' cy='60' r='1.2' opacity='.8'/%3E%3Ccircle fill='white' cx='340' cy='120' r='0.8' opacity='.6'/%3E%3Ccircle fill='white' cx='560' cy='40' r='1.5' opacity='.9'/%3E%3Ccircle fill='white' cx='700' cy='200' r='0.9' opacity='.5'/%3E%3Ccircle fill='white' cx='80' cy='300' r='1.1' opacity='.7'/%3E%3Ccircle fill='white' cx='240' cy='400' r='0.7' opacity='.4'/%3E%3Ccircle fill='white' cx='460' cy='350' r='1.3' opacity='.8'/%3E%3Ccircle fill='white' cx='620' cy='500' r='0.6' opacity='.5'/%3E%3Ccircle fill='white' cx='760' cy='420' r='1.0' opacity='.7'/%3E%3Ccircle fill='white' cx='150' cy='600' r='0.8' opacity='.6'/%3E%3Ccircle fill='white' cx='380' cy='680' r='1.2' opacity='.8'/%3E%3Ccircle fill='white' cx='540' cy='720' r='0.9' opacity='.5'/%3E%3Ccircle fill='white' cx='50' cy='750' r='1.1' opacity='.6'/%3E%3Ccircle fill='white' cx='680' cy='760' r='0.7' opacity='.4'/%3E%3Ccircle fill='%2388ccff' cx='200' cy='180' r='1.4' opacity='.6'/%3E%3Ccircle fill='%23cc88ff' cx='600' cy='300' r='1.0' opacity='.5'/%3E%3Ccircle fill='%2388ccff' cx='420' cy='560' r='1.2' opacity='.6'/%3E%3C/svg%3E") !important;
    background-size: cover, cover, cover, cover, 800px 800px !important;
}}
[data-testid="stHeader"] {{ background: transparent !important; }}
[data-testid="stSidebar"] {{ background: rgba(0,5,20,0.95) !important; border-right: 1px solid rgba(0,150,255,0.15) !important; }}
.block-container {{ padding: 2rem 3rem 3rem !important; max-width: 1400px; }}
section[data-testid="stSidebar"] .block-container {{ padding: 1.5rem 1rem !important; }}

/* ── Hero ── */
.hero {{
    text-align: center;
    padding: 60px 20px 40px;
    position: relative;
}}
.hero-glow {{
    position: absolute;
    top: 0; left: 50%;
    transform: translateX(-50%);
    width: 600px; height: 300px;
    background: radial-gradient(ellipse, rgba(0,120,255,0.15) 0%, transparent 70%);
    pointer-events: none;
}}
.hero-eyebrow {{
    display: inline-block;
    font-family: 'Orbitron', monospace;
    font-size: 0.7rem; font-weight: 700;
    letter-spacing: 0.25em; color: #22d3ee;
    background: rgba(34,211,238,0.08);
    border: 1px solid rgba(34,211,238,0.25);
    padding: 5px 18px; border-radius: 20px;
    margin-bottom: 20px;
    text-transform: uppercase;
}}
.hero-title {{
    font-family: 'Orbitron', monospace;
    font-size: clamp(2rem, 5vw, 3.8rem);
    font-weight: 900;
    line-height: 1.1;
    color: #ffffff;
    margin: 0 0 16px;
    text-shadow: 0 0 40px rgba(0,150,255,0.4);
}}
.hero-title span {{ color: #22d3ee; }}
.hero-subtitle {{
    font-size: 1rem; color: #94a3b8;
    max-width: 580px; margin: 0 auto 32px;
    line-height: 1.7;
}}
.hero-badges {{
    display: flex; gap: 10px;
    justify-content: center; flex-wrap: wrap;
    margin-bottom: 10px;
}}
.hbadge {{
    font-size: 0.72rem; font-weight: 500;
    padding: 5px 14px; border-radius: 20px;
    border: 1px solid; letter-spacing: 0.03em;
}}
.hbadge.blue  {{ color:#7dd3fc; border-color:rgba(125,211,252,.3); background:rgba(125,211,252,.06); }}
.hbadge.teal  {{ color:#5eead4; border-color:rgba(94,234,212,.3);  background:rgba(94,234,212,.06); }}
.hbadge.purple{{ color:#c4b5fd; border-color:rgba(196,181,253,.3); background:rgba(196,181,253,.06); }}
.hbadge.pink  {{ color:#f9a8d4; border-color:rgba(249,168,212,.3); background:rgba(249,168,212,.06); }}
.hbadge.amber {{ color:#fcd34d; border-color:rgba(252,211,77,.3);  background:rgba(252,211,77,.06); }}

/* ── Glass card ── */
.glass {{
    background: rgba(255,255,255,0.03);
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 20px;
    backdrop-filter: blur(12px);
    -webkit-backdrop-filter: blur(12px);
}}
.glass-blue {{
    background: rgba(0,80,200,0.07);
    border: 1px solid rgba(0,150,255,0.15);
    border-radius: 20px;
}}

/* ── Stat cards ── */
.stat-row {{
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 14px; margin-bottom: 28px;
}}
.stat-card {{
    background: rgba(255,255,255,0.03);
    border: 1px solid rgba(255,255,255,0.07);
    border-radius: 16px; padding: 20px 16px;
    text-align: center;
    transition: border-color .3s, transform .2s;
    position: relative; overflow: hidden;
}}
.stat-card::before {{
    content: '';
    position: absolute; top: 0; left: 0; right: 0;
    height: 2px;
}}
.stat-card.sc-blue::before  {{ background: linear-gradient(90deg,transparent,#3b82f6,transparent); }}
.stat-card.sc-teal::before  {{ background: linear-gradient(90deg,transparent,#14b8a6,transparent); }}
.stat-card.sc-purple::before{{ background: linear-gradient(90deg,transparent,#8b5cf6,transparent); }}
.stat-card.sc-pink::before  {{ background: linear-gradient(90deg,transparent,#ec4899,transparent); }}
.stat-card:hover {{ border-color:rgba(255,255,255,0.15); transform:translateY(-3px); }}
.stat-icon {{ font-size: 1.8rem; margin-bottom: 8px; }}
.stat-num  {{ font-family:'Orbitron',monospace; font-size:1.6rem; font-weight:700; color:#fff; margin-bottom:4px; }}
.stat-lbl  {{ font-size:0.75rem; color:#64748b; text-transform:uppercase; letter-spacing:.08em; }}

/* ── Section title ── */
.sec-title {{
    font-family: 'Orbitron', monospace;
    font-size: 0.75rem; font-weight: 700;
    color: #22d3ee; letter-spacing: .2em;
    text-transform: uppercase;
    margin: 28px 0 14px;
    display: flex; align-items: center; gap: 10px;
}}
.sec-title::after {{
    content: '';
    flex: 1; height: 1px;
    background: linear-gradient(90deg, rgba(34,211,238,.3), transparent);
}}

/* ── Upload zone ── */
.upload-zone {{
    border: 1.5px dashed rgba(34,211,238,0.35);
    border-radius: 16px;
    padding: 28px 20px;
    text-align: center;
    background: rgba(34,211,238,0.03);
    margin-bottom: 20px;
}}
.upload-zone p {{ color:#475569; font-size:.85rem; margin-top:6px; }}

/* ── Chat ── */
.chat-area {{
    background: rgba(0,0,0,0.4);
    border: 1px solid rgba(255,255,255,0.06);
    border-radius: 20px; padding: 20px;
    margin-bottom: 16px;
    backdrop-filter: blur(8px);
}}

.bubble-user {{ display:flex; justify-content:flex-end; margin-bottom:14px; }}
.bubble-user-inner {{
    background: linear-gradient(135deg,rgba(29,78,216,.7),rgba(124,58,237,.7));
    border: 1px solid rgba(139,92,246,.3);
    color: #fff !important;
    border-radius: 18px 18px 4px 18px;
    padding: 12px 18px; max-width: 72%;
    font-size: .88rem; line-height: 1.6;
    backdrop-filter: blur(8px);
}}
.bubble-user-label {{ font-size:.68rem; color:rgba(255,255,255,.55)!important; font-weight:600; margin-bottom:3px; text-align:right; font-family:'Orbitron',monospace; letter-spacing:.05em; }}

.bubble-bot {{ display:flex; justify-content:flex-start; margin-bottom:4px; }}
.bubble-bot-inner {{
    background: rgba(15,23,42,0.8);
    border: 1px solid rgba(34,211,238,0.2);
    border-top: 2px solid #22d3ee;
    color: #e2e8f0 !important;
    border-radius: 4px 18px 18px 18px;
    padding: 14px 18px; max-width: 84%;
    font-size: .85rem; line-height: 1.7;
    backdrop-filter: blur(8px);
}}
.bubble-bot-label {{ font-size:.68rem; color:#22d3ee!important; font-weight:700; margin-bottom:5px; letter-spacing:.1em; font-family:'Orbitron',monospace; }}
.src-pdf {{ display:inline-block; background:rgba(34,211,238,.1); border:1px solid rgba(34,211,238,.3); color:#22d3ee; font-size:.65rem; padding:1px 7px; border-radius:8px; margin:0 3px 5px 0; font-weight:600; }}
.src-ai  {{ display:inline-block; background:rgba(139,92,246,.1); border:1px solid rgba(139,92,246,.3); color:#a78bfa; font-size:.65rem; padding:1px 7px; border-radius:8px; margin:0 3px 5px 0; font-weight:600; }}

/* ── Input section ── */
.input-dock {{
    background: rgba(0,5,20,0.7);
    border: 1px solid rgba(34,211,238,0.2);
    border-radius: 16px; padding: 16px 18px;
    backdrop-filter: blur(12px);
}}

/* ── Alert ── */
.alert-box {{
    background: rgba(251,146,60,.08);
    border: 1px solid rgba(251,146,60,.3);
    border-left: 3px solid #f97316;
    color: #fed7aa !important;
    padding: 10px 14px; border-radius: 10px;
    margin: 5px 0; font-size:.85rem;
}}

/* ── Success ── */
.success-box {{
    background: rgba(34,197,94,.08);
    border: 1px solid rgba(34,197,94,.3);
    border-left: 3px solid #22c55e;
    color: #bbf7d0 !important;
    padding: 10px 14px; border-radius: 10px; font-size:.85rem;
}}

/* ── QR ── */
.qr-wrap {{
    text-align: center;
    background: rgba(255,255,255,0.03);
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 16px; padding: 20px 14px;
}}
.qr-wrap img {{ border-radius: 10px; }}
.qr-label {{ font-size:.72rem; color:#64748b; margin-top:8px; font-family:'Orbitron',monospace; letter-spacing:.08em; }}

/* ── Streamlit widget overrides ── */
.stTextInput > div > div > input {{
    background: rgba(0,5,20,0.6) !important;
    border: 1.5px solid rgba(34,211,238,0.25) !important;
    color: #e2e8f0 !important;
    border-radius: 10px !important;
    font-size: .88rem !important;
}}
.stTextInput > div > div > input:focus {{
    border-color: rgba(34,211,238,0.6) !important;
    box-shadow: 0 0 0 3px rgba(34,211,238,0.1) !important;
}}
.stTextInput > div > div > input::placeholder {{ color:#334155 !important; }}
.stButton > button {{
    background: linear-gradient(135deg,rgba(29,78,216,.8),rgba(124,58,237,.8)) !important;
    color: white !important; border: 1px solid rgba(139,92,246,.4) !important;
    border-radius: 10px !important; font-weight: 600 !important;
    font-family: 'Inter', sans-serif !important;
    transition: all .2s !important;
}}
.stButton > button:hover {{
    background: linear-gradient(135deg,rgba(29,78,216,1),rgba(124,58,237,1)) !important;
    box-shadow: 0 0 20px rgba(139,92,246,.4) !important;
}}
div[data-testid="stFileUploader"] {{
    background: transparent !important;
}}
.stSelectbox > div > div {{
    background: rgba(0,5,20,.7) !important;
    border: 1px solid rgba(34,211,238,.2) !important;
    color: #e2e8f0 !important;
    border-radius: 10px !important;
}}
label, .stSelectbox label, .stSlider label {{
    color: #94a3b8 !important;
    font-size: .82rem !important;
}}
.stSlider > div > div > div > div {{
    background: linear-gradient(90deg,#3b82f6,#8b5cf6) !important;
}}
p, li, span {{ color: #94a3b8; }}
h1,h2,h3 {{ color: #f1f5f9 !important; }}
</style>
""", unsafe_allow_html=True)

# ── HERO ─────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
    <div class="hero-glow"></div>
    <div class="hero-eyebrow">AI · RAG · Voice · Multilingual</div>
    <h1 class="hero-title">Voice-Enabled<br><span>Medical AI</span> Document QA</h1>
    <p class="hero-subtitle">
        Upload your medical report, ask questions in Hindi or English by voice or text,
        and receive AI-powered answers that merge PDF data with clinical insights.
    </p>
    <div class="hero-badges">
        <span class="hbadge blue">📄 PDF Extraction</span>
        <span class="hbadge teal">🤖 AI</span>
        <span class="hbadge purple">🎤 Voice I/O</span>
        <span class="hbadge pink">🌐 Hindi + English</span>
        <span class="hbadge amber">⚠️ Health Alerts</span>
    </div>
</div>
""", unsafe_allow_html=True)

# ── STAT CARDS ────────────────────────────────────────────────────────────────
st.markdown("""
<div class="stat-row">
    <div class="stat-card sc-blue">
        <div class="stat-icon">🔬</div>
        <div class="stat-num">40+</div>
        <div class="stat-lbl">Lab Tests Tracked</div>
    </div>
    <div class="stat-card sc-teal">
        <div class="stat-icon">🧠</div>
        <div class="stat-num">RAG</div>
        <div class="stat-lbl">FAISS + MiniLM</div>
    </div>
    <div class="stat-card sc-purple">
        <div class="stat-icon">🌐</div>
        <div class="stat-num">2</div>
        <div class="stat-lbl">Languages</div>
    </div>
    <div class="stat-card sc-pink">
        <div class="stat-icon">⚡</div>
        <div class="stat-num">2x</div>
        <div class="stat-lbl">Answer Pipeline</div>
    </div>
</div>
""", unsafe_allow_html=True)

# ── SIDEBAR ───────────────────────────────────────────────────────────────────
with st.sidebar:
    # QR Code — hardcoded deploy link
    st.markdown(f"""
    <div class="qr-wrap">
        <img src="data:image/png;base64,{qr_b64}" width="160"/>
        <div class="qr-label">SCAN TO OPEN APP</div>
        <div style="font-size:.65rem;color:#334155;margin-top:4px;">
            voice-enabled-medical-ai<br>-document-app.streamlit.app
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    st.markdown('<div class="sec-title">Settings</div>', unsafe_allow_html=True)
    language_pref = st.selectbox("🌐 Language", ["Auto-Detect", "English", "Hindi"])
    voice_speed   = st.selectbox("🔊 Voice Speed", ["Normal", "Slow"])

    st.markdown('<div class="sec-title">Session</div>', unsafe_allow_html=True)
    st.markdown(f'<p style="font-size:.82rem;">💬 Messages: <b style="color:#22d3ee">{len(st.session_state.chat_history.get_history())}</b></p>', unsafe_allow_html=True)
    if st.session_state.doc_processed:
        st.markdown(f'<p style="font-size:.82rem;">📦 Chunks: <b style="color:#22d3ee">{len(st.session_state.chunks)}</b></p>', unsafe_allow_html=True)
    if st.button("🗑️ Clear Chat", use_container_width=True):
        st.session_state.chat_history.clear()
        st.session_state.audio_files = []
        st.session_state.last_query  = ""
        st.session_state.last_alerts = []
        st.rerun()

# Re-register full doc text on every rerun (Streamlit reruns whole script on each click)
if st.session_state.doc_processed and "clean_text" in st.session_state:
    set_full_document(st.session_state.clean_text)

# ── UPLOAD ────────────────────────────────────────────────────────────────────
st.markdown('<div class="sec-title">Upload Medical Report</div>', unsafe_allow_html=True)
st.markdown('<div class="upload-zone">', unsafe_allow_html=True)
uploaded_file = st.file_uploader("PDF", type=["pdf"], label_visibility="collapsed")
st.markdown('<p>📋 Drop your lab report PDF here · Hindi & English supported · Processed locally</p></div>',
            unsafe_allow_html=True)

if uploaded_file and not st.session_state.doc_processed:
    with st.spinner("🔄 Extracting · Chunking · Embedding..."):
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(uploaded_file.read())
            tmp_path = tmp.name
        raw_text    = extract_text_from_pdf(tmp_path)
        clean_text  = preprocess_text(raw_text)
        chunks      = chunk_text(clean_text)
        embeddings  = generate_embeddings(chunks)
        faiss_index = build_faiss_index(embeddings)
        st.session_state.chunks        = chunks
        st.session_state.faiss_index   = faiss_index
        st.session_state.doc_processed = True
        st.session_state.clean_text    = clean_text   # store full text
        set_full_document(clean_text)                 # give full text to AI
        os.unlink(tmp_path)
    st.markdown('<div class="success-box">✅ Document processed — ask your questions below!</div>',
                unsafe_allow_html=True)

# ── CHAT ──────────────────────────────────────────────────────────────────────
if st.session_state.doc_processed:

    history     = st.session_state.chat_history.get_history()
    audio_files = st.session_state.audio_files

    # Conversation shown ABOVE input
    if history:
        st.markdown('<div class="sec-title">Conversation</div>', unsafe_allow_html=True)
        st.markdown('<div class="chat-area">', unsafe_allow_html=True)

        import re as _re
        for i, turn in enumerate(history):
            # User bubble
            st.markdown(f"""
            <div class="bubble-user">
                <div class="bubble-user-inner">
                    <div class="bubble-user-label">YOU</div>
                    {turn['user']}
                </div>
            </div>""", unsafe_allow_html=True)

            # Bot bubble
            answer_html = _re.sub(r'\*\*(.*?)\*\*', r'<b>\1</b>', turn['bot'].replace('\n','<br>'))
            st.markdown(f"""
            <div class="bubble-bot">
                <div class="bubble-bot-inner">
                    <div class="bubble-bot-label">🏥 MEDICAL AI ASSISTANT</div>
                    <span class="src-pdf">📄 PDF Extracted</span>
                    <span class="src-ai">🤖 AI Enriched</span><br><br>
                    {answer_html}
                </div>
            </div>""", unsafe_allow_html=True)

            # Individual audio player per turn
            if i < len(audio_files) and audio_files[i] and os.path.exists(audio_files[i]):
                st.audio(audio_files[i], format="audio/mp3")

        st.markdown('</div>', unsafe_allow_html=True)

    # Alerts
    if st.session_state.last_alerts:
        st.markdown('<div class="sec-title">Medical Alerts</div>', unsafe_allow_html=True)
        for alert in st.session_state.last_alerts:
            st.markdown(f'<div class="alert-box">{alert}</div>', unsafe_allow_html=True)

    # Input dock — BOTTOM
    st.markdown('<div class="sec-title">Ask Your Question</div>', unsafe_allow_html=True)
    st.markdown('<div class="input-dock">', unsafe_allow_html=True)
    col1, col2, col3 = st.columns([5, 1, 1])
    with col1:
        user_text = st.text_input(
            "q", label_visibility="collapsed",
            placeholder="Ask about your report / medicine / diet — e.g. 'What is my WBC?' · 'Explain paracetamol' · 'What food for low hemoglobin?'",
            key="user_input"
        )
    with col2:
        voice_btn = st.button("🎤", use_container_width=True)
    with col3:
        send_btn  = st.button("Send ➤", use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

    # Resolve query
    query = None
    if voice_btn:
        from voice_handler import SR_AVAILABLE, PYAUDIO_AVAILABLE
        if not SR_AVAILABLE or not PYAUDIO_AVAILABLE:
            st.warning("🎤 Voice input not available on Streamlit Cloud. Please type your question.")
        else:
            with st.spinner("🎤 Listening..."):
                spoken = speech_to_text()
                if spoken:
                    query = spoken
                    st.info(f"🎤 You said: **{spoken}**")
                else:
                    st.warning("No speech detected. Please try again.")

    if send_btn and user_text.strip():
        query = user_text.strip()
    elif user_text.strip() and user_text.strip() != st.session_state.last_query:
        query = user_text.strip()

    # Process
    if query and query != st.session_state.last_query:
        st.session_state.last_query = query
        with st.spinner("🧠 Extracting from PDF + consulting AI..."):
            lang     = detect_language(query)
            if language_pref != "Auto-Detect":
                lang = "hi" if language_pref == "Hindi" else "en"

            query_en   = translate_to_english(query, lang)
            q_emb      = embed_query(query_en)
            # Auto top_k: use more chunks for summary, fewer for specific questions
            _q = query_en.lower()
            if any(w in _q for w in ["summary","all","overall","complete","full report"]):
                _top_k = min(10, len(st.session_state.chunks))
            else:
                _top_k = min(6, len(st.session_state.chunks))
            top_chunks = search_faiss(
                st.session_state.faiss_index,
                st.session_state.chunks,
                q_emb, top_k=_top_k, query_text=query_en
            )
            alerts  = check_medical_values(" ".join(top_chunks))
            st.session_state.last_alerts = alerts

            # Debug: show raw PDF text if enabled
            if st.session_state.debug_mode:
                from rag_pipeline import debug_chunks
                st.expander("🔍 Raw PDF text retrieved").text(debug_chunks(top_chunks))

            answer_en = generate_answer(
                query_en, top_chunks,
                st.session_state.chat_history.get_context()
            )
            if lang == "hi":
                from language_detector import translate_to_hindi
                answer_display = translate_to_hindi(answer_en)
            else:
                answer_display = answer_en

            st.session_state.chat_history.add(query, answer_display)
            tts_path = text_to_speech(answer_display, lang=lang, slow=(voice_speed=="Slow"))
            st.session_state.audio_files.append(tts_path)

        st.rerun()

else:
    # No PDF yet — show general health chatbot
    st.markdown("""
    <div style="
        text-align:center; padding:32px 20px 20px;
        background:rgba(0,0,0,0.3);
        border:1px dashed rgba(34,211,238,0.2);
        border-radius:20px; margin-bottom:20px;
    ">
        <div style="font-size:3rem;margin-bottom:10px;">📤</div>
        <div style="font-family:'Orbitron',monospace;font-size:1rem;color:#22d3ee;font-weight:700;margin-bottom:6px;">
            Upload a Medical PDF above to ask about your report
        </div>
        <div style="font-size:.82rem;color:#475569;">
            Or use the Health Chatbot below to ask general health questions
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ── General Health Chatbot (no PDF needed) ──
    st.markdown('<div class="sec-title">💬 General Health Chatbot</div>', unsafe_allow_html=True)
    st.markdown("""
    <div style="background:rgba(139,92,246,0.08);border:1px solid rgba(139,92,246,0.25);
    border-radius:12px;padding:12px 16px;margin-bottom:14px;font-size:.82rem;color:#a78bfa;">
        🤖 Ask anything about health, diet, food, recovery, symptoms, and lifestyle — no PDF needed!
        <br>Examples: "What food should I eat if I have low hemoglobin?" · "How to recover from high cholesterol?"
        · "What are symptoms of vitamin D deficiency?"
    </div>
    """, unsafe_allow_html=True)

    if "general_chat_history" not in st.session_state:
        st.session_state.general_chat_history = ChatHistory()
    if "general_audio_files" not in st.session_state:
        st.session_state.general_audio_files = []
    if "general_last_query" not in st.session_state:
        st.session_state.general_last_query = ""

    # Show general chat history
    gen_history = st.session_state.general_chat_history.get_history()
    gen_audio   = st.session_state.general_audio_files

    if gen_history:
        import re as _re2
        st.markdown('<div class="chat-area">', unsafe_allow_html=True)
        for i, turn in enumerate(gen_history):
            st.markdown(f"""
            <div class="bubble-user">
                <div class="bubble-user-inner">
                    <div class="bubble-user-label">YOU</div>
                    {turn['user']}
                </div>
            </div>""", unsafe_allow_html=True)
            answer_html = _re2.sub(r'\*\*(.*?)\*\*', r'<b>\1</b>', turn['bot'].replace('\n','<br>'))
            st.markdown(f"""
            <div class="bubble-bot">
                <div class="bubble-bot-inner">
                    <div class="bubble-bot-label">🤖 HEALTH ASSISTANT</div>
                    <span class="src-ai">🤖 AI Powered</span><br><br>
                    {answer_html}
                </div>
            </div>""", unsafe_allow_html=True)
            if i < len(gen_audio) and gen_audio[i] and os.path.exists(gen_audio[i]):
                st.audio(gen_audio[i], format="audio/mp3")
        st.markdown('</div>', unsafe_allow_html=True)

    # General chatbot input
    st.markdown('<div class="input-dock">', unsafe_allow_html=True)
    gc1, gc2 = st.columns([5, 1])
    with gc1:
        gen_input = st.text_input(
            "gen_q", label_visibility="collapsed",
            placeholder="Ask any health question: food, diet, recovery, symptoms...",
            key="gen_input"
        )
    with gc2:
        gen_send = st.button("Ask 🤖", use_container_width=True, key="gen_send")
    st.markdown('</div>', unsafe_allow_html=True)

    gen_query = None
    if gen_send and gen_input.strip():
        gen_query = gen_input.strip()
    elif gen_input.strip() and gen_input.strip() != st.session_state.general_last_query:
        gen_query = gen_input.strip()

    if gen_query and gen_query != st.session_state.general_last_query:
        st.session_state.general_last_query = gen_query
        with st.spinner("🤖 Health AI thinking..."):
            from rag_pipeline import _answer_health_chat, _call_claude
            # General chat has no PDF — pass empty chunks
            system = """You are a friendly, expert health and wellness AI assistant.
Answer health questions about diet, food, recovery, symptoms, lifestyle, and medical conditions.
Give specific, practical, actionable advice with clear sections.
Use bullet points for food lists. Be warm and encouraging.
Always recommend consulting a doctor for diagnosis or treatment.
Match the language the patient uses (Hindi or English)."""
            user = gen_query
            if st.session_state.general_chat_history.get_history():
                user = f"Previous conversation:\n{st.session_state.general_chat_history.get_context()}\n\nQuestion: {gen_query}"
            ai_resp = _call_claude(system, user, max_tokens=600)
            if not ai_resp:
                ai_resp = "⚠️ Please add your Anthropic API key in the sidebar to enable the AI health chatbot."

            st.session_state.general_chat_history.add(gen_query, ai_resp)
            tts_path = text_to_speech(ai_resp[:500], lang="en", slow=False)
            st.session_state.general_audio_files.append(tts_path)
        st.rerun()

    if st.button("🗑️ Clear Health Chat", key="clear_gen"):
        st.session_state.general_chat_history.clear()
        st.session_state.general_audio_files = []
        st.session_state.general_last_query = ""
        st.rerun()
