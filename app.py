import streamlit as st
import os
import tempfile
import base64
import re as _re

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

DEPLOY_URL = "https://voice-enabled-medical-ai-document-app.streamlit.app/"

st.set_page_config(
    page_title="MedAI — Smart Medical Assistant",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ── Session state ─────────────────────────────────────────────────────────────
for k, v in {
    "page":           "welcome",
    "chat_history":   ChatHistory(),
    "faiss_index":    None,
    "chunks":         [],
    "doc_processed":  False,
    "audio_files":    [],
    "last_query":     "",
    "last_alerts":    [],
    "debug_mode":     False,
    "groq_key":       "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",
    "anthropic_key":  "",
    "clean_text":     "",
    "general_chat_history": ChatHistory(),
    "general_audio_files":  [],
    "general_last_query":   "",
}.items():
    if k not in st.session_state:
        st.session_state[k] = v

if st.session_state.groq_key:
    os.environ["GROQ_API_KEY"] = st.session_state.groq_key
if st.session_state.anthropic_key:
    os.environ["ANTHROPIC_API_KEY"] = st.session_state.anthropic_key
if st.session_state.doc_processed and st.session_state.clean_text:
    set_full_document(st.session_state.clean_text)

@st.cache_data
def get_qr_b64():
    path = generate_qr_code(DEPLOY_URL)
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode()
qr_b64 = get_qr_b64()

# ══════════════════════════════════════════════════════════════════════════════
# WELCOME PAGE
# ══════════════════════════════════════════════════════════════════════════════
if st.session_state.page == "welcome":

    st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;700;900&family=Inter:wght@300;400;500;600&display=swap');
*{box-sizing:border-box;margin:0;padding:0}
html,body,[data-testid="stAppViewContainer"]{
  font-family:'Inter',sans-serif!important;
  background:#000510!important;
}
[data-testid="stAppViewContainer"]{
  background:
    radial-gradient(ellipse at 15% 15%,rgba(0,80,200,0.2) 0%,transparent 55%),
    radial-gradient(ellipse at 85% 10%,rgba(120,0,220,0.15) 0%,transparent 50%),
    radial-gradient(ellipse at 50% 85%,rgba(0,160,220,0.1) 0%,transparent 50%),
    url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='900' height='900'%3E%3Crect fill='%23000510' width='900' height='900'/%3E%3Ccircle fill='white' cx='120' cy='60' r='1.2' opacity='.8'/%3E%3Ccircle fill='white' cx='340' cy='120' r='0.8' opacity='.6'/%3E%3Ccircle fill='white' cx='560' cy='40' r='1.5' opacity='.9'/%3E%3Ccircle fill='white' cx='720' cy='200' r='0.9' opacity='.5'/%3E%3Ccircle fill='white' cx='80' cy='300' r='1.1' opacity='.7'/%3E%3Ccircle fill='white' cx='250' cy='420' r='0.7' opacity='.4'/%3E%3Ccircle fill='white' cx='470' cy='370' r='1.3' opacity='.8'/%3E%3Ccircle fill='white' cx='640' cy='520' r='0.6' opacity='.5'/%3E%3Ccircle fill='white' cx='780' cy='440' r='1.0' opacity='.7'/%3E%3Ccircle fill='white' cx='160' cy='620' r='0.8' opacity='.6'/%3E%3Ccircle fill='white' cx='390' cy='700' r='1.2' opacity='.8'/%3E%3Ccircle fill='white' cx='550' cy='750' r='0.9' opacity='.5'/%3E%3Ccircle fill='white' cx='60' cy='770' r='1.1' opacity='.6'/%3E%3Ccircle fill='white' cx='700' cy='780' r='0.7' opacity='.4'/%3E%3Ccircle fill='%2388ccff' cx='210' cy='190' r='1.4' opacity='.6'/%3E%3Ccircle fill='%23cc88ff' cx='610' cy='310' r='1.0' opacity='.5'/%3E%3Ccircle fill='%2388ccff' cx='430' cy='580' r='1.2' opacity='.6'/%3E%3Ccircle fill='white' cx='820' cy='90' r='0.9' opacity='.7'/%3E%3Ccircle fill='white' cx='30' cy='500' r='1.0' opacity='.5'/%3E%3Ccircle fill='%23aaddff' cx='760' cy='600' r='1.1' opacity='.5'/%3E%3C/svg%3E")!important;
  background-size:cover,cover,cover,900px 900px!important;
}
[data-testid="stHeader"]{display:none!important}
[data-testid="stSidebar"]{display:none!important}
.block-container{padding:0!important;max-width:100%!important}

/* ── Top nav bar ── */
.topnav{
  padding:16px 60px;
  display:flex;align-items:center;justify-content:space-between;
  border-bottom:1px solid rgba(255,255,255,0.06);
  background:rgba(0,5,20,0.4);
  backdrop-filter:blur(10px);
}
.nav-logo{
  font-family:'Orbitron',monospace;font-size:1.1rem;font-weight:900;
  color:#fff;display:flex;align-items:center;gap:10px;
}
.nav-logo .logo-dot{
  width:10px;height:10px;border-radius:50%;background:#22d3ee;
  box-shadow:0 0 12px #22d3ee;animation:blink 2s infinite;
}
@keyframes blink{0%,100%{opacity:1}50%{opacity:.3}}
.nav-pills{display:flex;gap:8px}
.npill{
  font-size:11px;font-weight:500;
  padding:5px 14px;border-radius:20px;
  border:1px solid rgba(255,255,255,0.12);
  color:rgba(255,255,255,0.6);
  background:rgba(255,255,255,0.04);
  letter-spacing:.03em;
}

/* ── Hero area ── */
.w-hero{
  text-align:center;
  padding:56px 40px 40px;
  position:relative;
}
.w-hero-glow{
  position:absolute;top:0;left:50%;transform:translateX(-50%);
  width:700px;height:350px;
  background:radial-gradient(ellipse,rgba(0,130,255,0.12) 0%,transparent 65%);
  pointer-events:none;
}
.w-eyebrow{
  display:inline-flex;align-items:center;gap:8px;
  font-family:'Orbitron',monospace;font-size:0.68rem;font-weight:700;
  letter-spacing:.25em;color:#22d3ee;text-transform:uppercase;
  background:rgba(34,211,238,0.07);border:1px solid rgba(34,211,238,0.22);
  padding:6px 20px;border-radius:20px;margin-bottom:22px;
}
.w-eyebrow .ew-dot{width:6px;height:6px;border-radius:50%;background:#22d3ee}
.w-title{
  font-family:'Orbitron',monospace;
  font-size:clamp(2rem,5vw,3.5rem);font-weight:900;
  color:#ffffff;line-height:1.05;margin-bottom:16px;
  text-shadow:0 0 60px rgba(0,150,255,0.3);
  animation:fadeUp .8s ease both;
}
@keyframes fadeUp{from{opacity:0;transform:translateY(30px)}to{opacity:1;transform:translateY(0)}}
.w-title .t-cyan{color:#22d3ee}
.w-title .t-purple{color:#a78bfa}
.w-desc{
  font-size:1rem;color:#8892b0;
  max-width:600px;margin:0 auto 32px;
  line-height:1.75;
  animation:fadeUp .8s .1s ease both;
}

/* ── Feature cards ── */
.feat-section{padding:0 50px 50px;position:relative;z-index:2}
.feat-title{
  font-family:'Orbitron',monospace;font-size:.68rem;font-weight:700;
  color:#22d3ee;letter-spacing:.2em;text-transform:uppercase;
  text-align:center;margin-bottom:4px;
}
.feat-subtitle{
  text-align:center;font-size:1rem;color:#e2e8f0;
  font-weight:500;margin-bottom:24px;
}
.feat-grid{
  display:grid;
  grid-template-columns:repeat(3,1fr);
  gap:14px;margin-bottom:36px;
}
.feat-card{
  background:rgba(255,255,255,0.02);
  border:1px solid rgba(255,255,255,0.07);
  border-radius:16px;padding:18px 16px;
  position:relative;overflow:hidden;
  transition:transform .2s,border-color .2s;
}
.feat-card::before{
  content:'';position:absolute;top:0;left:0;right:0;height:2px;
}
.fc-blue::before{background:linear-gradient(90deg,transparent,#3b82f6,transparent)}
.fc-teal::before{background:linear-gradient(90deg,transparent,#14b8a6,transparent)}
.fc-purple::before{background:linear-gradient(90deg,transparent,#8b5cf6,transparent)}
.fc-rose::before{background:linear-gradient(90deg,transparent,#f43f5e,transparent)}
.fc-amber::before{background:linear-gradient(90deg,transparent,#f59e0b,transparent)}
.fc-green::before{background:linear-gradient(90deg,transparent,#22c55e,transparent)}
.feat-card:hover{transform:translateY(-3px);border-color:rgba(255,255,255,0.14)}
.fc-icon{
  width:38px;height:38px;border-radius:10px;
  display:flex;align-items:center;justify-content:center;
  font-size:18px;margin-bottom:10px;
}
.fci-blue{background:linear-gradient(135deg,rgba(59,130,246,.18),rgba(59,130,246,.06));border:1px solid rgba(59,130,246,.25)}
.fci-teal{background:linear-gradient(135deg,rgba(20,184,166,.18),rgba(20,184,166,.06));border:1px solid rgba(20,184,166,.25)}
.fci-purple{background:linear-gradient(135deg,rgba(139,92,246,.18),rgba(139,92,246,.06));border:1px solid rgba(139,92,246,.25)}
.fci-rose{background:linear-gradient(135deg,rgba(244,63,94,.18),rgba(244,63,94,.06));border:1px solid rgba(244,63,94,.25)}
.fci-amber{background:linear-gradient(135deg,rgba(245,158,11,.18),rgba(245,158,11,.06));border:1px solid rgba(245,158,11,.25)}
.fci-green{background:linear-gradient(135deg,rgba(34,197,94,.18),rgba(34,197,94,.06));border:1px solid rgba(34,197,94,.25)}
.fc-title{font-size:.88rem;font-weight:700;color:#f1f5f9;margin-bottom:5px}
.fc-desc{
  font-size:.78rem;color:#64748b;line-height:1.55;
  display:-webkit-box;-webkit-line-clamp:3;
  -webkit-box-orient:vertical;overflow:hidden;
}

/* ── Footer ── */
.w-footer{
  text-align:center;padding:24px 60px;
  border-top:1px solid rgba(255,255,255,0.06);
  background:rgba(0,0,0,0.3);
}
.footer-note{font-size:.75rem;color:#334155;line-height:1.8}
.footer-highlight{color:#22d3ee;font-weight:600}

/* ── Streamlit button overrides — WHITE text ── */
.stButton>button{
  background:linear-gradient(135deg,#0ea5e9,#6366f1)!important;
  color:#ffffff!important;
  border:none!important;
  border-radius:12px!important;
  font-size:1rem!important;
  font-weight:700!important;
  padding:14px 36px!important;
  font-family:'Inter',sans-serif!important;
  box-shadow:0 8px 28px rgba(99,102,241,.4)!important;
  transition:all .2s!important;
  letter-spacing:.01em!important;
  width:auto!important;
  -webkit-text-fill-color:#ffffff!important;
}
.stButton>button:hover{
  transform:translateY(-2px)!important;
  box-shadow:0 12px 36px rgba(99,102,241,.6)!important;
  color:#ffffff!important;
  -webkit-text-fill-color:#ffffff!important;
}
.stButton>button:active{
  transform:scale(0.97)!important;
  color:#ffffff!important;
  -webkit-text-fill-color:#ffffff!important;
}
.stButton>button p,
.stButton>button span{
  color:#ffffff!important;
  -webkit-text-fill-color:#ffffff!important;
}
p,li,span{color:#8892b0}
</style>
""", unsafe_allow_html=True)

    # ── TOP NAV ──────────────────────────────────────────────────────────────
    st.markdown("""
<div class="topnav">
  <div class="nav-logo">
    <div class="logo-dot"></div>
    MedAI System
  </div>
  <div class="nav-pills">
    <span class="npill">🔬 NLP + RAG</span>
    <span class="npill">🤖 LLaMA 3.3 70B</span>
    <span class="npill">🎤 Voice-Enabled</span>
    <span class="npill">🌐 Hindi + English</span>
  </div>
</div>
""", unsafe_allow_html=True)

    # ── HERO ─────────────────────────────────────────────────────────────────
    st.markdown("""
<div class="w-hero">
  <div class="w-hero-glow"></div>
  <div class="w-eyebrow"><div class="ew-dot"></div> AI-Powered Medical Intelligence</div>
  <h1 class="w-title">
    Voice-Enabled<br>
    <span class="t-cyan">Medical Report</span><br>
    <span class="t-purple">AI Assistant</span>
  </h1>
  <p class="w-desc">
    Upload your medical lab report, ask questions in <b style="color:#e2e8f0">Hindi or English</b> using voice or text,
    and get instant AI-powered answers — including <b style="color:#e2e8f0">medicine explanations</b>,
    <b style="color:#e2e8f0">diet advice</b>, and <b style="color:#e2e8f0">health alerts</b>.
  </p>
</div>
""", unsafe_allow_html=True)

    # ── CTA BUTTON ────────────────────────────────────────────────────────────
    col_l, col_c, col_r = st.columns([2, 2, 2])
    with col_c:
        if st.button("🚀  Get Started — Launch App", use_container_width=True):
            st.session_state.page = "app"
            st.rerun()

    # ── FEATURE CARDS ─────────────────────────────────────────────────────────
    st.markdown("""

""", unsafe_allow_html=True)

    # ── SECOND CTA ────────────────────────────────────────────────────────────
    col_l2, col_c2, col_r2 = st.columns([2, 2, 2])
    with col_c2:
        if st.button():
            st.session_state.page = "app"
            st.rerun()

    # ── FOOTER ────────────────────────────────────────────────────────────────
    st.markdown("""
<div class="w-footer">
  <div class="footer-note">
    <span class="footer-highlight">MedAI</span> · Python · Streamlit · FAISS · MiniLM · LLaMA 3.3 70B (Groq) · Google STT/TTS · Hindi &amp; English · Voice + Text · PDF Extraction<br>
    <span style="color:#1e293b">⚕️ For educational purposes only. Always consult a qualified physician for medical advice.</span>
  </div>
</div>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# MAIN APP PAGE
# ══════════════════════════════════════════════════════════════════════════════
else:

    st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;700;900&family=Inter:wght@300;400;500;600&display=swap');
*{{box-sizing:border-box}}
html,body,[data-testid="stAppViewContainer"]{{
  font-family:'Inter',sans-serif!important;
  background:#000510!important;
}}
[data-testid="stAppViewContainer"]{{
  background:
    radial-gradient(ellipse at 20% 20%,rgba(0,80,180,0.18) 0%,transparent 55%),
    radial-gradient(ellipse at 80% 10%,rgba(100,0,200,0.13) 0%,transparent 50%),
    radial-gradient(ellipse at 60% 80%,rgba(0,150,200,0.10) 0%,transparent 50%),
    radial-gradient(ellipse at 10% 80%,rgba(180,0,120,0.08) 0%,transparent 40%),
    url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='800' height='800'%3E%3Crect fill='%23000510' width='800' height='800'/%3E%3Ccircle fill='white' cx='120' cy='60' r='1.2' opacity='.8'/%3E%3Ccircle fill='white' cx='340' cy='120' r='0.8' opacity='.6'/%3E%3Ccircle fill='white' cx='560' cy='40' r='1.5' opacity='.9'/%3E%3Ccircle fill='white' cx='700' cy='200' r='0.9' opacity='.5'/%3E%3Ccircle fill='white' cx='80' cy='300' r='1.1' opacity='.7'/%3E%3Ccircle fill='white' cx='240' cy='400' r='0.7' opacity='.4'/%3E%3Ccircle fill='white' cx='460' cy='350' r='1.3' opacity='.8'/%3E%3Ccircle fill='white' cx='620' cy='500' r='0.6' opacity='.5'/%3E%3Ccircle fill='white' cx='760' cy='420' r='1.0' opacity='.7'/%3E%3Ccircle fill='white' cx='150' cy='600' r='0.8' opacity='.6'/%3E%3Ccircle fill='white' cx='380' cy='680' r='1.2' opacity='.8'/%3E%3Ccircle fill='white' cx='540' cy='720' r='0.9' opacity='.5'/%3E%3Ccircle fill='white' cx='50' cy='750' r='1.1' opacity='.6'/%3E%3Ccircle fill='white' cx='680' cy='760' r='0.7' opacity='.4'/%3E%3Ccircle fill='%2388ccff' cx='200' cy='180' r='1.4' opacity='.6'/%3E%3Ccircle fill='%23cc88ff' cx='600' cy='300' r='1.0' opacity='.5'/%3E%3Ccircle fill='%2388ccff' cx='420' cy='560' r='1.2' opacity='.6'/%3E%3C/svg%3E")!important;
  background-size:cover,cover,cover,cover,800px 800px!important;
}}
[data-testid="stHeader"]{{background:transparent!important}}
[data-testid="stSidebar"]{{background:rgba(0,5,20,0.95)!important;border-right:1px solid rgba(0,150,255,0.15)!important}}
.block-container{{padding:2rem 3rem 3rem!important;max-width:1400px}}
section[data-testid="stSidebar"] .block-container{{padding:1.5rem 1rem!important}}
.app-topbar{{
  display:flex;align-items:center;justify-content:space-between;
  padding:12px 0 20px;margin-bottom:8px;
  border-bottom:1px solid rgba(255,255,255,0.06);
}}
.app-logo{{
  font-family:'Orbitron',monospace;font-size:.95rem;font-weight:900;
  color:#22d3ee;display:flex;align-items:center;gap:8px;
}}
.hero{{text-align:center;padding:48px 20px 36px;position:relative}}
.hero-glow{{position:absolute;top:0;left:50%;transform:translateX(-50%);width:600px;height:300px;background:radial-gradient(ellipse,rgba(0,120,255,0.15) 0%,transparent 70%);pointer-events:none}}
.hero-eyebrow{{display:inline-block;font-family:'Orbitron',monospace;font-size:0.7rem;font-weight:700;letter-spacing:0.25em;color:#22d3ee;background:rgba(34,211,238,0.08);border:1px solid rgba(34,211,238,0.25);padding:5px 18px;border-radius:20px;margin-bottom:20px;text-transform:uppercase}}
.hero-title{{font-family:'Orbitron',monospace;font-size:clamp(1.6rem,4vw,3rem);font-weight:900;line-height:1.1;color:#ffffff;margin:0 0 14px;text-shadow:0 0 40px rgba(0,150,255,0.4)}}
.hero-title span{{color:#22d3ee}}
.hero-subtitle{{font-size:.95rem;color:#94a3b8;max-width:560px;margin:0 auto 28px;line-height:1.7}}
.hero-badges{{display:flex;gap:10px;justify-content:center;flex-wrap:wrap;margin-bottom:10px}}
.hbadge{{font-size:.72rem;font-weight:500;padding:5px 14px;border-radius:20px;border:1px solid;letter-spacing:.03em}}
.hbadge.blue{{color:#7dd3fc;border-color:rgba(125,211,252,.3);background:rgba(125,211,252,.06)}}
.hbadge.teal{{color:#5eead4;border-color:rgba(94,234,212,.3);background:rgba(94,234,212,.06)}}
.hbadge.purple{{color:#c4b5fd;border-color:rgba(196,181,253,.3);background:rgba(196,181,253,.06)}}
.hbadge.pink{{color:#f9a8d4;border-color:rgba(249,168,212,.3);background:rgba(249,168,212,.06)}}
.hbadge.amber{{color:#fcd34d;border-color:rgba(252,211,77,.3);background:rgba(252,211,77,.06)}}
.stat-row{{display:grid;grid-template-columns:repeat(4,1fr);gap:14px;margin-bottom:28px}}
.stat-card{{background:rgba(255,255,255,0.03);border:1px solid rgba(255,255,255,0.07);border-radius:16px;padding:20px 16px;text-align:center;transition:border-color .3s,transform .2s;position:relative;overflow:hidden}}
.stat-card::before{{content:'';position:absolute;top:0;left:0;right:0;height:2px}}
.stat-card.sc-blue::before{{background:linear-gradient(90deg,transparent,#3b82f6,transparent)}}
.stat-card.sc-teal::before{{background:linear-gradient(90deg,transparent,#14b8a6,transparent)}}
.stat-card.sc-purple::before{{background:linear-gradient(90deg,transparent,#8b5cf6,transparent)}}
.stat-card.sc-pink::before{{background:linear-gradient(90deg,transparent,#ec4899,transparent)}}
.stat-card:hover{{border-color:rgba(255,255,255,0.15);transform:translateY(-3px)}}
.stat-icon{{font-size:1.8rem;margin-bottom:8px}}
.stat-num{{font-family:'Orbitron',monospace;font-size:1.6rem;font-weight:700;color:#fff;margin-bottom:4px}}
.stat-lbl{{font-size:0.75rem;color:#64748b;text-transform:uppercase;letter-spacing:.08em}}
.sec-title{{font-family:'Orbitron',monospace;font-size:0.75rem;font-weight:700;color:#22d3ee;letter-spacing:.2em;text-transform:uppercase;margin:28px 0 14px;display:flex;align-items:center;gap:10px}}
.sec-title::after{{content:'';flex:1;height:1px;background:linear-gradient(90deg,rgba(34,211,238,.3),transparent)}}
.upload-zone{{border:1.5px dashed rgba(34,211,238,0.35);border-radius:16px;padding:28px 20px;text-align:center;background:rgba(34,211,238,0.03);margin-bottom:20px}}
.upload-zone p{{color:#475569;font-size:.85rem;margin-top:6px}}
.chat-area{{background:rgba(0,0,0,0.4);border:1px solid rgba(255,255,255,0.06);border-radius:20px;padding:20px;margin-bottom:16px;backdrop-filter:blur(8px)}}
.bubble-user{{display:flex;justify-content:flex-end;margin-bottom:14px}}
.bubble-user-inner{{background:linear-gradient(135deg,rgba(29,78,216,.7),rgba(124,58,237,.7));border:1px solid rgba(139,92,246,.3);color:#fff!important;border-radius:18px 18px 4px 18px;padding:12px 18px;max-width:72%;font-size:.88rem;line-height:1.6;backdrop-filter:blur(8px)}}
.bubble-user-label{{font-size:.68rem;color:rgba(255,255,255,.55)!important;font-weight:600;margin-bottom:3px;text-align:right;font-family:'Orbitron',monospace;letter-spacing:.05em}}
.bubble-bot{{display:flex;justify-content:flex-start;margin-bottom:4px}}
.bubble-bot-inner{{background:rgba(15,23,42,0.8);border:1px solid rgba(34,211,238,0.2);border-top:2px solid #22d3ee;color:#e2e8f0!important;border-radius:4px 18px 18px 18px;padding:14px 18px;max-width:84%;font-size:.85rem;line-height:1.7;backdrop-filter:blur(8px)}}
.bubble-bot-label{{font-size:.68rem;color:#22d3ee!important;font-weight:700;margin-bottom:5px;letter-spacing:.1em;font-family:'Orbitron',monospace}}
.src-pdf{{display:inline-block;background:rgba(34,211,238,.1);border:1px solid rgba(34,211,238,.3);color:#22d3ee;font-size:.65rem;padding:1px 7px;border-radius:8px;margin:0 3px 5px 0;font-weight:600}}
.src-ai{{display:inline-block;background:rgba(139,92,246,.1);border:1px solid rgba(139,92,246,.3);color:#a78bfa;font-size:.65rem;padding:1px 7px;border-radius:8px;margin:0 3px 5px 0;font-weight:600}}
.input-dock{{background:rgba(0,5,20,0.7);border:1px solid rgba(34,211,238,0.2);border-radius:16px;padding:16px 18px;backdrop-filter:blur(12px)}}
.alert-box{{background:rgba(251,146,60,.08);border:1px solid rgba(251,146,60,.3);border-left:3px solid #f97316;color:#fed7aa!important;padding:10px 14px;border-radius:0;margin:5px 0;font-size:.85rem}}
.success-box{{background:rgba(34,197,94,.08);border:1px solid rgba(34,197,94,.3);border-left:3px solid #22c55e;color:#bbf7d0!important;padding:10px 14px;border-radius:0;font-size:.85rem}}
.qr-wrap{{text-align:center;background:rgba(255,255,255,0.03);border:1px solid rgba(255,255,255,0.08);border-radius:16px;padding:20px 14px}}
.qr-wrap img{{border-radius:10px}}
.qr-label{{font-size:.72rem;color:#64748b;margin-top:8px;font-family:'Orbitron',monospace;letter-spacing:.08em}}
.stTextInput>div>div>input{{background:rgba(0,5,20,0.6)!important;border:1.5px solid rgba(34,211,238,0.25)!important;color:#e2e8f0!important;border-radius:10px!important;font-size:.88rem!important}}
.stTextInput>div>div>input:focus{{border-color:rgba(34,211,238,0.6)!important;box-shadow:0 0 0 3px rgba(34,211,238,0.1)!important}}
.stTextInput>div>div>input::placeholder{{color:#334155!important}}
.stButton>button{{
  background:linear-gradient(135deg,rgba(29,78,216,.8),rgba(124,58,237,.8))!important;
  color:#ffffff!important;
  -webkit-text-fill-color:#ffffff!important;
  border:1px solid rgba(139,92,246,.4)!important;
  border-radius:10px!important;
  font-weight:600!important;
  font-family:'Inter',sans-serif!important;
  transition:all .2s!important;
}}
.stButton>button p,
.stButton>button span{{
  color:#ffffff!important;
  -webkit-text-fill-color:#ffffff!important;
}}
.stButton>button:hover{{
  background:linear-gradient(135deg,rgba(29,78,216,1),rgba(124,58,237,1))!important;
  box-shadow:0 0 20px rgba(139,92,246,.4)!important;
  color:#ffffff!important;
  -webkit-text-fill-color:#ffffff!important;
}}
div[data-testid="stFileUploader"]{{background:transparent!important}}
.stSelectbox>div>div{{background:rgba(0,5,20,.7)!important;border:1px solid rgba(34,211,238,.2)!important;color:#e2e8f0!important;border-radius:10px!important}}
label,.stSelectbox label,.stSlider label{{color:#94a3b8!important;font-size:.82rem!important}}
p,li,span{{color:#94a3b8}}
h1,h2,h3{{color:#f1f5f9!important}}
</style>
""", unsafe_allow_html=True)

    # ── APP TOP BAR with back button ──────────────────────────────────────────
    st.markdown('<div class="app-topbar">', unsafe_allow_html=True)
    col_back, col_logo, col_spacer = st.columns([1, 4, 1])
    with col_back:
        if st.button("← Welcome"):
            st.session_state.page = "welcome"
            st.rerun()
    with col_logo:
        st.markdown('<div class="app-logo">🏥 MedAI — Medical AI Assistant</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

    # ── HERO ─────────────────────────────────────────────────────────────────
    st.markdown("""
<div class="hero">
    <div class="hero-glow"></div>
    <div class="hero-eyebrow">AI · RAG · Voice · Multilingual</div>
    <h1 class="hero-title">Voice-Enabled <span>Medical AI</span> Document QA</h1>
    <p class="hero-subtitle">Upload your report · Ask in Hindi or English · Get AI-powered answers with voice output</p>
    <div class="hero-badges">
        <span class="hbadge blue">📄 PDF Extraction</span>
        <span class="hbadge teal">🤖 LLaMA 3.3 AI</span>
        <span class="hbadge purple">🎤 Voice I/O</span>
        <span class="hbadge pink">🌐 Hindi + English</span>
        <span class="hbadge amber">⚠️ Health Alerts</span>
    </div>
</div>
""", unsafe_allow_html=True)

    # ── STAT CARDS ────────────────────────────────────────────────────────────
    st.markdown("""
<div class="stat-row">
    <div class="stat-card sc-blue"><div class="stat-icon">🔬</div><div class="stat-num">40+</div><div class="stat-lbl">Lab Tests</div></div>
    <div class="stat-card sc-teal"><div class="stat-icon">🧠</div><div class="stat-num">RAG</div><div class="stat-lbl">FAISS + MiniLM</div></div>
    <div class="stat-card sc-purple"><div class="stat-icon">🌐</div><div class="stat-num">2</div><div class="stat-lbl">Languages</div></div>
    <div class="stat-card sc-pink"><div class="stat-icon">⚡</div><div class="stat-num">Free</div><div class="stat-lbl">Groq API</div></div>
</div>
""", unsafe_allow_html=True)

    # ── SIDEBAR ───────────────────────────────────────────────────────────────
    with st.sidebar:
        st.markdown(f"""
        <div class="qr-wrap">
            <img src="data:image/png;base64,{qr_b64}" width="160"/>
            <div class="qr-label">SCAN TO OPEN APP</div>
            <div style="font-size:.65rem;color:#334155;margin-top:4px">voice-enabled-medical-ai<br>-document-app.streamlit.app</div>
        </div>
        """, unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown('<div class="sec-title">Settings</div>', unsafe_allow_html=True)
        language_pref = st.selectbox("🌐 Language", ["Auto-Detect", "English", "Hindi"])
        voice_speed   = st.selectbox("🔊 Voice Speed", ["Normal", "Slow"])
        st.markdown('<div class="sec-title">API Keys</div>', unsafe_allow_html=True)
        st.markdown('<p style="font-size:11px;color:#22d3ee;font-weight:600;margin-bottom:4px">🆓 GROQ — FREE</p>', unsafe_allow_html=True)
        groq_input = st.text_input("Groq Key", type="password", placeholder="gsk_...", value=st.session_state.groq_key, key="gk")
        if groq_input and groq_input != st.session_state.groq_key:
            st.session_state.groq_key = groq_input.strip()
            os.environ["GROQ_API_KEY"]  = groq_input.strip()
            st.rerun()
        if st.session_state.groq_key:
            st.success("✅ Groq active (FREE)")
        else:
            st.markdown('<a href="https://console.groq.com" target="_blank" style="font-size:11px;color:#22d3ee">👉 Get free key</a>', unsafe_allow_html=True)
        st.markdown('<p style="font-size:11px;color:#a78bfa;font-weight:600;margin:8px 0 4px">💳 Anthropic (optional)</p>', unsafe_allow_html=True)
        anth_input = st.text_input("Anthropic Key", type="password", placeholder="sk-ant-...", value=st.session_state.anthropic_key, key="ak")
        if anth_input and anth_input != st.session_state.anthropic_key:
            st.session_state.anthropic_key = anth_input.strip()
            os.environ["ANTHROPIC_API_KEY"] = anth_input.strip()
            st.rerun()
        if st.session_state.anthropic_key:
            st.success("✅ Anthropic active")
        if not st.session_state.groq_key and not st.session_state.anthropic_key:
            st.warning("⚠️ Add Groq key to enable AI")
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
        st.session_state.debug_mode = st.checkbox("🔍 Debug mode", value=st.session_state.debug_mode)

    # Re-register full doc text on every rerun
    if st.session_state.doc_processed and st.session_state.clean_text:
        set_full_document(st.session_state.clean_text)

    # ── UPLOAD ────────────────────────────────────────────────────────────────
    st.markdown('<div class="sec-title">📄 Upload Medical Report</div>', unsafe_allow_html=True)
    st.markdown('<div class="upload-zone">', unsafe_allow_html=True)
    uploaded_file = st.file_uploader("PDF", type=["pdf"], label_visibility="collapsed")
    st.markdown('<p>📋 Drop your lab report PDF · Hindi & English supported · Processed locally</p></div>', unsafe_allow_html=True)

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
            st.session_state.clean_text    = clean_text
            set_full_document(clean_text)
            os.unlink(tmp_path)
        st.markdown('<div class="success-box">✅ Document processed — ask your questions below!</div>', unsafe_allow_html=True)

    # ── CHAT ─────────────────────────────────────────────────────────────────
    if st.session_state.doc_processed:
        history     = st.session_state.chat_history.get_history()
        audio_files = st.session_state.audio_files

        if history:
            st.markdown('<div class="sec-title">🗨️ Conversation</div>', unsafe_allow_html=True)
            st.markdown('<div class="chat-area">', unsafe_allow_html=True)
            for i, turn in enumerate(history):
                st.markdown(f"""
                <div class="bubble-user"><div class="bubble-user-inner">
                    <div class="bubble-user-label">YOU</div>{turn['user']}
                </div></div>""", unsafe_allow_html=True)
                answer_html = _re.sub(r'\*\*(.*?)\*\*', r'<b>\1</b>', turn['bot'].replace('\n','<br>'))
                st.markdown(f"""
                <div class="bubble-bot"><div class="bubble-bot-inner">
                    <div class="bubble-bot-label">🏥 MEDICAL AI ASSISTANT</div>
                    <span class="src-pdf">📄 PDF Extracted</span>
                    <span class="src-ai">🤖 AI Enriched</span><br><br>
                    {answer_html}
                </div></div>""", unsafe_allow_html=True)
                if i < len(audio_files) and audio_files[i] and os.path.exists(audio_files[i]):
                    st.audio(audio_files[i], format="audio/mp3")
            st.markdown('</div>', unsafe_allow_html=True)

        if st.session_state.last_alerts:
            st.markdown('<div class="sec-title">⚠️ Medical Alerts</div>', unsafe_allow_html=True)
            for alert in st.session_state.last_alerts:
                st.markdown(f'<div class="alert-box">{alert}</div>', unsafe_allow_html=True)

        st.markdown('<div class="sec-title">💬 Ask Your Question</div>', unsafe_allow_html=True)
        st.markdown('<div class="input-dock">', unsafe_allow_html=True)
        col1, col2, col3 = st.columns([5, 1, 1])
        with col1:
            user_text = st.text_input("q", label_visibility="collapsed",
                placeholder="Ask about your report · medicine · food · symptoms · e.g. 'What is my WBC?' · 'Explain Paracetamol'",
                key="user_input")
        with col2:
            voice_btn = st.button("🎤", use_container_width=True)
        with col3:
            send_btn  = st.button("Send ➤", use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

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

        if query and query != st.session_state.last_query:
            st.session_state.last_query = query
            with st.spinner("🧠 Extracting from PDF + consulting AI..."):
                lang = detect_language(query)
                if language_pref != "Auto-Detect":
                    lang = "hi" if language_pref == "Hindi" else "en"
                query_en = translate_to_english(query, lang)
                q_emb    = embed_query(query_en)
                _q = query_en.lower()
                _top_k = min(10, len(st.session_state.chunks)) if any(w in _q for w in ["summary","all","overall","complete","full report"]) else min(6, len(st.session_state.chunks))
                top_chunks = search_faiss(st.session_state.faiss_index, st.session_state.chunks, q_emb, top_k=_top_k, query_text=query_en)
                alerts     = check_medical_values(" ".join(top_chunks))
                st.session_state.last_alerts = alerts
                if st.session_state.debug_mode:
                    from rag_pipeline import debug_chunks
                    st.expander("🔍 Raw PDF text retrieved").text(debug_chunks(top_chunks))
                answer_en = generate_answer(query_en, top_chunks, st.session_state.chat_history.get_context())
                if lang == "hi":
                    from language_detector import translate_to_hindi
                    answer_display = translate_to_hindi(answer_en)
                else:
                    answer_display = answer_en
                st.session_state.chat_history.add(query, answer_display)
                tts_path = text_to_speech(answer_display[:500], lang=lang, slow=(voice_speed=="Slow"))
                st.session_state.audio_files.append(tts_path)
            st.rerun()

    else:
        # No PDF — general health chatbot
        st.markdown("""
        <div style="text-align:center;padding:28px 20px 20px;background:rgba(0,0,0,0.3);border:1px dashed rgba(34,211,238,0.2);border-radius:20px;margin-bottom:20px">
            <div style="font-size:3rem;margin-bottom:10px">📤</div>
            <div style="font-family:'Orbitron',monospace;font-size:1rem;color:#22d3ee;font-weight:700;margin-bottom:6px">Upload a Medical PDF above to ask about your report</div>
            <div style="font-size:.82rem;color:#475569">Or use the Health Chatbot below for general health questions</div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown('<div class="sec-title">💬 General Health Chatbot</div>', unsafe_allow_html=True)
        st.markdown("""
        <div style="background:rgba(139,92,246,0.08);border:1px solid rgba(139,92,246,0.25);border-radius:12px;padding:12px 16px;margin-bottom:14px;font-size:.82rem;color:#a78bfa">
            🤖 Ask anything about health, diet, food, recovery, symptoms, and lifestyle — no PDF needed!
            <br>Examples: "What food for low hemoglobin?" · "Explain Paracetamol 500mg" · "How to recover from high cholesterol?"
        </div>
        """, unsafe_allow_html=True)

        gen_history = st.session_state.general_chat_history.get_history()
        gen_audio   = st.session_state.general_audio_files

        if gen_history:
            st.markdown('<div class="chat-area">', unsafe_allow_html=True)
            for i, turn in enumerate(gen_history):
                st.markdown(f"""
                <div class="bubble-user"><div class="bubble-user-inner"><div class="bubble-user-label">YOU</div>{turn['user']}</div></div>""", unsafe_allow_html=True)
                answer_html = _re.sub(r'\*\*(.*?)\*\*', r'<b>\1</b>', turn['bot'].replace('\n','<br>'))
                st.markdown(f"""
                <div class="bubble-bot"><div class="bubble-bot-inner">
                    <div class="bubble-bot-label">🤖 HEALTH ASSISTANT</div>
                    <span class="src-ai">🤖 AI Powered</span><br><br>{answer_html}
                </div></div>""", unsafe_allow_html=True)
                if i < len(gen_audio) and gen_audio[i] and os.path.exists(gen_audio[i]):
                    st.audio(gen_audio[i], format="audio/mp3")
            st.markdown('</div>', unsafe_allow_html=True)

        st.markdown('<div class="input-dock">', unsafe_allow_html=True)
        gc1, gc2 = st.columns([5, 1])
        with gc1:
            gen_input = st.text_input("gen_q", label_visibility="collapsed",
                placeholder="Ask any health question: food, diet, recovery, symptoms, medicine...",
                key="gen_input")
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
                system = """You are a clinical nutritionist and health AI assistant.
For food questions: every food MUST have symptom + biological reason + quantity.
For medicine: full structured explanation with dosage, cautions, interactions.
For general health: specific actionable advice with clear sections.
Always respond in English only. Only Hindi if patient writes Devanagari script."""
                hist_ctx = st.session_state.general_chat_history.get_context()
                user_msg = f"{('Conversation:\n'+hist_ctx+chr(10)) if hist_ctx else ''}Question: {gen_query}\n[Respond in English only]"
                ai_resp = _call_ai(system, user_msg, max_tokens=700)
                if not ai_resp:
                    ai_resp = "⚠️ Please add your Groq API key (free) in the sidebar to enable AI responses.\n\nGet your free key at: https://console.groq.com"
                st.session_state.general_chat_history.add(gen_query, ai_resp)
                tts_path = text_to_speech(ai_resp[:500], lang="en", slow=False)
                st.session_state.general_audio_files.append(tts_path)
            st.rerun()

        if gen_history:
            if st.button("🗑️ Clear Health Chat", key="clear_gen"):
                st.session_state.general_chat_history.clear()
                st.session_state.general_audio_files = []
                st.session_state.general_last_query = ""
                st.rerun()
