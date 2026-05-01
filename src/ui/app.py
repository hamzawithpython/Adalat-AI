import streamlit as st
import requests
import uuid
import json

API_URL = os.getenv("API_URL", "http://127.0.0.1:8001")

# ── Page Config ───────────────────────────────────────────────
st.set_page_config(
    page_title="Adalat-AI",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Custom CSS ────────────────────────────────────────────────
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 800;
        color: #1F4E79;
        text-align: center;
        padding: 1rem 0;
    }
    .sub-header {
        font-size: 1rem;
        color: #595959;
        text-align: center;
        margin-bottom: 2rem;
    }
    .jurisdiction-badge {
        display: inline-block;
        padding: 4px 12px;
        border-radius: 20px;
        font-size: 0.8rem;
        font-weight: 600;
        margin: 2px;
    }
    .badge-pk { background-color: #006600; color: white; }
    .badge-uk { background-color: #003087; color: white; }
    .badge-de { background-color: #CC0000; color: white; }
    .rights-card {
        background: #f0f7ff;
        border-left: 4px solid #2E75B6;
        padding: 12px 16px;
        margin: 8px 0;
        border-radius: 0 8px 8px 0;
        color: #1a1a1a !important;
    }
    .citation-item {
        background: #f9f9f9;
        border: 1px solid #ddd;
        padding: 8px 12px;
        margin: 4px 0;
        border-radius: 4px;
        font-size: 0.85rem;
        color: #1a1a1a !important;
    }
    .confidence-high { color: #006600; font-weight: 600; }
    .confidence-med  { color: #CC6600; font-weight: 600; }
    .confidence-low  { color: #CC0000; font-weight: 600; }
    .disclaimer {
        background: #fff3cd;
        border: 1px solid #ffc107;
        padding: 10px 14px;
        border-radius: 6px;
        font-size: 0.9rem;
        margin-top: 12px;
        color: #1a1a1a !important;
    }
    .user-message {
        background: #E8F4FD;
        padding: 12px 16px;
        border-radius: 12px 12px 4px 12px;
        margin: 8px 0;
        text-align: right;
        color: #1a1a1a !important;
    }
    .assistant-message {
        background: #F8F9FA;
        padding: 12px 16px;
        border-radius: 12px 12px 12px 4px;
        margin: 8px 0;
        border-left: 3px solid #2E75B6;
        color: #1a1a1a !important;
    }
</style>
""", unsafe_allow_html=True)


# ── Session State ─────────────────────────────────────────────
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())
if "messages" not in st.session_state:
    st.session_state.messages = []
if "last_result" not in st.session_state:
    st.session_state.last_result = None


# ── Sidebar ───────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚖️ Adalat-AI")
    st.markdown("**Legal Assistant for PK/UK/DE Law**")
    st.divider()

    st.markdown("### 🌍 Supported Jurisdictions")
    st.markdown("🟢 **Pakistan** — Constitution + PPC")
    st.markdown("🔵 **United Kingdom** — Tenant Fees Act")
    st.markdown("🔴 **Germany** — BGB §§535–548")
    st.divider()

    st.markdown("### 💬 You can ask in")
    st.markdown("- English")
    st.markdown("- Roman-Urdu (e.g. *mera landlord deposit wapas nahi de raha*)")
    st.markdown("- German (e.g. *Vermieter gibt Kaution nicht zurück*)")
    st.divider()

    st.markdown("### 📋 Sample Questions")
    sample_questions = [
        "What fees can my landlord charge me in the UK?",
        "mera landlord deposit wapas nahi de raha",
        "Can police detain me without charge in Pakistan?",
        "Mein Vermieter gibt meine Kaution nicht zurück",
        "What are my fundamental rights if arrested in Pakistan?",
    ]
    for q in sample_questions:
        if st.button(q[:45] + "..." if len(q) > 45 else q, key=q):
            st.session_state.pending_query = q

    st.divider()
    if st.button("🗑️ Clear Chat"):
        st.session_state.messages = []
        st.session_state.session_id = str(uuid.uuid4())
        st.session_state.last_result = None
        st.rerun()

    st.markdown(f"**Session:** `{st.session_state.session_id[:8]}...`")


# ── Main Header ───────────────────────────────────────────────
st.markdown('<div class="main-header">⚖️ Adalat-AI</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="sub-header">Roman-Urdu & English Legal Assistant · Pakistan · UK · Germany</div>',
    unsafe_allow_html=True
)


# ── Helper Functions ──────────────────────────────────────────
def get_confidence_class(score):
    if score >= 0.6:
        return "confidence-high"
    elif score >= 0.4:
        return "confidence-med"
    return "confidence-low"


def get_jurisdiction_badge(jurisdiction):
    j = str(jurisdiction).replace("Jurisdiction.", "")
    badge_map = {
        "PK": "badge-pk", "UK": "badge-uk", "DE": "badge-de"
    }
    css = badge_map.get(j, "badge-pk")
    return f'<span class="jurisdiction-badge {css}">{j}</span>'


def call_api(query: str) -> dict:
    try:
        response = requests.post(
            f"{API_URL}/ask",
            json={"query": query, "session_id": st.session_state.session_id},
            timeout=120
        )
        if response.status_code == 200:
            return response.json()
        return {"error": f"API error: {response.status_code}"}
    except requests.exceptions.Timeout:
        return {"error": "Request timed out. Please try again."}
    except requests.exceptions.ConnectionError:
        return {"error": "Cannot connect to API. Make sure the server is running."}


def display_result(result: dict):
    if "error" in result:
        st.error(result["error"])
        return

    col1, col2, col3 = st.columns(3)
    with col1:
        j = str(result.get("jurisdiction", "")).replace("Jurisdiction.", "")
        st.markdown(get_jurisdiction_badge(j), unsafe_allow_html=True)
        st.caption("Jurisdiction")
    with col2:
        lang = str(result.get("language", "")).replace("Language.", "")
        st.info(f"🗣️ {lang}")
    with col3:
        conf = result.get("confidence", 0)
        cls = get_confidence_class(conf)
        st.markdown(
            f'<span class="{cls}">📊 Confidence: {conf:.0%}</span>',
            unsafe_allow_html=True
        )

    if result.get("translated_query"):
        st.caption(f"🔄 Translated: *{result['translated_query']}*")

    st.markdown("---")

    # Answer
    st.markdown("### 📜 Legal Answer")
    st.markdown(
        f'<div class="assistant-message">{result.get("answer", "")}</div>',
        unsafe_allow_html=True
    )

    # Rights Cards
    rights = result.get("rights", [])
    if rights:
        st.markdown("### ✅ Your Structured Rights")
        for r in rights:
            with st.expander(f"⚖️ {r.get('right', 'Right')}", expanded=True):
                col_a, col_b = st.columns(2)
                with col_a:
                    st.markdown(f"**Legal Basis:** {r.get('legal_basis', 'N/A')}")
                    if r.get('deadline'):
                        st.markdown(f"**⏰ Deadline:** {r.get('deadline')}")
                    if r.get('obligation'):
                        st.markdown(f"**📋 Obligation:** {r.get('obligation')}")
                with col_b:
                    st.markdown(f"**🔧 Recourse:** {r.get('recourse', 'N/A')}")

    # Citations
    citations = result.get("citations", [])
    if citations:
        st.markdown("### 📚 Citations")
        for c in citations:
            score = c.get("relevance_score", 0)
            cls = get_confidence_class(score)
            st.markdown(
                f'<div class="citation-item">'
                f'📄 <b>{c.get("source")}</b> | Page {c.get("page")} | '
                f'<span class="{cls}">Score: {score:.2f}</span>'
                f'</div>',
                unsafe_allow_html=True
            )

    # Disclaimer
    st.markdown(
        '<div class="disclaimer">⚠️ This is informational only. '
        'Consult a qualified lawyer for legal advice.</div>',
        unsafe_allow_html=True
    )


# ── Chat History Display ──────────────────────────────────────
for msg in st.session_state.messages:
    if msg["role"] == "user":
        st.markdown(
            f'<div class="user-message">🧑 {msg["content"]}</div>',
            unsafe_allow_html=True
        )
    else:
        display_result(msg["content"])


# ── Handle Sidebar Sample Question ───────────────────────────
if "pending_query" in st.session_state:
    query = st.session_state.pending_query
    del st.session_state.pending_query
    st.session_state.messages.append({"role": "user", "content": query})
    with st.spinner("🔍 Searching legal documents..."):
        result = call_api(query)
    st.session_state.messages.append({"role": "assistant", "content": result})
    st.rerun()


# ── Input Box ─────────────────────────────────────────────────
st.markdown("---")
query = st.chat_input("Ask a legal question in English, Roman-Urdu, or German...")

if query:
    st.session_state.messages.append({"role": "user", "content": query})
    with st.spinner("🔍 Searching legal documents and generating answer..."):
        result = call_api(query)
    st.session_state.messages.append({"role": "assistant", "content": result})
    st.rerun()
