import streamlit as st
from pathlib import Path
from src.chatbot import RAGChatbot

st.set_page_config(
    page_title="RAG Q&A Chatbot",
    layout="wide",
    page_icon="💎"
)

# --- STRONG VISUALS THROUGH ADVANCED CSS ---
st.markdown("""
<style>
body, .stApp {background: linear-gradient(120deg, #e0e7ff, #f8fafc 60%, #ccfbf1 100%) !important;}
.header-title {
    font-size: 2.8rem;
    font-family: "Montserrat Black", "Montserrat", sans-serif;
    letter-spacing: 3px;
    font-weight: 900;
    background: linear-gradient(90deg, #3b82f6, #38bdf8 60%, #0ea5e9 100%);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    margin-bottom: 0.28em; margin-top: 1.2em;
    text-align: center;
    text-shadow: 1px 3px 10px #4f8cff33;
}
.subtitle {
    color: #265d88; font-size: 1.23rem; margin-bottom: 2.0em;
    text-align: center; letter-spacing: 1px;
}
.chat-row { display: flex; align-items: flex-end; margin-bottom: 1.7em;}
.avatar { width: 42px; height: 42px; border-radius: 50%; margin-right: 1.3em; box-shadow: 0 2px 12px #3b82f61a;}
.chat-bubble-user {
    background: linear-gradient(120deg, #3b82f6, #60a5fa 80%);
    color: #fff; border-radius: 16px 18px 4px 18px;
    padding: 1.2em 1.3em; font-size: 1.08rem; max-width: 70%;
    box-shadow: 0 4px 24px #3b82f630, 0 1px 8px #4f8cff0d;
    margin-left: auto; animation: popIn .35s cubic-bezier(.43,1.16,.77,1.05);
}
.chat-bubble-bot {
    background: rgba(255,255,255,0.82);
    color: #232946; border-radius: 18px 16px 18px 6px;
    padding: 1.18em 1.22em; font-size: 1.08rem; max-width: 70%;
    border-left: 4px solid #34d399; margin-right: auto;
    box-shadow: 0 1px 14px #60a5fa33, 0 1px 8px #6ee7b70d;
    animation: popIn .35s cubic-bezier(.51,1.2,.87,1.07);
}
@keyframes popIn {
    0% {transform: scale(0.6) translateY(30px); opacity: 0;}
    100% {transform: scale(1) translateY(0); opacity: 1;}
}
.source-card {
    backdrop-filter: blur(9px);
    background: rgba(207,237,255,0.51);
    border-radius: 14px;
    font-size: 1.01rem; color: #23577d; margin: 1.15em 0 0.64em 70px;
    padding: 0.74em 1.3em 0.72em 1.1em;
    border-left: 4px solid #36b3ab;
    box-shadow: 0 8px 20px #bbf7d033;
}
.input-area {
    background: rgba(255,255,255,0.82);
    border-radius: 14px; padding: 1.23em 2.1em;
    box-shadow: 0 2px 24px #3b82f628, 0 2px 8px #0ea5e91a;
    margin-bottom: 1.6em;
}
.stButton>button {
    font-weight: 800; background: linear-gradient(90deg,#4f8cff,#22d3eb 80%);
    border-radius: 22px; min-width: 150px; font-size: 1.12rem;
    color: white; letter-spacing: 1.2px;
    border: none; box-shadow: 0 1px 10px #bae6fd44;
    transition: box-shadow .12s, transform .11s;
}
.stButton>button:hover {
    background: linear-gradient(90deg,#38bdf8,#4f8cff 90%);
    box-shadow: 0 4px 16px #38bdf899, 0 1px 16px #bae6fd55;
    transform: translateY(-2px) scale(1.01);
}
div[role="status"] > div, .stSpinner > div {
    color: #000 !important;
}
</style>
""", unsafe_allow_html=True)

AVATAR_USER = "https://cdn-icons-png.flaticon.com/512/2202/2202112.png"          # Optionally change to your logo or another URL
AVATAR_BOT = "https://cdn-icons-png.flaticon.com/512/4712/4712035.png"           # Optionally change to your logo or another URL

# --- Header ---
st.markdown('<div class="header-title">💎 RAG Q&A Chatbot</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">🚀 Powered by Retrieval-Augmented Generation & Local LLMs</div>', unsafe_allow_html=True)

# --- Session State ---
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

DATA_PATH = Path(__file__).parent / "data" / "Training_Dataset.csv"

if "chatbot" not in st.session_state:
    with st.spinner("🌀 Warming up your AI engine..."):
        st.session_state.chatbot = RAGChatbot(csv_path=DATA_PATH)

# --- Chat Input Area ---
with st.form(key="chat_form", clear_on_submit=True):
    with st.container():
        st.markdown('<div class="input-area">', unsafe_allow_html=True)
        query = st.text_input(
            "Ask a question about loan approvals:",
            placeholder="e.g., What increases loan approval chances for self-employed women?",
            label_visibility="collapsed"
        )
        st.markdown('</div>', unsafe_allow_html=True)
        submitted = st.form_submit_button("🧠 Ask Now")

if submitted and query.strip():
    with st.spinner("✨ Thinking... Generating answer..."):
        response = st.session_state.chatbot.ask(query)
    st.session_state.chat_history.append({
        "role": "user",
        "message": query,
        "sources": None
    })
    st.session_state.chat_history.append({
        "role": "bot",
        "message": response["answer"],
        "sources": response.get("sources", [])
    })

# --- Chat History Rendering ---
for entry in st.session_state.chat_history:
    if entry["role"] == "user":
        st.markdown(
            f"""<div class="chat-row" style="justify-content:flex-end;">
                <div class="chat-bubble-user">{entry["message"]}</div>
                <img src="{AVATAR_USER}" class="avatar"/>
            </div>""",
            unsafe_allow_html=True,
        )
    elif entry["role"] == "bot":
        st.markdown(
            f"""<div class="chat-row">
                <img src="{AVATAR_BOT}" class="avatar"/>
                <div class="chat-bubble-bot">{entry["message"]}</div>
            </div>""",
            unsafe_allow_html=True,
        )
        sources = entry.get("sources", [])
        if sources:
            st.markdown('<div class="source-card"><b>🔎 Sources Included</b><br>', unsafe_allow_html=True)
            for meta in sources:
                info = ", ".join([
                    f"<span style='color:#19428c; font-weight:600;'>{k.replace('_',' ').title()}:</span> "
                    f"<span style='color:#228e7d; font-weight:600;'>{v}</span>"
                    for k, v in meta.items()
                ])

                st.markdown(f"- {info}", unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)

# --- Footer ---
st.markdown(
    "<hr><center style='color: #4f8cff; font-size: 0.99rem; opacity:0.72;'>"
    "✨ Built with Streamlit, FAISS, Sentence-Transformers & Hugging Face 💙"
    "</center>", unsafe_allow_html=True
)
