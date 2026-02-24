import streamlit as st
from datetime import datetime

st.set_page_config(
    page_title="Phoenix Cyber Security Bot",
    page_icon="🔥",
    layout="centered"
)

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Share+Tech+Mono&family=Orbitron:wght@400;700;900&family=Rajdhani:wght@400;600;700&display=swap');

    html, body, [data-testid="stAppViewContainer"] {
        background-color: #080010 !important;
        font-family: 'Rajdhani', sans-serif;
    }

    [data-testid="stAppViewContainer"]::before {
        content: '';
        position: fixed;
        inset: 0;
        background:
            radial-gradient(ellipse at 0% 0%,    rgba(255,80,0,0.18)  0%, transparent 45%),
            radial-gradient(ellipse at 100% 0%,  rgba(120,0,255,0.2)  0%, transparent 45%),
            radial-gradient(ellipse at 50% 100%, rgba(80,0,180,0.15)  0%, transparent 50%),
            radial-gradient(ellipse at 100% 100%,rgba(255,80,0,0.10)  0%, transparent 40%);
        pointer-events: none;
        z-index: 0;
    }

    [data-testid="stAppViewContainer"] > * { position: relative; z-index: 1; }
    [data-testid="stHeader"]  { background: transparent !important; }
    [data-testid="stToolbar"] { display: none; }
    footer { visibility: hidden; }

    /* ── Background Shapes ── */
    .bg-shapes { position: fixed; inset: 0; pointer-events: none; overflow: hidden; z-index: 0; }
    .shape { position: absolute; }

    .shape-hex-tl {
        top: -80px; left: -80px; width: 320px; height: 320px;
        background: conic-gradient(from 0deg, #ff5500, #9900ff, #ff5500);
        clip-path: polygon(50% 0%,100% 25%,100% 75%,50% 100%,0% 75%,0% 25%);
        animation: spin-slow 20s linear infinite; opacity: 0.09;
    }
    .shape-hex-tr {
        top: 40px; right: -60px; width: 220px; height: 220px;
        background: conic-gradient(from 90deg, #9900ff, #ff5500, #9900ff);
        clip-path: polygon(50% 0%,100% 25%,100% 75%,50% 100%,0% 75%,0% 25%);
        animation: spin-slow 15s linear infinite reverse; opacity: 0.11;
    }
    .shape-tri-bl {
        bottom: 100px; left: -30px; width: 180px; height: 180px;
        background: linear-gradient(135deg, #ff5500, transparent);
        clip-path: polygon(0% 100%, 50% 0%, 100% 100%); opacity: 0.12;
    }
    .shape-dia-br {
        bottom: 60px; right: -20px; width: 160px; height: 160px;
        background: linear-gradient(45deg, #9900ff, #ff5500);
        clip-path: polygon(50% 0%,100% 50%,50% 100%,0% 50%);
        animation: pulse-shape 4s ease-in-out infinite; opacity: 0.10;
    }
    .shape-sq-1 {
        top: 35%; left: 2%; width: 60px; height: 60px;
        border: 2px solid rgba(255,85,0,0.3);
        animation: spin-slow 8s linear infinite;
    }
    .shape-sq-2 {
        top: 60%; right: 3%; width: 45px; height: 45px;
        border: 2px solid rgba(153,0,255,0.3);
        animation: spin-slow 6s linear infinite reverse;
    }
    .shape-circle-1 {
        top: 15%; right: 8%; width: 100px; height: 100px;
        border: 1px solid rgba(255,85,0,0.25); border-radius: 50%;
        animation: pulse-shape 5s ease-in-out infinite;
    }
    .shape-circle-2 {
        bottom: 25%; left: 5%; width: 70px; height: 70px;
        border: 1px solid rgba(153,0,255,0.25); border-radius: 50%;
        animation: pulse-shape 3.5s ease-in-out infinite 1s;
    }
    .shape-grid {
        top: 0; right: 0; width: 200px; height: 200px;
        background-image: radial-gradient(rgba(255,85,0,0.2) 1px, transparent 1px);
        background-size: 18px 18px; opacity: 0.4;
    }
    .shape-grid-bl {
        bottom: 0; left: 0; width: 180px; height: 180px;
        background-image: radial-gradient(rgba(153,0,255,0.2) 1px, transparent 1px);
        background-size: 18px 18px; opacity: 0.4;
    }

    @keyframes spin-slow {
        from { transform: rotate(0deg); }
        to   { transform: rotate(360deg); }
    }
    @keyframes pulse-shape {
        0%,100% { opacity: 0.07; transform: scale(1); }
        50%      { opacity: 0.18; transform: scale(1.08); }
    }

    /* ── Nav Bar ── */
    .nav-bar {
        display: flex; align-items: center; justify-content: space-between;
        padding: 10px 0 18px;
        border-bottom: 1px solid rgba(255,85,0,0.15);
        margin-bottom: 28px;
    }
    .nav-brand {
        font-family: 'Orbitron', monospace; font-size: 0.75rem; font-weight: 700;
        letter-spacing: 0.2em; color: #ff5500;
        text-shadow: 0 0 10px rgba(255,85,0,0.5);
    }
    .nav-status {
        display: flex; align-items: center; gap: 6px;
        font-family: 'Share Tech Mono', monospace; font-size: 0.65rem;
        color: #5a3a6a; letter-spacing: 0.1em;
    }
    .nav-dot {
        width: 7px; height: 7px; border-radius: 50%;
        background: #00ff88; box-shadow: 0 0 8px #00ff88;
        animation: blink-dot 2s ease-in-out infinite;
    }
    @keyframes blink-dot { 0%,100%{opacity:1;} 50%{opacity:0.3;} }

    .nav-version {
        font-family: 'Share Tech Mono', monospace; font-size: 0.62rem;
        color: #9900ff; letter-spacing: 0.12em;
        border: 1px solid rgba(153,0,255,0.3); padding: 2px 8px; border-radius: 2px;
    }

    /* ── Hero ── */
    .hero { text-align: center; padding: 10px 0 26px; }
    .hero-eye {
        font-size: 3.5rem; display: block; margin-bottom: 8px;
        filter: drop-shadow(0 0 20px rgba(255,85,0,0.7));
        animation: float 3s ease-in-out infinite;
    }
    @keyframes float {
        0%,100%{transform:translateY(0);} 50%{transform:translateY(-8px);}
    }
    .hero-title {
        font-family: 'Orbitron', monospace; font-size: 2.6rem; font-weight: 900;
        letter-spacing: 0.08em; margin: 0 0 6px;
        background: linear-gradient(90deg, #ff5500 0%, #cc44ff 50%, #ff5500 100%);
        background-size: 200% auto;
        -webkit-background-clip: text; -webkit-text-fill-color: transparent; background-clip: text;
        animation: shimmer 4s linear infinite;
    }
    @keyframes shimmer {
        0%  { background-position: 0% center; }
        100%{ background-position: 200% center; }
    }
    .hero-sub {
        font-family: 'Share Tech Mono', monospace; font-size: 0.78rem;
        color: #6a4a8a; letter-spacing: 0.18em; margin-bottom: 18px;
    }

    /* ── Stats ── */
    .stats-row {
        display: flex; justify-content: center; gap: 30px;
        margin: 0 auto 24px; max-width: 480px;
    }
    .stat-card {
        flex: 1; background: rgba(255,85,0,0.04);
        border: 1px solid rgba(255,85,0,0.15);
        border-radius: 4px; padding: 10px 6px; text-align: center;
    }
    .stat-card.purple {
        background: rgba(153,0,255,0.04);
        border-color: rgba(153,0,255,0.15);
    }
    .stat-val {
        font-family: 'Orbitron', monospace; font-size: 1.3rem;
        font-weight: 700; color: #ff5500;
    }
    .stat-card.purple .stat-val { color: #cc44ff; }
    .stat-lbl {
        font-family: 'Share Tech Mono', monospace; font-size: 0.58rem;
        color: #4a3060; letter-spacing: 0.1em; margin-top: 2px;
    }

    /* ── Section Label ── */
    .section-label {
        font-family: 'Share Tech Mono', monospace; font-size: 0.62rem;
        color: #9900ff; letter-spacing: 0.18em; text-transform: uppercase;
        margin-bottom: 8px; display: flex; align-items: center; gap: 8px;
    }
    .section-label::after {
        content: ''; flex: 1; height: 1px;
        background: linear-gradient(to right, rgba(153,0,255,0.3), transparent);
    }

    /* ── Chips ── */
    .chips-row { display: flex; flex-wrap: wrap; gap: 8px; margin-bottom: 22px; }
    .chip {
        background: rgba(153,0,255,0.07); border: 1px solid rgba(153,0,255,0.25);
        color: #cc44ff; font-family: 'Share Tech Mono', monospace; font-size: 0.68rem;
        padding: 5px 13px; border-radius: 3px; letter-spacing: 0.06em;
        cursor: default; transition: all 0.2s; position: relative; overflow: hidden;
    }
    .chip::before {
        content: ''; position: absolute; inset: 0;
        background: linear-gradient(90deg, transparent, rgba(255,85,0,0.1), transparent);
        transform: translateX(-100%); transition: transform 0.4s;
    }
    .chip:hover::before { transform: translateX(100%); }
    .chip:hover {
        border-color: rgba(255,85,0,0.4); color: #ff7744;
        background: rgba(255,85,0,0.07);
    }

    /* ── Messages ── */
    .messages-area { padding: 4px 0 8px; min-height: 60px; }
    .msg-row-user {
        display: flex; justify-content: flex-end;
        margin-bottom: 14px; animation: fadein 0.3s ease;
    }
    .msg-row-bot {
        display: flex; justify-content: flex-start; align-items: flex-start;
        gap: 10px; margin-bottom: 14px; animation: fadein 0.3s ease;
    }
    @keyframes fadein {
        from { opacity: 0; transform: translateY(8px); }
        to   { opacity: 1; transform: translateY(0); }
    }
    .user-bubble {
        background: linear-gradient(135deg, #cc3300 0%, #ff5500 60%, #cc44ff 100%);
        color: #fff; padding: 11px 16px;
        border-radius: 14px 14px 3px 14px; max-width: 74%;
        font-size: 0.95rem; line-height: 1.55;
        box-shadow: 0 4px 20px rgba(255,85,0,0.3), 0 0 0 1px rgba(255,255,255,0.05) inset;
        word-break: break-word;
    }
    .bot-bubble {
        background: rgba(18, 6, 30, 0.92); color: #d0b8f0;
        padding: 11px 16px; border-radius: 14px 14px 14px 3px; max-width: 74%;
        font-size: 0.95rem; line-height: 1.55;
        border: 1px solid rgba(153,0,255,0.2);
        border-top: 1px solid rgba(255,85,0,0.15);
        box-shadow: 0 4px 24px rgba(0,0,0,0.4), 0 0 0 1px rgba(255,255,255,0.02) inset;
        word-break: break-word;
    }
    .bot-avatar {
        width: 34px; height: 34px; border-radius: 50%;
        background: linear-gradient(135deg, #1a0030, #0d0015);
        border: 1px solid rgba(255,85,0,0.4);
        display: flex; align-items: center; justify-content: center;
        font-size: 1rem; flex-shrink: 0; margin-top: 2px;
        box-shadow: 0 0 14px rgba(255,85,0,0.3), 0 0 28px rgba(153,0,255,0.2);
    }

    /* ── Dividers ── */
    .gradient-divider {
        height: 1px;
        background: linear-gradient(to right, transparent, rgba(255,85,0,0.4), rgba(153,0,255,0.4), transparent);
        margin: 20px 0;
    }
    .deco-row {
        display: flex; align-items: center; gap: 12px;
        margin: 6px 0 18px; opacity: 0.5;
    }
    .deco-hex {
        width: 18px; height: 18px; background: #ff5500;
        clip-path: polygon(50% 0%,100% 25%,100% 75%,50% 100%,0% 75%,0% 25%);
    }
    .deco-hex.purple { background: #9900ff; }
    .deco-hex.small  { width: 12px; height: 12px; }
    .deco-line {
        flex: 1; height: 1px;
        background: linear-gradient(to right, rgba(255,85,0,0.5), rgba(153,0,255,0.5));
    }
    .deco-diamond {
        width: 14px; height: 14px; background: #cc44ff;
        clip-path: polygon(50% 0%,100% 50%,50% 100%,0% 50%);
    }

    /* ── Input ── */
    .input-section-label {
        font-family: 'Share Tech Mono', monospace; font-size: 0.62rem;
        color: #ff5500; letter-spacing: 0.18em;
        margin-bottom: 8px; display: flex; align-items: center; gap: 8px;
    }
    .input-section-label::after {
        content: ''; flex: 1; height: 1px;
        background: linear-gradient(to right, rgba(255,85,0,0.3), transparent);
    }

    .stTextInput > div > div {
        background: rgba(12, 3, 20, 0.9) !important;
        border: 1px solid rgba(153,0,255,0.3) !important;
        border-radius: 6px !important;
        box-shadow: none !important;
        transition: border-color 0.2s, box-shadow 0.2s !important;
    }
    .stTextInput > div > div:focus-within {
        border-color: rgba(255,85,0,0.6) !important;
        box-shadow: 0 0 0 3px rgba(255,85,0,0.08) !important;
    }
    .stTextInput input {
        background: transparent !important; color: #e8d0ff !important;
        font-family: 'Share Tech Mono', monospace !important;
        font-size: 0.88rem !important; caret-color: #ff5500 !important;
        letter-spacing: 0.03em;
    }
    .stTextInput input::placeholder { color: #3a2050 !important; }

    .stFormSubmitButton > button {
        width: 100% !important;
        background: linear-gradient(135deg, #cc3300, #ff5500 40%, #9900ff) !important;
        color: #fff !important; border: none !important; border-radius: 6px !important;
        padding: 10px 20px !important; font-family: 'Orbitron', monospace !important;
        font-size: 0.75rem !important; font-weight: 700 !important;
        letter-spacing: 0.15em !important; cursor: pointer !important;
        transition: all 0.25s !important;
        box-shadow: 0 4px 20px rgba(255,85,0,0.4), 0 0 30px rgba(153,0,255,0.2) !important;
    }
    .stFormSubmitButton > button:hover {
        background: linear-gradient(135deg, #ff5500, #cc44ff) !important;
        box-shadow: 0 6px 28px rgba(255,85,0,0.55), 0 0 40px rgba(153,0,255,0.35) !important;
        transform: translateY(-2px) !important;
    }
    .stFormSubmitButton > button:active { transform: translateY(0) !important; }

    /* ── Footer ── */
    .cyber-footer {
        text-align: center; padding: 18px 0 10px;
        font-family: 'Share Tech Mono', monospace; font-size: 0.62rem;
        color: #3a2050; letter-spacing: 0.15em;
    }
    .cyber-footer .hl-orange { color: #ff5500; }
    .cyber-footer .hl-purple { color: #9900ff; }

    [data-testid="stForm"] { border: none !important; padding: 0 !important; }

    .empty-state {
        display: flex; flex-direction: column; align-items: center;
        gap: 8px; padding: 30px 0; opacity: 0.25; text-align: center;
    }
    .empty-icon { font-size: 2.8rem; }
    .empty-text {
        font-family: 'Share Tech Mono', monospace; font-size: 0.7rem;
        color: #8a5aaa; letter-spacing: 0.12em;
    }
</style>
""", unsafe_allow_html=True)

# ── Background Shapes ──
st.markdown("""
<div class="bg-shapes">
    <div class="shape shape-hex-tl"></div>
    <div class="shape shape-hex-tr"></div>
    <div class="shape shape-tri-bl"></div>
    <div class="shape shape-dia-br"></div>
    <div class="shape shape-sq-1"></div>
    <div class="shape shape-sq-2"></div>
    <div class="shape shape-circle-1"></div>
    <div class="shape shape-circle-2"></div>
    <div class="shape shape-grid"></div>
    <div class="shape shape-grid-bl"></div>
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────
# Session State
# ─────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []

# ─────────────────────────────────────────
# Response Logic
# ─────────────────────────────────────────
RESPONSES = {
    "phishing": (
        "🎣 <b>Phishing</b> is a social engineering attack where adversaries "
        "impersonate trusted entities via email, SMS, or fake websites to harvest "
        "credentials or deploy malware. Always verify sender domains, hover over "
        "links before clicking, and enable MFA on all accounts."
    ),
    "xss": (
        "💉 <b>Cross-Site Scripting (XSS)</b> lets attackers inject malicious "
        "client-side scripts into pages viewed by other users. Mitigations include "
        "output encoding, Content Security Policy (CSP) headers, and frameworks "
        "that auto-escape HTML by default."
    ),
    "sql injection": (
        "🗄️ <b>SQL Injection</b> manipulates databases by injecting crafted SQL "
        "via input fields. Defend with parameterized queries, ORMs, input "
        "validation, and least-privilege database accounts."
    ),
    "ransomware": (
        "🔒 <b>Ransomware</b> encrypts victim files and demands payment for "
        "decryption keys. Prevention: offline backups (3-2-1 rule), endpoint "
        "detection & response (EDR), network segmentation, and regular patching."
    ),
    "ddos": (
        "🌊 <b>DDoS</b> floods a target with traffic to exhaust resources. "
        "Mitigations include CDN-based traffic scrubbing, rate limiting, "
        "Anycast routing, and WAF rules to filter malicious requests."
    ),
    "firewall": (
        "🛡️ <b>Firewalls</b> filter network traffic based on rule sets. "
        "Next-gen firewalls (NGFW) add deep-packet inspection, IPS, and "
        "application-layer awareness. Always follow least-access principles."
    ),
    "zero day": (
        "🕳️ A <b>Zero-Day</b> is an unpatched, publicly unknown vulnerability. "
        "Defense-in-depth, behavioral EDR/XDR detection, virtual patching "
        "via WAFs, and quick patch management reduce exposure."
    ),
    "vpn": (
        "🔐 A <b>VPN</b> encrypts traffic between client and remote server, "
        "masking IP and protecting data on untrusted networks. Prefer modern "
        "protocols like WireGuard or IKEv2 over legacy VPNs."
    ),
    "malware": (
        "🦠 <b>Malware</b> is software designed to damage or gain unauthorized "
        "access. Categories include viruses, worms, trojans, spyware, and "
        "rootkits. Use AV/EDR, disable macros, and practice safe browsing."
    ),
}

def generate_response(user_input: str) -> str:
    text = user_input.lower()
    for keyword, response in RESPONSES.items():
        if keyword in text:
            return response
    return (
        "🔥 I'm <b>Phoenix</b> — your cybersecurity intelligence core. "
        "Ask me about phishing, XSS, SQL injection, ransomware, DDoS, "
        "zero-days, malware, firewalls, or VPNs."
    )

# ─────────────────────────────────────────
# Nav Bar
# ─────────────────────────────────────────
now       = datetime.now()
msg_count = len(st.session_state.messages)

st.markdown(f"""
<div class="nav-bar">
    <span class="nav-brand">◈ PHOENIX</span>
    <span class="nav-status">
        <span class="nav-dot"></span>ONLINE &nbsp;|&nbsp; {now.strftime("%H:%M:%S")}
    </span>
    <span class="nav-version">v2.4.1</span>
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────
# Hero
# ─────────────────────────────────────────
st.markdown("""
<div class="hero">
    <span class="hero-eye">🔥</span>
    <h1 class="hero-title">PHOENIX CYBER BOT</h1>
    <p class="hero-sub">// ADVANCED CYBERSECURITY LEARNING ASSISTANT //</p>
</div>
""", unsafe_allow_html=True)

# Deco shapes row
st.markdown("""
<div class="deco-row">
    <div class="deco-hex small"></div>
    <div class="deco-hex"></div>
    <div class="deco-diamond"></div>
    <div class="deco-line"></div>
    <div class="deco-diamond"></div>
    <div class="deco-hex purple"></div>
    <div class="deco-hex purple small"></div>
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────
# Stats Row
# ─────────────────────────────────────────
st.markdown(f"""
<div class="stats-row">
    <div class="stat-card">
        <div class="stat-val">09</div>
        <div class="stat-lbl">THREAT TOPICS</div>
    </div>
    <div class="stat-card purple">
        <div class="stat-val">{msg_count:04d}</div>
        <div class="stat-lbl">MESSAGES</div>
    </div>
    <div class="stat-card">
        <div class="stat-val">24/7</div>
        <div class="stat-lbl">INTEL ACTIVE</div>
    </div>
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────
# Quick Topics
# ─────────────────────────────────────────
st.markdown('<div class="section-label">▸ QUICK TOPICS</div>', unsafe_allow_html=True)
st.markdown("""
<div class="chips-row">
    <span class="chip">Phishing</span>
    <span class="chip">XSS</span>
    <span class="chip">SQL Injection</span>
    <span class="chip">Ransomware</span>
    <span class="chip">DDoS</span>
    <span class="chip">Zero-Day</span>
    <span class="chip">Malware</span>
    <span class="chip">Firewall</span>
    <span class="chip">VPN</span>
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────
# Conversation — flat, no box
# ─────────────────────────────────────────
st.markdown('<div class="section-label">▸ CONVERSATION</div>', unsafe_allow_html=True)
st.markdown('<div class="messages-area">', unsafe_allow_html=True)

if not st.session_state.messages:
    st.markdown("""
    <div class="empty-state">
        <div class="empty-icon">🛡️</div>
        <div class="empty-text">[ SYSTEM READY — AWAITING YOUR QUERY ]</div>
    </div>
    """, unsafe_allow_html=True)
else:
    for msg in st.session_state.messages:
        if msg["role"] == "user":
            st.markdown(
                f'<div class="msg-row-user">'
                f'<div class="user-bubble">{msg["content"]}</div>'
                f'</div>',
                unsafe_allow_html=True
            )
        else:
            st.markdown(
                f'<div class="msg-row-bot">'
                f'<div class="bot-avatar">🔥</div>'
                f'<div class="bot-bubble">{msg["content"]}</div>'
                f'</div>',
                unsafe_allow_html=True
            )

st.markdown('</div>', unsafe_allow_html=True)

# ─────────────────────────────────────────
# Input
# ─────────────────────────────────────────
st.markdown('<div class="gradient-divider"></div>', unsafe_allow_html=True)
st.markdown('<div class="input-section-label">▸ ENTER QUERY</div>', unsafe_allow_html=True)

with st.form(key="chat_form", clear_on_submit=True):
    col1, col2 = st.columns([5, 1])
    with col1:
        user_input = st.text_input(
            label="input",
            label_visibility="collapsed",
            placeholder="› Type your cybersecurity question...",
        )
    with col2:
        submit = st.form_submit_button("SEND ▶")

if submit and user_input:
    st.session_state.messages.append({"role": "user",      "content": user_input})
    st.session_state.messages.append({"role": "assistant", "content": generate_response(user_input)})
    st.rerun()

# ─────────────────────────────────────────
# Footer
# ─────────────────────────────────────────
st.markdown("""
<div class="deco-row" style="margin-top:18px;">
    <div class="deco-hex purple small"></div>
    <div class="deco-line"></div>
    <div class="deco-diamond"></div>
    <div class="deco-line"></div>
    <div class="deco-hex small"></div>
</div>
""", unsafe_allow_html=True)

st.markdown(f"""
<div class="cyber-footer">
    ◈ PHOENIX CYBER BOT &nbsp;
    <span class="hl-orange">© {now.year}</span>
    &nbsp;—&nbsp;
    <span class="hl-purple">SECURE</span>
    &nbsp;·&nbsp;
    <span class="hl-orange">EDUCATE</span>
    &nbsp;·&nbsp;
    <span class="hl-purple">DEFEND</span>
    &nbsp; ◈
</div>
""", unsafe_allow_html=True)


