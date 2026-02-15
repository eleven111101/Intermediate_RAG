import sys
from pathlib import Path
import yaml
import streamlit as st
import requests

# ------------------------------------------------------------
# Ensure project root is importable
# ------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from rag_project.db_checker_files import SystemChecker

# ------------------------------------------------------------
# Load config
# ------------------------------------------------------------
with open(PROJECT_ROOT / "config.yaml") as f:
    config = yaml.safe_load(f)

# ------------------------------------------------------------
# Page Configuration
# ------------------------------------------------------------
st.set_page_config(
    page_title="Ashwa Riders - Conquer Every Terrain",
    page_icon="🏁",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ------------------------------------------------------------
# Custom CSS for Motorsport Theme
# ------------------------------------------------------------
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;700;900&family=Rajdhani:wght@300;400;600;700&display=swap');
    
    /* Global Styles */
    .stApp {
        background: linear-gradient(135deg, #000000 0%, #1a0000 50%, #0a0000 100%);
        background-attachment: fixed;
    }
    
    /* Hide Streamlit Branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    /* Racing Stripe Background Effect */
    .stApp::before {
        content: '';
        position: fixed;
        top: 0;
        left: 0;
        right: 0;
        bottom: 0;
        background: 
            repeating-linear-gradient(
                45deg,
                transparent,
                transparent 50px,
                rgba(220, 20, 20, 0.03) 50px,
                rgba(220, 20, 20, 0.03) 100px
            );
        pointer-events: none;
        z-index: 0;
    }
    
    /* Header Styling */
    .main-header {
        font-family: 'Orbitron', sans-serif;
        font-size: 4.5rem;
        font-weight: 900;
        text-align: center;
        background: linear-gradient(135deg, #dc143c 0%, #ff0000 50%, #cc0000 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        text-transform: uppercase;
        letter-spacing: 8px;
        margin: 2rem 0 1rem 0;
        text-shadow: 0 0 30px rgba(220, 20, 60, 0.3);
        animation: pulseGlow 3s ease-in-out infinite;
    }
    
    @keyframes pulseGlow {
        0%, 100% { filter: brightness(1); }
        50% { filter: brightness(1.2); }
    }
    
    .tagline {
        font-family: 'Rajdhani', sans-serif;
        font-size: 1.8rem;
        font-weight: 300;
        text-align: center;
        color: #cccccc;
        letter-spacing: 4px;
        margin-bottom: 3rem;
        text-transform: uppercase;
    }
    
    /* Section Headers */
    .section-header {
        font-family: 'Orbitron', sans-serif;
        font-size: 2rem;
        font-weight: 700;
        color: #dc143c;
        text-transform: uppercase;
        letter-spacing: 3px;
        margin: 2rem 0 1rem 0;
        border-left: 5px solid #cc0000;
        padding-left: 15px;
    }
    
    /* Info Cards */
    .info-card {
        background: rgba(30, 30, 30, 0.8);
        border: 2px solid #333;
        border-left: 5px solid #dc143c;
        border-radius: 10px;
        padding: 1.5rem;
        margin: 1rem 0;
        backdrop-filter: blur(10px);
        transition: all 0.3s ease;
    }
    
    .info-card:hover {
        border-color: #ff0000;
        transform: translateX(5px);
        box-shadow: 0 5px 30px rgba(220, 20, 60, 0.3);
    }
    
    .info-card h3 {
        font-family: 'Orbitron', sans-serif;
        color: #ff0000;
        font-size: 1.5rem;
        margin-bottom: 0.5rem;
    }
    
    .info-card p {
        font-family: 'Rajdhani', sans-serif;
        color: #cccccc;
        font-size: 1.1rem;
        line-height: 1.6;
    }
    
    /* Stats Display */
    .stat-box {
        background: linear-gradient(135deg, rgba(220, 20, 60, 0.1) 0%, rgba(204, 0, 0, 0.1) 100%);
        border: 2px solid #dc143c;
        border-radius: 15px;
        padding: 2rem;
        text-align: center;
        margin: 1rem;
        transition: all 0.3s ease;
    }
    
    .stat-box:hover {
        transform: scale(1.05);
        box-shadow: 0 10px 40px rgba(220, 20, 60, 0.4);
    }
    
    .stat-number {
        font-family: 'Orbitron', sans-serif;
        font-size: 3rem;
        font-weight: 900;
        color: #ff0000;
        display: block;
    }
    
    .stat-label {
        font-family: 'Rajdhani', sans-serif;
        font-size: 1.2rem;
        color: #aaaaaa;
        text-transform: uppercase;
        letter-spacing: 2px;
    }
    
    /* Chatbot Sidebar Styling */
    .css-1d391kg, [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0f0000 0%, #1a0000 100%);
        border-right: 3px solid #dc143c;
    }
    
    /* Chat Messages */
    .chat-message {
        padding: 1rem;
        margin: 0.5rem 0;
        border-radius: 10px;
        font-family: 'Rajdhani', sans-serif;
        font-size: 1.05rem;
        line-height: 1.5;
    }
    
    .user-message {
        background: rgba(220, 20, 60, 0.1);
        border-left: 4px solid #dc143c;
        color: #ffffff;
    }
    
    .bot-message {
        background: rgba(30, 30, 30, 0.8);
        border-left: 4px solid #00ff88;
        color: #e0e0e0;
    }
    
    /* Buttons */
    .stButton > button {
        font-family: 'Orbitron', sans-serif;
        background: linear-gradient(135deg, #dc143c 0%, #cc0000 100%);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 0.75rem 2rem;
        font-size: 1.1rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 2px;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(220, 20, 60, 0.3);
    }
    
    .stButton > button:hover {
        background: linear-gradient(135deg, #ff0000 0%, #dc143c 100%);
        transform: translateY(-2px);
        box-shadow: 0 6px 25px rgba(220, 20, 60, 0.5);
    }
    
    /* Input Fields */
    .stTextInput > div > div > input,
    .stTextArea > div > div > textarea {
        font-family: 'Rajdhani', sans-serif;
        background: rgba(20, 20, 20, 0.8);
        border: 2px solid #333;
        border-radius: 8px;
        color: #ffffff;
        font-size: 1.1rem;
        padding: 0.75rem;
    }
    
    .stTextInput > div > div > input:focus,
    .stTextArea > div > div > textarea:focus {
        border-color: #dc143c;
        box-shadow: 0 0 20px rgba(220, 20, 60, 0.3);
    }
    
    /* Sidebar Text */
    .css-1d391kg p, [data-testid="stSidebar"] p,
    .css-1d391kg label, [data-testid="stSidebar"] label {
        color: #e0e0e0;
        font-family: 'Rajdhani', sans-serif;
    }
    
    /* Success/Warning/Error Messages */
    .stSuccess, .stWarning, .stError, .stInfo {
        font-family: 'Rajdhani', sans-serif;
        border-radius: 8px;
    }
    
    /* Scrollbar */
    ::-webkit-scrollbar {
        width: 10px;
        background: #1a0000;
    }
    
    ::-webkit-scrollbar-track {
        background: #0f0000;
    }
    
    ::-webkit-scrollbar-thumb {
        background: linear-gradient(180deg, #dc143c 0%, #cc0000 100%);
        border-radius: 5px;
    }
    
    ::-webkit-scrollbar-thumb:hover {
        background: linear-gradient(180deg, #ff0000 0%, #dc143c 100%);
    }
    
    /* Feature Grid */
    .feature-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
        gap: 1.5rem;
        margin: 2rem 0;
    }
    
    .feature-item {
        background: rgba(20, 20, 20, 0.6);
        border: 2px solid #222;
        border-radius: 12px;
        padding: 1.5rem;
        text-align: center;
        transition: all 0.3s ease;
    }
    
    .feature-item:hover {
        border-color: #dc143c;
        transform: translateY(-5px);
        box-shadow: 0 10px 30px rgba(220, 20, 60, 0.2);
    }
    
    .feature-icon {
        font-size: 3rem;
        margin-bottom: 1rem;
    }
    
    .feature-title {
        font-family: 'Orbitron', sans-serif;
        color: #ff0000;
        font-size: 1.3rem;
        font-weight: 700;
        margin-bottom: 0.5rem;
    }
    
    .feature-desc {
        font-family: 'Rajdhani', sans-serif;
        color: #aaaaaa;
        font-size: 1rem;
        line-height: 1.5;
    }
</style>
""", unsafe_allow_html=True)

# ------------------------------------------------------------
# Sidebar - Chatbot Interface
# ------------------------------------------------------------
with st.sidebar:
    st.markdown("<h2 style='font-family: Orbitron; color: #dc143c; text-align: center; margin-bottom: 1.5rem;'>🏁 ASHWA AI ASSISTANT</h2>", unsafe_allow_html=True)
    
    # Check DB status
    checker = SystemChecker(PROJECT_ROOT)
    db_status = checker.db_info()
    
    if db_status["exists"]:
        st.success("✅ Knowledge Base Online")
        
        # Initialize chat history
        if "messages" not in st.session_state:
            st.session_state.messages = []
        
        # Display chat history
        st.markdown("---")
        for message in st.session_state.messages:
            if message["role"] == "user":
                st.markdown(f"<div class='chat-message user-message'>👤 {message['content']}</div>", unsafe_allow_html=True)
            else:
                st.markdown(f"<div class='chat-message bot-message'>🤖 {message['content']}</div>", unsafe_allow_html=True)
        
        # Chat input
        st.markdown("---")
        user_query = st.text_input("Ask about Ashwa Riders...", key="chat_input", placeholder="e.g., What services do you offer?")
        
        if st.button("Send", key="send_button", use_container_width=True):
            if user_query:
                # Add user message to history
                st.session_state.messages.append({"role": "user", "content": user_query})
                
                # Query the RAG system
                with st.spinner("🔍 Searching knowledge base..."):
                    try:
                        response = requests.post(
                            "http://127.0.0.1:8000/query",
                            json={"query": user_query},
                            timeout=300
                        ).json()
                        
                        status = response.get("status")
                        answer = response.get("answer", "I couldn't find an answer to that question.")
                        
                        # Add bot response to history
                        st.session_state.messages.append({"role": "assistant", "content": answer})
                        
                        # Rerun to update chat
                        st.rerun()
                        
                    except Exception as e:
                        st.error(f"Connection error: {str(e)}")
                        st.info("Make sure the RAG backend is running on http://127.0.0.1:8000")
        
        if st.button("Clear Chat", key="clear_chat", use_container_width=True):
            st.session_state.messages = []
            st.rerun()
    
    else:
        st.warning("⚠️ Knowledge Base Not Available")
        st.info("Please set up the RAG system first using the control panel.")

# ------------------------------------------------------------
# Main Content - Company Page
# ------------------------------------------------------------

# Hero Section
st.markdown("<h1 class='main-header'>ASHWA RIDERS</h1>", unsafe_allow_html=True)
st.markdown("<p class='tagline'>Conquer Every Terrain • Embrace the Wild</p>", unsafe_allow_html=True)

# Welcome Section
col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    st.markdown("""
    <div class='info-card'>
        <h3>🏁 Welcome to the Ultimate Off-Road Experience</h3>
        <p>
            At Ashwa Riders, we don't just ride – we dominate. Powered by our flagship 
            <strong style='color: #dc143c;'>Black Stallion ATV</strong> featuring a 410cc engine 
            with 12 HP and battle-tested Carlisle off-roading tyres, we're revolutionizing 
            motorsport adventure. From rugged mountain trails to amusement park thrills, 
            our 100+ hour tested machines deliver unmatched performance and safety.
        </p>
    </div>
    """, unsafe_allow_html=True)

# Stats Section
st.markdown("<h2 class='section-header'>⚡ By The Numbers</h2>", unsafe_allow_html=True)
stat_col1, stat_col2, stat_col3, stat_col4 = st.columns(4)

with stat_col1:
    st.markdown("""
    <div class='stat-box'>
        <span class='stat-number'>250+</span>
        <span class='stat-label'>ATVs Deployed</span>
    </div>
    """, unsafe_allow_html=True)

with stat_col2:
    st.markdown("""
    <div class='stat-box'>
        <span class='stat-number'>100+</span>
        <span class='stat-label'>Testing Hours</span>
    </div>
    """, unsafe_allow_html=True)

with stat_col3:
    st.markdown("""
    <div class='stat-box'>
        <span class='stat-number'>15+</span>
        <span class='stat-label'>Amusement Parks</span>
    </div>
    """, unsafe_allow_html=True)

with stat_col4:
    st.markdown("""
    <div class='stat-box'>
        <span class='stat-number'>410cc</span>
        <span class='stat-label'>Engine Power</span>
    </div>
    """, unsafe_allow_html=True)

# Services Section
st.markdown("<h2 class='section-header'>🏍️ Our Services</h2>", unsafe_allow_html=True)

st.markdown("""
<div class='feature-grid'>
    <div class='feature-item'>
        <div class='feature-icon'>🏔️</div>
        <div class='feature-title'>MOUNTAIN EXPEDITIONS</div>
        <div class='feature-desc'>Black Stallion ATVs conquering Himalayan terrain with power and precision</div>
    </div>
    <div class='feature-item'>
        <div class='feature-icon'>🏜️</div>
        <div class='feature-title'>DESERT RALLIES</div>
        <div class='feature-desc'>410cc powerhouse performance across sand dunes and desert landscapes</div>
    </div>
    <div class='feature-item'>
        <div class='feature-icon'>🎢</div>
        <div class='feature-title'>AMUSEMENT PARK RIDES</div>
        <div class='feature-desc'>Controlled off-road experiences for theme parks and entertainment venues</div>
    </div>
    <div class='feature-item'>
        <div class='feature-icon'>🏁</div>
        <div class='feature-title'>ATV RACING EVENTS</div>
        <div class='feature-desc'>Competitive motorsport championships with our tested fleet</div>
    </div>
    <div class='feature-item'>
        <div class='feature-icon'>🔧</div>
        <div class='feature-title'>CUSTOM MODIFICATIONS</div>
        <div class='feature-desc'>Performance upgrades and specialized builds for Black Stallion ATVs</div>
    </div>
    <div class='feature-item'>
        <div class='feature-icon'>🎓</div>
        <div class='feature-title'>ATV TRAINING ACADEMY</div>
        <div class='feature-desc'>Professional coaching from beginner to expert level riders</div>
    </div>
</div>
""", unsafe_allow_html=True)

# About Section
st.markdown("<h2 class='section-header'>🔥 The Ashwa Spirit</h2>", unsafe_allow_html=True)

col_about1, col_about2 = st.columns(2)

with col_about1:
    st.markdown("""
    <div class='info-card'>
        <h3>Our Mission</h3>
        <p>
            To revolutionize off-road motorsport with the Black Stallion - our flagship 410cc ATV 
            that's redefining adventure experiences. From extreme terrain expeditions to safe, 
            thrilling amusement park attractions, we're bringing world-class engineering and 
            100+ hours of rigorous testing to every ride.
        </p>
    </div>
    """, unsafe_allow_html=True)

with col_about2:
    st.markdown("""
    <div class='info-card'>
        <h3>Why Choose Us</h3>
        <p>
            <strong>The Black Stallion Advantage:</strong> 410cc engine with 12 HP raw power, 
            premium Carlisle off-roading tyres, 100+ hours extreme testing, proven reliability. 
            Now expanding into amusement markets - bringing motorsport thrills to theme parks 
            and entertainment venues across India with the safest, most powerful ATVs available.
        </p>
    </div>
    """, unsafe_allow_html=True)

# Gear Section
st.markdown("<h2 class='section-header'>⚙️ The Black Stallion - Our Flagship ATV</h2>", unsafe_allow_html=True)

col_spec1, col_spec2 = st.columns(2)

with col_spec1:
    st.markdown("""
    <div class='info-card'>
        <h3>🏎️ Technical Specifications</h3>
        <p>
            <strong>Engine:</strong> 410cc High-Performance<br>
            <strong>Power Output:</strong> 12 HP<br>
            <strong>Tyres:</strong> Carlisle Off-Roading Series<br>
            <strong>Testing:</strong> 100+ Hours Extreme Terrain Validation<br>
            <strong>Chassis:</strong> Reinforced Steel Frame<br>
            <strong>Suspension:</strong> Heavy-Duty All-Terrain
        </p>
    </div>
    """, unsafe_allow_html=True)

with col_spec2:
    st.markdown("""
    <div class='info-card'>
        <h3>🎯 Performance Features</h3>
        <p>
            Built to dominate the harshest terrains, the Black Stallion represents 
            the pinnacle of ATV engineering. With its robust 410cc engine delivering 
            12 HP of raw power and equipped with premium Carlisle off-roading tyres, 
            this beast has been rigorously tested for over 100 hours across mountains, 
            deserts, and extreme conditions.
        </p>
    </div>
    """, unsafe_allow_html=True)

st.markdown("<h2 class='section-header'>🎢 Expanding Into Amusement Markets</h2>", unsafe_allow_html=True)

st.markdown("""
<div class='info-card' style='text-align: center; padding: 2rem;'>
    <h3 style='font-size: 1.8rem; margin-bottom: 1rem;'>Taking Adventure to Theme Parks</h3>
    <p style='font-size: 1.2rem; margin-bottom: 1rem;'>
        The Black Stallion is making waves in the amusement industry! Leading theme parks 
        and adventure destinations are now introducing our ATVs for controlled off-road 
        experiences that bring motorsport thrills to family entertainment.
    </p>
    <p style='color: #cccccc; font-size: 1.1rem;'>
        <strong>🎪 Perfect for:</strong> Adventure Parks • Theme Parks • Resort Activities • 
        Corporate Team Building • Entertainment Complexes
    </p>
</div>
""", unsafe_allow_html=True)

# Additional Equipment Section
st.markdown("<h2 class='section-header'>🛠️ Support Equipment & Safety</h2>", unsafe_allow_html=True)

gear_col1, gear_col2, gear_col3 = st.columns(3)

with gear_col1:
    st.markdown("""
    <div class='info-card'>
        <h3>🚙 Support Vehicles</h3>
        <p>
            4x4 Mahindra Thar, Force Gurkha, Modified Gypsys 
            with rescue equipment and medical kits for safety backup
        </p>
    </div>
    """, unsafe_allow_html=True)

with gear_col2:
    st.markdown("""
    <div class='info-card'>
        <h3>🛡️ Safety Equipment</h3>
        <p>
            Professional helmets, riding suits, GPS trackers, 
            satellite phones, and trained medical support personnel
        </p>
    </div>
    """, unsafe_allow_html=True)

with gear_col3:
    st.markdown("""
    <div class='info-card'>
        <h3>🔧 Maintenance</h3>
        <p>
            On-site mechanics, spare parts inventory, 
            24/7 technical support for all Black Stallion units
        </p>
    </div>
    """, unsafe_allow_html=True)

# CTA Section
st.markdown("<h2 class='section-header'>🚀 Ready to Ride?</h2>", unsafe_allow_html=True)

col_cta1, col_cta2, col_cta3 = st.columns([1, 2, 1])
with col_cta2:
    st.markdown("""
    <div class='info-card' style='text-align: center; padding: 2rem;'>
        <h3 style='font-size: 1.8rem; margin-bottom: 1rem;'>Experience the Black Stallion</h3>
        <p style='font-size: 1.2rem; margin-bottom: 1.5rem;'>
            Whether you're an adventure seeker looking for extreme terrain challenges or 
            an amusement park operator seeking the ultimate attraction, the Black Stallion 
            delivers. 410cc of pure power, Carlisle tyres, and 100+ hours of proven performance.
        </p>
        <p style='color: #dc143c; font-size: 1.3rem; font-weight: 700;'>
            📞 +91 98765 43210 | 📧 ride@ashwariders.com
        </p>
    </div>
    """, unsafe_allow_html=True)

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; font-family: Rajdhani; color: #666; padding: 2rem 0;'>
    <p style='font-size: 1.1rem;'>
        <strong style='color: #dc143c;'>ASHWA RIDERS</strong> | Est. 2020 | 
        Black Stallion • 410cc • Carlisle Tyres • 100+ Hours Tested
    </p>
    <p style='font-size: 0.9rem; margin-top: 0.5rem;'>
        📍 Headquarters: Pune, Maharashtra | Now in Amusement Parks Across India
    </p>
</div>
""", unsafe_allow_html=True)