import streamlit as st

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="Ashwa Terrain Vehicles",
    page_icon="🏍️",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ---------------- SESSION STATE ----------------
if "messages" not in st.session_state:
    st.session_state.messages = []

if "dark_mode" not in st.session_state:
    st.session_state.dark_mode = True

# ---------------- CHATBOT LOGIC ----------------
def get_atv_response(user_message):
    """Intelligent ATV chatbot responses"""
    msg = user_message.lower()
    
    if any(word in msg for word in ['hi', 'hello', 'hey']):
        return "👋 Welcome to Ashwa Terrain Vehicles! I can help you learn about our ATVs, experience zones, and training programs. What would you like to know?"
    
    elif any(word in msg for word in ['atv', 'model', 'vehicle', 'product']):
        return """🏍️ **Our ATV Range:**

• **ATV 200** - Entry-level, 200cc, perfect for beginners
• **ATV 300** - Mid-range, 300cc, trail & endurance
• **ATV 450** - Advanced, 450cc, motorsport ready

Each model features automatic CVT, independent suspension, and speed-limited modes for safety."""
    
    elif any(word in msg for word in ['price', 'pricing', 'cost']):
        return """💰 **ATV Pricing:**

• ATV 200: ₹2,50,000 - ₹3,00,000
• ATV 300: ₹3,50,000 - ₹4,50,000
• ATV 450: ₹5,00,000 - ₹6,50,000

Contact us for bulk orders, corporate packages, and financing options!"""
    
    elif any(word in msg for word in ['zone', 'location', 'experience', 'where']):
        return """📍 **Experience Zones:**

✅ **Gujarat** - Flagship Zone (Operational)
🔜 **Rajasthan** - Desert Zone (Coming Soon)
🔜 **Maharashtra** - Western Ghats (Coming Soon)
🔜 **Karnataka** - Training & Motorsport (Coming Soon)

Book your test ride today!"""
    
    elif any(word in msg for word in ['safe', 'safety', 'training']):
        return """🛡️ **Safety First:**

• Mandatory safety briefing before rides
• Certified instructor supervision
• Speed-limited beginner modes
• Full protective gear provided (helmet, pads, gloves)
• Emergency response team on-site

Your safety is our top priority!"""
    
    elif any(word in msg for word in ['book', 'booking', 'reserve', 'test']):
        return """📅 **Book Your Experience:**

1. Choose your ATV model
2. Select preferred location
3. Pick a date & time
4. Complete online registration

📧 bookings@ashwaatv.com
📞 +91 80000 00000

Test rides available every weekend!"""
    
    elif any(word in msg for word in ['200', 'entry', 'beginner']):
        return """🏍️ **ATV 200 - Entry Level:**

• 200cc Petrol Engine
• Automatic CVT Transmission
• Top Speed: 60 km/h (limited)
• Perfect for training & parks
• Ideal for beginners
• Price: ₹2,50,000 - ₹3,00,000

Great choice for getting started!"""
    
    elif any(word in msg for word in ['300', 'mid', 'intermediate']):
        return """🏍️ **ATV 300 - Mid Range:**

• 300cc Petrol Engine
• Independent Suspension
• Top Speed: 80 km/h
• Trail & endurance use
• Balanced power & control
• Price: ₹3,50,000 - ₹4,50,000

Perfect for intermediate riders!"""
    
    elif any(word in msg for word in ['450', 'advanced', 'pro']):
        return """🏍️ **ATV 450 - Advanced:**

• 450cc High-Performance Engine
• Reinforced Chassis
• Top Speed: 110 km/h
• Motorsport academy ready
• Professional-grade suspension
• Price: ₹5,00,000 - ₹6,50,000

For experienced riders only!"""
    
    elif any(word in msg for word in ['corporate', 'event', 'group']):
        return """👥 **Corporate & Group Events:**

• Team-building off-road experiences
• Custom training programs
• Bulk ATV bookings
• Instructor-led sessions
• Catering & facilities available
• Group discounts available

Perfect for corporate off-sites!"""
    
    else:
        return """I can help you with:

• 🏍️ ATV Models & Specifications
• 💰 Pricing & Packages
• 📍 Experience Zone Locations
• 🛡️ Safety & Training Programs
• 📅 Bookings & Test Rides
• 👥 Corporate Events

What would you like to know?"""

# ---------------- THEME ----------------
if st.session_state.dark_mode:
    BG = "#050812"
    CARD = "#0b1020"
    TEXT = "#e6e9ff"
    BLUE = "#00e5ff"
    PURPLE = "#9b7cff"
    ACCENT = "rgba(0,229,255,0.12)"
else:
    BG = "#f4f6fb"
    CARD = "#ffffff"
    TEXT = "#050812"
    BLUE = "#2962ff"
    PURPLE = "#7c4dff"
    ACCENT = "rgba(124,77,255,0.12)"

# ---------------- CSS ----------------
st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700;800;900&display=swap');

* {{
    font-family: 'Inter', sans-serif;
}}

#MainMenu, footer, header {{ visibility: hidden; }}

.stApp {{
    background: {BG};
    color: {TEXT};
}}

/* Top Bar */
.top-bar {{
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 1rem 0;
    margin-bottom: 2rem;
    border-bottom: 2px solid rgba(155,124,255,0.2);
}}

/* Title */
.title {{
    font-size: 48px;
    font-weight: 900;
    background: linear-gradient(90deg, {BLUE}, {PURPLE});
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    margin-bottom: 0.5rem;
    letter-spacing: -1px;
}}

.subtitle {{
    color: {BLUE};
    font-size: 1.1rem;
    font-weight: 600;
    opacity: 0.9;
    margin-bottom: 2rem;
}}

/* Section */
.section {{
    margin-top: 2.5rem;
}}

/* Cards */
.card {{
    background: {CARD};
    border-radius: 20px;
    padding: 1.8rem;
    border: 1px solid rgba(155,124,255,0.3);
    box-shadow: 0 0 30px {ACCENT};
    transition: all 0.3s ease;
    height: 100%;
}}

.card:hover {{
    box-shadow: 0 0 40px rgba(0,229,255,0.25);
    transform: translateY(-5px);
}}

.card h3 {{
    color: {BLUE};
    font-weight: 700;
    margin-bottom: 1rem;
    font-size: 1.3rem;
}}

.card ul {{
    line-height: 1.9;
    padding-left: 1.5rem;
}}

.card ul li {{
    margin-bottom: 0.6rem;
}}

.card b {{
    color: {PURPLE};
    font-weight: 700;
}}

/* ATV Model Circles */
.circle-container {{
    display: flex;
    justify-content: center;
    gap: 2rem;
    margin: 2rem 0;
}}

.circle {{
    width: 180px;
    height: 180px;
    border-radius: 50%;
    background: linear-gradient(135deg, {BLUE}, {PURPLE});
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    text-align: center;
    font-weight: 800;
    color: #000;
    box-shadow: 0 0 40px rgba(155,124,255,0.6);
    animation: pulse 3s infinite;
    cursor: pointer;
    transition: all 0.3s ease;
}}

.circle:hover {{
    transform: scale(1.1);
    box-shadow: 0 0 60px rgba(0,229,255,0.9);
}}

@keyframes pulse {{
    0% {{ box-shadow: 0 0 30px rgba(155,124,255,0.5); }}
    50% {{ box-shadow: 0 0 50px rgba(0,229,255,1); }}
    100% {{ box-shadow: 0 0 30px rgba(155,124,255,0.5); }}
}}

.circle-title {{
    font-size: 1.8rem;
    margin-bottom: 0.3rem;
}}

.circle-subtitle {{
    font-size: 0.9rem;
    opacity: 0.8;
    font-weight: 600;
}}

/* Section Headers */
.section-header {{
    color: {BLUE};
    font-weight: 700;
    font-size: 1.8rem;
    margin-bottom: 1.5rem;
    display: flex;
    align-items: center;
    gap: 0.5rem;
}}

/* Info Box */
.info-box {{
    background: linear-gradient(135deg, rgba(0,229,255,0.15), rgba(155,124,255,0.15));
    border-left: 4px solid {PURPLE};
    border-radius: 12px;
    padding: 1.5rem;
    margin: 1.5rem 0;
}}

/* Theme Toggle */
.stButton button {{
    background: linear-gradient(135deg, {BLUE}, {PURPLE}) !important;
    color: #000 !important;
    border: none !important;
    border-radius: 50% !important;
    width: 50px !important;
    height: 50px !important;
    font-size: 24px !important;
    box-shadow: 0 0 20px rgba(155,124,255,0.5) !important;
    transition: all 0.3s ease !important;
}}

.stButton button:hover {{
    transform: scale(1.1) !important;
    box-shadow: 0 0 30px rgba(0,229,255,0.8) !important;
}}

/* Chat Input */
.stChatInput {{
    border-radius: 15px !important;
}}

.stChatInput input {{
    background: {CARD} !important;
    border: 2px solid rgba(155,124,255,0.3) !important;
    color: {TEXT} !important;
    border-radius: 15px !important;
}}

.stChatInput input:focus {{
    border-color: {BLUE} !important;
    box-shadow: 0 0 15px rgba(0,229,255,0.3) !important;
}}

/* Chat Messages */
.stChatMessage {{
    background: {CARD} !important;
    border: 1px solid rgba(155,124,255,0.2) !important;
    border-radius: 15px !important;
    margin: 0.8rem 0 !important;
    box-shadow: 0 2px 10px {ACCENT} !important;
}}

/* Responsive */
@media (max-width: 768px) {{
    .circle {{
        width: 140px;
        height: 140px;
    }}
    
    .title {{
        font-size: 32px;
    }}
}}
</style>
""", unsafe_allow_html=True)

# ---------------- TOP BAR ----------------
col1, col2 = st.columns([11, 1])
with col2:
    if st.button("🌙" if st.session_state.dark_mode else "☀️", key="theme_toggle"):
        st.session_state.dark_mode = not st.session_state.dark_mode
        st.rerun()

# ---------------- LAYOUT ----------------
chat_col, main_col = st.columns([3, 9])

# ================= LEFT CHAT PANEL =================
with chat_col:
    st.markdown('<div class="section-header">💬 Ashwa Bot</div>', unsafe_allow_html=True)
    
    st.markdown(f"""
    <div class="info-box">
        <strong style="color:{PURPLE};">Ask me anything!</strong><br>
        I can help with ATVs, zones, bookings, safety & more.
    </div>
    """, unsafe_allow_html=True)
    
    # Display chat messages
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
    
    # Chat input
    user_input = st.chat_input("Ask about ATVs, zones, models...")
    
    if user_input:
        # Add user message
        st.session_state.messages.append({"role": "user", "content": user_input})
        
        # Get bot response
        bot_reply = get_atv_response(user_input)
        
        # Add bot message
        st.session_state.messages.append({"role": "assistant", "content": bot_reply})
        
        st.rerun()

# ================= MAIN CONTENT =================
with main_col:
    
    # Header
    st.markdown('<div class="title">🏍️ ASHWA TERRAIN VEHICLES</div>', unsafe_allow_html=True)
    st.markdown('<div class="subtitle">ATV DIVISION · Built for Adventure, Engineered for Safety</div>', unsafe_allow_html=True)
    
    # Introduction
    st.markdown(f"""
    <div class="card">
        <h3>What Are Ashwa ATVs?</h3>
        <p>
        Ashwa Terrain Vehicles (ATVs) are purpose-built all-terrain machines designed for
        controlled off-road environments such as adventure parks, training tracks,
        farms, and recreational facilities.
        </p>
        <ul>
            <li><b>High Durability</b> - Built to withstand extreme terrain</li>
            <li><b>Low Maintenance</b> - Reliable & cost-effective operation</li>
            <li><b>Beginner-Friendly</b> - Easy learning curve with training modes</li>
            <li><b>Safety-First Design</b> - Speed limiters & protective features</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)
    
    # ATV Models Section
    st.markdown('<div class="section"><div class="section-header">🏍️ ATV Product Range</div></div>', unsafe_allow_html=True)
    
    # Model Circles
    m1, m2, m3 = st.columns(3)
    
    with m1:
        st.markdown("""
        <div style="display:flex; justify-content:center;">
            <div class="circle">
                <div class="circle-title">ATV 200</div>
                <div class="circle-subtitle">Entry Level</div>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    with m2:
        st.markdown("""
        <div style="display:flex; justify-content:center;">
            <div class="circle">
                <div class="circle-title">ATV 300</div>
                <div class="circle-subtitle">Mid Range</div>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    with m3:
        st.markdown("""
        <div style="display:flex; justify-content:center;">
            <div class="circle">
                <div class="circle-title">ATV 450</div>
                <div class="circle-subtitle">Advanced</div>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    # Model Details
    st.markdown('<div style="margin-top:2rem;"></div>', unsafe_allow_html=True)
    d1, d2, d3 = st.columns(3)
    
    with d1:
        st.markdown("""
        <div class="card">
            <h3>🟢 Ashwa ATV 200</h3>
            <ul>
                <li><b>200cc Petrol Engine</b></li>
                <li>Automatic CVT Transmission</li>
                <li>Top Speed: 60 km/h (limited)</li>
                <li>Training & park use</li>
                <li>Speed-limited modes</li>
                <li><b>Price: ₹2.5L - ₹3L</b></li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    with d2:
        st.markdown("""
        <div class="card">
            <h3>🟡 Ashwa ATV 300</h3>
            <ul>
                <li><b>300cc Petrol Engine</b></li>
                <li>Independent suspension</li>
                <li>Top Speed: 80 km/h</li>
                <li>Trail & endurance use</li>
                <li>All-terrain capability</li>
                <li><b>Price: ₹3.5L - ₹4.5L</b></li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    with d3:
        st.markdown("""
        <div class="card">
            <h3>🔴 Ashwa ATV 450</h3>
            <ul>
                <li><b>450cc High-Performance</b></li>
                <li>Reinforced chassis</li>
                <li>Top Speed: 110 km/h</li>
                <li>Motorsport academies</li>
                <li>Professional-grade</li>
                <li><b>Price: ₹5L - ₹6.5L</b></li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    # Experience Zones
    st.markdown('<div class="section"><div class="section-header">📍 Ashwa Experience Zones</div></div>', unsafe_allow_html=True)
    
    st.markdown(f"""
    <div class="card">
        <h3>Real Terrain, Real Experience</h3>
        <p>
        Dedicated off-road facilities where customers experience real terrain,
        not showroom floors. Each zone is designed to test vehicle capabilities
        and rider skills in a controlled, safe environment.
        </p>
        <ul>
            <li><b>Off-Road Trails</b> - Endurance & technical routes</li>
            <li><b>Water Crossings</b> - Shallow stream obstacles</li>
            <li><b>Articulation Zones</b> - Rock crawling & flex testing</li>
            <li><b>Instructor-Led Sessions</b> - Professional guidance</li>
            <li><b>Safety Gear Provided</b> - Helmets, pads, gloves</li>
            <li><b>Pre-Ride Briefing</b> - Mandatory safety training</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)
    
    # Locations
    st.markdown('<div class="section"><div class="section-header">🗺️ Experience Locations</div></div>', unsafe_allow_html=True)
    
    loc1, loc2 = st.columns(2)
    
    with loc1:
        st.markdown(f"""
        <div class="card">
            <h3>✅ Gujarat - Flagship Zone</h3>
            <p><b>Status: Operational</b></p>
            <ul>
                <li>25+ km of trails</li>
                <li>All vehicle categories</li>
                <li>Training programs available</li>
                <li>Weekend test rides</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    with loc2:
        st.markdown(f"""
        <div class="card">
            <h3>🔜 Coming Soon</h3>
            <ul>
                <li><b>Rajasthan</b> - Desert Zone</li>
                <li><b>Maharashtra</b> - Western Ghats</li>
                <li><b>Karnataka</b> - Training & Motorsport Hub</li>
            </ul>
            <p style="margin-top:1rem; opacity:0.8;">
            Stay tuned for launch announcements!
            </p>
        </div>
        """, unsafe_allow_html=True)
    
    # Safety & Training
    st.markdown('<div class="section"><div class="section-header">🛡️ Safety & Training</div></div>', unsafe_allow_html=True)
    
    safe1, safe2 = st.columns(2)
    
    with safe1:
        st.markdown("""
        <div class="card">
            <h3>Safety Protocols</h3>
            <ul>
                <li>Mandatory safety briefing</li>
                <li>Certified instructor supervision</li>
                <li>Speed-limited beginner modes</li>
                <li>Full protective gear provided</li>
                <li>Emergency response team on-site</li>
                <li>First-aid facilities available</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    with safe2:
        st.markdown("""
        <div class="card">
            <h3>Training Programs</h3>
            <ul>
                <li>Beginner ATV handling courses</li>
                <li>Intermediate trail riding</li>
                <li>Advanced off-road techniques</li>
                <li>Corporate team-building programs</li>
                <li>Youth motorsport training</li>
                <li>Instructor certification courses</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    # Community & Events
    st.markdown('<div class="section"><div class="section-header">🏆 Community & Events</div></div>', unsafe_allow_html=True)
    
    st.markdown(f"""
    <div class="card">
        <h3>Building India's Off-Road Culture</h3>
        <p>
        Ashwa builds long-term off-road culture through clubs, training camps,
        corporate off-sites, and youth motorsport programs. Join our growing
        community of adventure enthusiasts!
        </p>
        <ul>
            <li><b>Ashwa Riders Club</b> - Exclusive member benefits</li>
            <li><b>Monthly Rides</b> - Community trail rides</li>
            <li><b>Corporate Off-Sites</b> - Team-building adventures</li>
            <li><b>Youth Programs</b> - Motorsport development</li>
            <li><b>Annual Championships</b> - ATV racing events</li>
            <li><b>Skills Workshops</b> - Technical training sessions</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)
    
    # Footer CTA
    st.markdown(f"""
    <div class="info-box" style="margin-top:3rem; text-align:center;">
        <h3 style="color:{PURPLE}; margin-bottom:1rem;">Ready to Experience Ashwa ATVs?</h3>
        <p style="font-size:1.1rem;">
        📧 bookings@ashwaatv.com | 📞 +91 80000 00000<br>
        🌐 www.ashwaatv.com
        </p>
        <p style="margin-top:1rem; opacity:0.8;">
        Visit our Gujarat flagship zone for test rides every weekend!
        </p>
    </div>
    """, unsafe_allow_html=True)